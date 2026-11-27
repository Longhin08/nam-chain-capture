// Prévia da cadeia no próprio aparelho: mesma conta de poc/chaincap/chain.py.
// Suporta WaveNet (A1 clássico e A2) e IR; usado no Web Worker e nos testes (Node).
(function (root) {
  "use strict";

  const SAMPLE_RATE = 48000;
  const PEAK_LIMIT = 0.98;
  const CHUNK = 1 << 14; // blocos pequenos cabem no cache do processador

  class PreviewError extends Error {}

  const dbToGain = (db) => 10 ** (db / 20);

  // ------------------------------------------------- memória e kernel SIMD

  // Os buffers temporários vivem numa área fixa que o WebAssembly enxerga.
  // Sem SIMD (aparelho muito antigo), a mesma conta roda em JavaScript.
  const kernel = (() => {
    try {
      const base64 = typeof module !== "undefined" ? require("./kernel-wasm.js") : root.KERNEL_WASM;
      const bytes = Uint8Array.from(atob(base64), (c) => c.charCodeAt(0));
      const memory = new WebAssembly.Memory({ initial: 512, maximum: 512 }); // 32 MB
      const instance = new WebAssembly.Instance(new WebAssembly.Module(bytes), { env: { memory } });
      return { buffer: memory.buffer, mac4: instance.exports.mac4 };
    } catch {
      return null;
    }
  })();
  let used = 0;

  function alloc(len) {
    if (kernel && used + len * 4 <= kernel.buffer.byteLength) {
      const view = new Float32Array(kernel.buffer, used, len);
      used += Math.ceil(len / 4) * 16;
      return view.fill(0);
    }
    return new Float32Array(len);
  }

  // ---------------------------------------------------------------- WaveNet

  function activationOf(spec) {
    const name = typeof spec === "string" ? spec : spec?.type || spec?.name;
    switch (name) {
      case "Tanh": return Math.tanh;
      case "ReLU": return (v) => (v > 0 ? v : 0);
      case "LeakyReLU": {
        const slope = spec.negative_slope ?? 0.01;
        return (v) => (v > 0 ? v : slope * v);
      }
      case "Sigmoid": return (v) => 1 / (1 + Math.exp(-v));
      case "Hardtanh": return (v) => Math.max(-1, Math.min(1, v));
      case "SiLU": return (v) => v / (1 + Math.exp(-v));
      default: throw new PreviewError(`ativação ${name} sem suporte na prévia`);
    }
  }

  const at = (v, i) => (Array.isArray(v) ? v[i] : v);

  function unsupported(layer) {
    const film = Object.keys(layer).some((k) => k.endsWith("_film") && layer[k]?.active);
    return film || layer.head1x1?.active || layer.layer1x1?.active === false ||
      (layer.groups_input ?? 1) !== 1 || (layer.groups_input_mixin ?? 1) !== 1 ||
      (layer.layer1x1?.groups ?? 1) !== 1 || layer.input_size === undefined ||
      (layer.condition_size ?? 1) !== 1;
  }

  // Lê os pesos na mesma ordem em que o NAM os exporta.
  function buildWaveNet(config, weights) {
    if (config.head || config.condition_dsp) throw new PreviewError("modelo sem suporte na prévia");
    let pos = 0;
    const take = (n) => {
      if (pos + n > weights.length) throw new PreviewError("arquivo .nam incompleto");
      const w = Float32Array.from(weights.slice(pos, pos + n));
      pos += n;
      return w;
    };
    const conv = (inCh, outCh, kernel, dilation, bias) => ({
      inCh, outCh, kernel, dilation,
      w: take(inCh * outCh * kernel),
      b: bias ? take(outCh) : null,
    });

    let receptive = 1;
    const arrays = config.layers.map((cfg) => {
      if (unsupported(cfg)) throw new PreviewError("modelo com recursos sem suporte na prévia");
      const channels = cfg.channels;
      const bottleneck = cfg.bottleneck ?? channels;
      const legacy = cfg.head === undefined;
      const head = legacy
        ? { out_channels: cfg.head_size, kernel_size: 1, bias: cfg.head_bias }
        : cfg.head;
      const rechannel = conv(cfg.input_size, channels, 1, 1, false);
      const layers = cfg.dilations.map((dilation, i) => {
        const kernel = at(cfg.kernel_sizes ?? cfg.kernel_size, i);
        let activation = at(cfg.activation, i);
        let secondary = null;
        if (legacy ? cfg.gated : false) secondary = "Sigmoid";
        const mode = at(cfg.gating_mode, i) ?? "none";
        if (mode === "gated") secondary = at(cfg.secondary_activation, i);
        else if (mode !== "none") throw new PreviewError(`gating ${mode} sem suporte na prévia`);
        if (activation?.name === "PairMultiply") {
          secondary = activation.secondary;
          activation = activation.primary;
        }
        const mid = secondary ? 2 * bottleneck : bottleneck;
        receptive += (kernel - 1) * dilation;
        return {
          conv: conv(channels, mid, kernel, dilation, true),
          mixer: conv(1, mid, 1, 1, false),
          out: conv(bottleneck, channels, 1, 1, true),
          act: activationOf(activation),
          gate: secondary ? activationOf(secondary) : null,
          bottleneck,
        };
      });
      const headConv = conv(bottleneck, head.out_channels, head.kernel_size, 1, !!head.bias);
      receptive += head.kernel_size - 1;
      return { channels, bottleneck, rechannel, layers, head: headConv };
    });
    const headScale = weights[pos];
    if (pos + 1 !== weights.length) throw new PreviewError("formato de .nam sem suporte na prévia");
    return { arrays, headScale, receptive };
  }

  const channelsOf = (n, len) => Array.from({ length: n }, () => alloc(len));

  // Quatro parcelas por passada: y[t] += w[j] * x[j][t - atraso[j]].
  const xs = new Array(4);
  const ws = new Float32Array(4);
  const ds = new Int32Array(4);

  function mac4(y, len) {
    const start = Math.min(len, Math.max(ds[0], ds[1], ds[2], ds[3]));
    for (let j = 0; j < 4; j++) { // começo: só as parcelas que já têm passado
      const x = xs[j], w = ws[j], d = ds[j];
      for (let t = d; t < start; t++) y[t] += w * x[t - d];
    }
    if (start >= len) return;
    const [a, b, c, e] = xs;
    const w0 = ws[0], w1 = ws[1], w2 = ws[2], w3 = ws[3];
    const d0 = start - ds[0], d1 = start - ds[1], d2 = start - ds[2], d3 = start - ds[3];
    const mem = kernel?.buffer;
    if (mem && y.buffer === mem && a.buffer === mem && b.buffer === mem && c.buffer === mem && e.buffer === mem) {
      kernel.mac4(y.byteOffset + 4 * start, a.byteOffset + 4 * d0, b.byteOffset + 4 * d1,
        c.byteOffset + 4 * d2, e.byteOffset + 4 * d3, w0, w1, w2, w3, len - start);
      return;
    }
    for (let n = 0; n < len - start; n++) {
      y[start + n] += w0 * a[d0 + n] + w1 * b[d1 + n] + w2 * c[d2 + n] + w3 * e[d3 + n];
    }
  }

  // out += convolução causal de `input`.
  function convolve(c, input, out, len) {
    const { inCh, outCh, kernel: size, dilation, w, b } = c;
    for (let o = 0; o < outCh; o++) {
      const y = out[o];
      if (b) {
        const bias = b[o];
        for (let t = 0; t < len; t++) y[t] += bias;
      }
      let count = 0;
      for (let i = 0; i < inCh; i++) {
        for (let k = 0; k < size; k++) {
          xs[count] = input[i];
          ws[count] = w[(o * inCh + i) * size + k];
          ds[count] = dilation * (size - 1 - k);
          if (++count === 4) {
            mac4(y, len);
            count = 0;
          }
        }
      }
      if (count) {
        for (let j = count; j < 4; j++) { // completa o grupo com parcelas nulas
          xs[j] = xs[0];
          ws[j] = 0;
          ds[j] = ds[0];
        }
        mac4(y, len);
      }
    }
  }

  const clear = (channels) => channels.forEach((c) => c.fill(0));

  // `cond`: bloco de entrada (com contexto). Devolve a saída do bloco inteiro.
  function forward(net, block) {
    used = 0;
    const len = block.length;
    const cond = alloc(len);
    cond.set(block);
    const input = [cond];
    let signal = input;
    let headIn = null;
    for (const array of net.arrays) {
      let x = channelsOf(array.channels, len);
      convolve(array.rechannel, signal, x, len);
      const sum = headIn || channelsOf(array.bottleneck, len);
      const z = channelsOf(array.layers[0].conv.outCh, len);
      const post = channelsOf(array.bottleneck, len);
      let next = channelsOf(array.channels, len);
      for (const layer of array.layers) {
        clear(z);
        convolve(layer.conv, x, z, len);
        convolve(layer.mixer, input, z, len);
        for (let ch = 0; ch < layer.bottleneck; ch++) {
          const a = z[ch], p = post[ch], s = sum[ch], act = layer.act;
          if (layer.gate) {
            const g = z[ch + layer.bottleneck], gate = layer.gate;
            for (let t = 0; t < len; t++) p[t] = act(a[t]) * gate(g[t]);
          } else {
            for (let t = 0; t < len; t++) p[t] = act(a[t]);
          }
          for (let t = 0; t < len; t++) s[t] += p[t];
        }
        for (let ch = 0; ch < array.channels; ch++) next[ch].set(x[ch]);
        convolve(layer.out, post, next, len);
        [x, next] = [next, x];
      }
      headIn = channelsOf(array.head.outCh, len);
      convolve(array.head, sum, headIn, len);
      signal = x;
    }
    const y = headIn[0];
    for (let t = 0; t < len; t++) y[t] *= net.headScale;
    return y;
  }

  function parseNam(data) {
    let model;
    try {
      model = JSON.parse(typeof data === "string" ? data : new TextDecoder().decode(data));
    } catch {
      throw new PreviewError("não é um arquivo .nam válido");
    }
    if (model.sample_rate && Math.round(model.sample_rate) !== SAMPLE_RATE) {
      throw new PreviewError(`modelo em ${model.sample_rate} Hz (só 48 kHz)`);
    }
    if (model.architecture === "SlimmableContainer") {
      // Como no computador: o submodelo de maior qualidade.
      const subs = model.config.submodels;
      model = subs.reduce((a, b) => ((b.max_value || 0) > (a.max_value || 0) ? b : a)).model;
    }
    if (model.architecture !== "WaveNet") {
      throw new PreviewError(`modelo ${model.architecture} sem prévia (o treino no Colab não depende dela)`);
    }
    return buildWaveNet(model.config, model.weights);
  }

  // Mesmo comprimento da entrada, causal, com o passado em silêncio.
  function namInfer(net, x, onProgress) {
    const context = net.receptive - 1;
    const padded = new Float32Array(context + x.length);
    padded.set(x, context);
    const out = new Float32Array(x.length);
    for (let start = 0; start < x.length; start += CHUNK) {
      const end = Math.min(start + CHUNK, x.length);
      out.set(forward(net, padded.subarray(start, end + context)).subarray(context), start);
      onProgress?.(end / x.length);
    }
    return out;
  }

  // ---------------------------------------------------------------- WAV e IR

  function readWav(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const tag = (p) => String.fromCharCode(...bytes.subarray(p, p + 4));
    if (bytes.length < 44 || tag(0) !== "RIFF" || tag(8) !== "WAVE") {
      throw new PreviewError("não é um arquivo WAV");
    }
    let fmt = null, data = null;
    for (let p = 12; p + 8 <= bytes.length;) {
      const size = view.getUint32(p + 4, true);
      if (tag(p) === "fmt ") {
        let format = view.getUint16(p + 8, true);
        if (format === 0xfffe) format = view.getUint16(p + 32, true); // WAVE_FORMAT_EXTENSIBLE
        fmt = { format, channels: view.getUint16(p + 10, true), rate: view.getUint32(p + 12, true),
          bits: view.getUint16(p + 22, true) };
      } else if (tag(p) === "data") {
        data = { start: p + 8, size: Math.min(size, bytes.length - p - 8) };
      }
      p += 8 + size + (size & 1);
    }
    if (!fmt || !data) throw new PreviewError("WAV incompleto");
    const width = fmt.bits / 8;
    const frame = width * fmt.channels;
    const n = Math.floor(data.size / frame);
    let read;
    if (fmt.format === 3 && fmt.bits === 32) read = (p) => view.getFloat32(p, true);
    else if (fmt.format === 3 && fmt.bits === 64) read = (p) => view.getFloat64(p, true);
    else if (fmt.format === 1 && fmt.bits === 16) read = (p) => view.getInt16(p, true) / 32768;
    else if (fmt.format === 1 && fmt.bits === 24) {
      read = (p) => ((bytes[p] | (bytes[p + 1] << 8) | (bytes[p + 2] << 24 >> 8))) / 8388608;
    } else if (fmt.format === 1 && fmt.bits === 32) read = (p) => view.getInt32(p, true) / 2147483648;
    else throw new PreviewError("formato de WAV sem suporte");
    const samples = new Float32Array(n);
    for (let i = 0; i < n; i++) samples[i] = read(data.start + i * frame); // só o 1º canal
    return { samples, rate: fmt.rate };
  }

  // Sinc com janela. Só para IR fora de 48 kHz; difere um pouco do scipy.
  function resample(x, from, to) {
    const ratio = to / from;
    const cutoff = Math.min(1, ratio);
    const half = Math.ceil(16 / cutoff);
    const out = new Float32Array(Math.ceil(x.length * ratio));
    for (let n = 0; n < out.length; n++) {
      const center = n / ratio;
      let acc = 0;
      for (let k = Math.ceil(center - half); k <= center + half; k++) {
        if (k < 0 || k >= x.length) continue;
        const d = center - k;
        const s = d === 0 ? 1 : Math.sin(Math.PI * cutoff * d) / (Math.PI * cutoff * d);
        acc += x[k] * s * (0.5 + 0.5 * Math.cos((Math.PI * d) / (half + 1)));
      }
      out[n] = acc * cutoff;
    }
    return out;
  }

  function loadIr(bytes, maxMs, normalize) {
    const wav = readWav(bytes);
    let h = wav.rate === SAMPLE_RATE ? wav.samples : resample(wav.samples, wav.rate, SAMPLE_RATE);
    if (maxMs != null) h = h.slice(0, Math.round((maxMs * SAMPLE_RATE) / 1000));
    if (normalize) {
      let energy = 0;
      for (const v of h) energy += v * v;
      if (!energy) throw new PreviewError("o IR está todo em silêncio");
      const g = 1 / Math.sqrt(energy);
      h = h.map((v) => v * g);
    }
    return h;
  }

  function applyIr(x, h) {
    used = 0;
    const input = alloc(x.length);
    input.set(x);
    const out = alloc(x.length);
    convolve({ inCh: 1, outCh: 1, kernel: h.length, dilation: 1, b: null, w: h.slice().reverse() },
      [input], [out], x.length);
    return Float32Array.from(out); // a área temporária é reaproveitada
  }

  // ---------------------------------------------------------------- cadeia

  const gain = (x, g) => x.map((v) => v * g);

  function runStage(x, stage, onProgress) {
    try {
      let y = gain(x, dbToGain(stage.input_gain_db));
      if (stage.type === "nam") y = namInfer(parseNam(stage.data), y, onProgress);
      else y = applyIr(y, loadIr(stage.data, stage.ir_max_ms ?? 20, stage.ir_normalize !== false));
      onProgress?.(1);
      return gain(y, dbToGain(stage.output_gain_db));
    } catch (e) {
      if (e instanceof PreviewError) throw new PreviewError(`${stage.fileName}: ${e.message}`);
      throw e;
    }
  }

  function countStages(nodes) {
    return nodes.reduce((n, node) => n + (node.type === "split"
      ? node.branches.filter((b) => b.enabled).reduce((m, b) => m + b.stages.length, 0)
      : 1), 0);
  }

  // `nodes`: mesma estrutura do chain.json, com os bytes de cada arquivo em `data`.
  function renderChain(input, nodes, onProgress) {
    const total = Math.max(1, countStages(nodes));
    let done = 0;
    const run = (x, stage) => {
      const y = runStage(x, stage, (f) => onProgress?.((done + f) / total));
      done++;
      return y;
    };
    let y = input;
    for (const node of nodes) {
      if (node.type !== "split") {
        y = run(y, node);
        continue;
      }
      const live = node.branches.filter((b) => b.enabled);
      if (!live.length) throw new PreviewError("todas as cadeias paralelas estão no mudo");
      const mixed = new Float32Array(y.length);
      for (const branch of live) {
        const out = branch.stages.reduce(run, y); // sem estágios = sinal limpo
        const g = dbToGain(branch.level_db) * (branch.invert ? -1 : 1);
        for (let t = 0; t < mixed.length; t++) mixed[t] += g * out[t];
      }
      y = mixed;
    }
    let peak = 0;
    for (const v of y) {
      if (!Number.isFinite(v)) throw new PreviewError("a cadeia gerou um som inválido");
      peak = Math.max(peak, Math.abs(v));
    }
    if (peak === 0) throw new PreviewError("a cadeia produziu silêncio");
    if (peak >= PEAK_LIMIT) y = gain(y, PEAK_LIMIT / peak);
    return { y, peak };
  }

  const api = { renderChain, parseNam, namInfer, readWav, loadIr, PreviewError, SAMPLE_RATE };
  if (typeof module !== "undefined") module.exports = api;
  else root.NamEngine = api;
})(this);
