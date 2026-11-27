"use strict";

const $ = (sel) => document.querySelector(sel);
const NATIVE = !!window.Capacitor?.isNativePlatform?.();
const MAX_STAGES = 12;
const LETTERS = "ABCD";

const nodes = [];

// ---------------------------------------------------------------- cadeia

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

const allStages = () => nodes.flatMap((n) => (n.type === "split" ? n.branches.flatMap((b) => b.stages) : [n]));
const newBranch = () => ({ stages: [], level_db: -6, invert: false, enabled: true });

// Qualquer ajuste torna a prévia antiga.
function changed() {
  stopPreview();
  $("#preview-msg").textContent = "";
  $("#save-msg").textContent = "";
}

function update() {
  changed();
  renderChain();
}

function slider(target, key, label, min, max) {
  const text = () => `${label}: ${target[key] > 0 ? "+" : ""}${target[key]} dB`;
  const caption = el("span", { textContent: text() });
  const input = el("input", { type: "range", min, max, step: 0.5 });
  input.value = target[key];
  input.addEventListener("input", () => {
    target[key] = Number(input.value);
    caption.textContent = text();
    changed();
  });
  return el("label", {}, caption, input);
}

function iconButton(text, label, disabled, onClick) {
  const b = el("button", { type: "button", textContent: text, disabled });
  b.setAttribute("aria-label", label);
  b.addEventListener("click", onClick);
  return b;
}

function itemHeader(list, i, title) {
  const move = (to) => { list.splice(to, 0, ...list.splice(i, 1)); update(); };
  return el("div", { className: "stage-top" },
    el("span", { className: "stage-name", textContent: title }),
    iconButton("↑", `Subir ${title}`, i === 0, () => move(i - 1)),
    iconButton("↓", `Descer ${title}`, i === list.length - 1, () => move(i + 1)),
    iconButton("✕", `Remover ${title}`, false, () => { list.splice(i, 1); update(); }));
}

function stageItem(list, i) {
  const s = list[i];
  return el("li", { className: `stage ${s.type}` },
    itemHeader(list, i, s.fileName.replace(/\.(nam|wav)$/i, "")),
    slider(s, "input_gain_db", "Entrada", -24, 24),
    slider(s, "output_gain_db", "Saída", -24, 24));
}

function toggle(target, key, label, inverted = false) {
  const box = el("input", { type: "checkbox", checked: inverted ? !target[key] : target[key] });
  box.addEventListener("change", () => {
    target[key] = inverted ? !box.checked : box.checked;
    update();
  });
  return el("label", {}, box, label);
}

function branchBox(split, k) {
  const b = split.branches[k];
  const head = el("div", { className: "stage-top" },
    el("span", { className: "stage-name", textContent: `Cadeia ${LETTERS[k]}` }));
  if (split.branches.length > 2) {
    head.append(iconButton("✕", `Remover cadeia ${LETTERS[k]}`, false, () => {
      split.branches.splice(k, 1);
      update();
    }));
  }
  const list = el("ol", { className: "chain" }, ...b.stages.map((_, i) => stageItem(b.stages, i)));
  if (!b.stages.length) {
    list.append(el("li", { className: "empty", textContent: "Vazia: passa o sinal limpo." }));
  }
  return el("div", { className: `branch${b.enabled ? "" : " muted"}` },
    head,
    slider(b, "level_db", "Volume na mistura", -24, 12),
    el("div", { className: "switches" },
      toggle(b, "enabled", "Mudo", true),
      toggle(b, "invert", "Inverter fase")),
    list,
    addButtons(b.stages));
}

function splitItem(i) {
  const split = nodes[i];
  const box = el("li", { className: "split" },
    itemHeader(nodes, i, "Cadeias paralelas"),
    ...split.branches.map((_, k) => branchBox(split, k)));
  if (split.branches.length < LETTERS.length) {
    const more = el("button", { type: "button", className: "ghost wide", textContent: "+ Outra cadeia paralela" });
    more.addEventListener("click", () => { split.branches.push(newBranch()); update(); });
    box.append(more);
  }
  return box;
}

function fileButton(text, type, list) {
  const input = el("input", { type: "file", multiple: true, hidden: true });
  input.addEventListener("change", () => addFiles(input, type, list));
  return el("label", { className: "button" }, text, input);
}

function addButtons(list) {
  const main = list === nodes;
  return el("div", { className: "add" },
    fileButton(main ? "+ Pedal ou amp (.nam)" : "+ Pedal/amp", "nam", list),
    fileButton(main ? "+ Gabinete (IR .wav)" : "+ Gabinete", "ir", list));
}

function renderChain() {
  const list = $("#chain");
  list.replaceChildren(...nodes.map((n, i) => (n.type === "split" ? splitItem(i) : stageItem(nodes, i))));
  if (!nodes.length) {
    list.append(el("li", { className: "empty", textContent: "Adicione os pedais, o amp e o gabinete." }));
  }
  const add = addButtons(nodes);
  add.id = "add";
  if (!nodes.some((n) => n.type === "split")) {
    const split = el("button", { type: "button", textContent: "+ Dividir em cadeias paralelas" });
    split.addEventListener("click", () => {
      nodes.push({ type: "split", branches: [newBranch(), newBranch()] });
      update();
    });
    add.append(split);
  }
  $("#add").replaceWith(add);
}

// O seletor do Android não filtra bem por extensão, então a conferência é aqui.
async function addFiles(input, type, list) {
  const ext = type === "nam" ? ".nam" : ".wav";
  const problems = [];
  for (const file of input.files) {
    if (!file.name.toLowerCase().endsWith(ext)) {
      problems.push(`${file.name}: escolha um arquivo ${ext}.`);
      continue;
    }
    if (allStages().length >= MAX_STAGES) {
      problems.push(`A cadeia pode ter no máximo ${MAX_STAGES} itens.`);
      break;
    }
    const data = new Uint8Array(await file.arrayBuffer());
    list.push({ type, fileName: file.name, data, input_gain_db: 0, output_gain_db: 0 });
  }
  input.value = "";
  $("#chain-msg").textContent = problems.join(" ");
  update();
}

function chainProblem() {
  if (!allStages().length) return "Adicione pelo menos um pedal, amp ou gabinete.";
  const split = nodes.find((n) => n.type === "split");
  if (split && !split.branches.some((b) => b.enabled)) return "Todas as cadeias paralelas estão no mudo.";
  return "";
}

// ---------------------------------------------------------------- prévia

let audio = null;
let source = null;
let worker = null;
let generation = 0; // descarta respostas de uma prévia que já foi cancelada

function stopPreview() {
  generation++;
  worker?.terminate();
  worker = null;
  if (source) {
    source.onended = null;
    source.stop();
    source = null;
  }
  $("#preview-bar").hidden = true;
  $("#stop").hidden = true;
}

function play(samples) {
  // Volume normalizado só para ouvir, como no app de computador.
  let peak = 0;
  for (const v of samples) peak = Math.max(peak, Math.abs(v));
  const buffer = audio.createBuffer(1, samples.length, NamEngine.SAMPLE_RATE);
  const out = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i++) out[i] = (0.89 * samples[i]) / (peak || 1);
  source = audio.createBufferSource();
  source.buffer = buffer;
  source.connect(audio.destination);
  source.onended = () => { source = null; $("#stop").hidden = true; };
  source.start();
  $("#stop").hidden = false;
}

async function loadDi() {
  const res = await fetch($("#di").value);
  if (!res.ok) throw new Error("não achei o áudio de teste");
  return NamEngine.readWav(new Uint8Array(await res.arrayBuffer())).samples;
}

async function preview(withChain) {
  const msg = $("#preview-msg");
  stopPreview();
  msg.className = "msg";
  msg.textContent = "";
  const problem = withChain ? chainProblem() : "";
  if (problem) {
    msg.textContent = problem;
    return;
  }
  // O áudio precisa ser liberado já no toque, antes de qualquer espera.
  audio ||= new AudioContext();
  audio.resume();
  const mine = generation;
  try {
    const input = await loadDi();
    if (mine !== generation) return;
    if (!withChain) return play(input);
    const bar = $("#preview-bar");
    bar.hidden = false;
    bar.value = 0;
    $("#stop").hidden = false;
    msg.className = "msg ok";
    msg.textContent = "Processando a cadeia...";
    worker = new Worker("preview-worker.js");
    const fail = (text) => {
      stopPreview();
      msg.className = "msg";
      msg.textContent = text;
    };
    worker.onerror = () => fail("Não foi possível processar a prévia neste aparelho.");
    worker.onmessage = ({ data }) => {
      if (mine !== generation) return;
      if (data.error) return fail(data.error);
      if (data.progress !== undefined) {
        bar.value = data.progress;
        return;
      }
      bar.hidden = true;
      worker.terminate();
      worker = null;
      msg.textContent = data.peak >= 0.98
        ? "Tocando. A saída passou do limite; no treino o volume é reduzido automaticamente."
        : "Tocando.";
      play(data.y);
    };
    worker.postMessage({ input, nodes });
  } catch (e) {
    msg.className = "msg";
    msg.textContent = `Falha na prévia: ${e.message}`;
  }
}

// ---------------------------------------------------------------- pacote

function toBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

// A WebView não baixa arquivos sozinha: grava e abre o "compartilhar" do
// Android (salvar em Arquivos ou no Drive).
async function shareNative(blob, filename) {
  const { Filesystem, Share } = window.Capacitor.Plugins;
  const { uri } = await Filesystem.writeFile({ path: filename, data: await toBase64(blob), directory: "CACHE" });
  await Share.share({ title: filename, files: [uri] });
}

async function savePackage() {
  const msg = $("#save-msg");
  msg.className = "msg";
  msg.textContent = chainProblem();
  if (msg.textContent) return;
  const button = $("#save");
  button.disabled = true;
  try {
    const name = $("#name").value.trim() || "Minha cadeia";
    const filename = `${name.replace(/[\\/:*?"<>|]+/g, "_")}.chaincap`;
    const blob = new Blob([ChainPackage.buildPackage(name, nodes)], { type: "application/octet-stream" });
    if (NATIVE) {
      await shareNative(blob, filename);
      msg.textContent = `Pacote “${filename}” gerado. Confira se ele foi salvo em Arquivos ou no Drive.`;
    } else {
      const a = el("a", { href: URL.createObjectURL(blob), download: filename });
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10000);
      msg.textContent = `“${filename}” foi para a pasta de downloads.`;
    }
    msg.className = "msg ok";
  } catch (e) {
    msg.textContent = /cancel/i.test(e.message)
      ? "Compartilhamento cancelado: o pacote não foi salvo."
      : `Não foi possível salvar: ${e.message}`;
  } finally {
    button.disabled = false;
  }
}

// ---------------------------------------------------------------- início

$("#play").addEventListener("click", () => preview(true));
$("#play-di").addEventListener("click", () => preview(false));
$("#stop").addEventListener("click", () => { stopPreview(); $("#preview-msg").textContent = ""; });
$("#di").addEventListener("change", changed);
$("#save").addEventListener("click", savePackage);
renderChain();
