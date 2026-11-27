// Monta o pacote .chaincap (zip sem compressão) no próprio aparelho.
// Mesmo formato de poc/chaincap/colab.py. Sem dependências; roda também no Node (testes).
(function (root) {
  "use strict";

  const CRC = new Uint32Array(256).map((_, n) => {
    for (let k = 0; k < 8; k++) n = n & 1 ? 0xedb88320 ^ (n >>> 1) : n >>> 1;
    return n >>> 0;
  });

  function crc32(bytes) {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 255] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }

  function header(size, fields) {
    const buf = new Uint8Array(size);
    const view = new DataView(buf.buffer);
    let at = 0;
    for (const [bytes, value] of fields) {
      if (bytes === 2) view.setUint16(at, value, true);
      else view.setUint32(at, value, true);
      at += bytes;
    }
    return buf;
  }

  // entries: [{name, data: Uint8Array}] -> Uint8Array
  function zip(entries) {
    const parts = [];
    const central = [];
    let offset = 0;
    const UTF8 = 0x0800;
    const DATE = 0x0021; // 1980-01-01
    for (const { name, data } of entries) {
      const n = new TextEncoder().encode(name);
      const crc = crc32(data);
      const common = [[2, 20], [2, UTF8], [2, 0], [2, 0], [2, DATE],
        [4, crc], [4, data.length], [4, data.length], [2, n.length], [2, 0]];
      parts.push(header(30, [[4, 0x04034b50], ...common]), n, data);
      central.push(header(46, [[4, 0x02014b50], [2, 20], ...common,
        [2, 0], [2, 0], [2, 0], [4, 0], [4, offset]]), n);
      offset += 30 + n.length + data.length;
    }
    const size = central.reduce((s, p) => s + p.length, 0);
    const end = header(22, [[4, 0x06054b50], [2, 0], [2, 0], [2, entries.length],
      [2, entries.length], [4, size], [4, offset], [2, 0]]);
    const all = [...parts, ...central, end];
    const out = new Uint8Array(all.reduce((s, p) => s + p.length, 0));
    let at = 0;
    for (const p of all) { out.set(p, at); at += p.length; }
    return out;
  }

  function safeName(name, used) {
    const clean = name.replace(/[\\/:*?"<>|\u0000-\u001f]+/g, "_").trim() || "captura";
    const dot = clean.lastIndexOf(".");
    const stem = (dot > 0 ? clean.slice(0, dot) : clean).slice(0, 80);
    const ext = dot > 0 ? clean.slice(dot).toLowerCase() : "";
    let out = stem + ext;
    for (let i = 2; used.has(out.toLowerCase()); i++) out = `${stem} (${i})${ext}`;
    used.add(out.toLowerCase());
    return out;
  }

  // nodes: estágios {type: "nam"|"ir", fileName, data, input_gain_db, output_gain_db}
  // ou divisões {type: "split", branches: [{stages, level_db, invert, enabled}]}.
  function buildPackage(name, nodes) {
    const used = new Set();
    const files = [];
    const stage = (s) => {
      const file = safeName(s.fileName, used);
      files.push({ name: `files/${file}`, data: s.data });
      return {
        type: s.type,
        file: `files/${file}`,
        input_gain_db: s.input_gain_db,
        output_gain_db: s.output_gain_db,
        ir_max_ms: s.type === "ir" ? 20 : null,
        ir_normalize: true,
      };
    };
    const stages = nodes.map((node) => (node.type !== "split" ? stage(node) : {
      type: "split",
      branches: node.branches.map((b) => ({
        level_db: b.level_db, invert: b.invert, enabled: b.enabled, stages: b.stages.map(stage),
      })),
    }));
    const json = (o) => new TextEncoder().encode(JSON.stringify(o));
    return zip([
      { name: "chain.json", data: json({ name, arch: "a2", stages }) },
      { name: "settings.json", data: json({
        format: 1, name, arch: "a2", epochs: 100, warm_start: true, auto_stop: true,
        app_version: "mobile" }) },
      ...files,
    ]);
  }

  const api = { buildPackage, zip, crc32 };
  if (typeof module !== "undefined") module.exports = api;
  else root.ChainPackage = api;
})(this);
