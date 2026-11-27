// Auxiliar de poc/tests/test_mobile.py: roda o motor ou o empacotador do app no Node.
const fs = require("fs");
const path = require("path");

const [mode, specPath] = process.argv.slice(2);
const spec = JSON.parse(fs.readFileSync(specPath, "utf8"));
if (spec.noSimd) globalThis.WebAssembly = undefined;

const load = (s) => ({ ...s, fileName: s.fileName || path.basename(s.file), data: new Uint8Array(fs.readFileSync(s.file)) });
const nodes = spec.nodes.map((n) => (n.type === "split"
  ? { ...n, branches: n.branches.map((b) => ({ ...b, stages: b.stages.map(load) })) }
  : load(n)));

if (mode === "render") {
  const { renderChain } = require("./www/nam-engine.js");
  const raw = fs.readFileSync(spec.input);
  const input = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.length));
  const { y } = renderChain(input, nodes);
  fs.writeFileSync(spec.output, Buffer.from(y.buffer, y.byteOffset, y.byteLength));
} else {
  const { buildPackage } = require("./www/package.js");
  fs.writeFileSync(spec.output, buildPackage(spec.name, nodes));
}
