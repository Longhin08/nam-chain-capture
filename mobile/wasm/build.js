// Compila kernel.wat para ../www/kernel-wasm.js. Precisa do pacote `wabt` (devDependency).
const fs = require("fs");
const path = require("path");

require("wabt")().then((wabt) => {
  const source = fs.readFileSync(path.join(__dirname, "kernel.wat"), "utf8");
  const module = wabt.parseWat("kernel.wat", source, { simd: true });
  module.validate();
  const base64 = Buffer.from(module.toBinary({}).buffer).toString("base64");
  fs.writeFileSync(path.join(__dirname, "..", "www", "kernel-wasm.js"),
    "// GERADO por mobile/wasm/build.js a partir de kernel.wat. Não editar.\n" +
    `(typeof module !== "undefined" ? module : { set exports(v) { self.KERNEL_WASM = v; } })` +
    `.exports = "${base64}";\n`);
});
