// A prévia roda fora da tela principal, para o app não travar.
importScripts("kernel-wasm.js", "nam-engine.js");

onmessage = ({ data: { input, nodes } }) => {
  try {
    let shown = -1;
    const { y, peak } = NamEngine.renderChain(input, nodes, (f) => {
      const pct = Math.floor(f * 100);
      if (pct !== shown) postMessage({ progress: (shown = pct) });
    });
    postMessage({ y, peak }, [y.buffer]);
  } catch (e) {
    postMessage({ error: e instanceof NamEngine.PreviewError ? e.message : `Falha na prévia: ${e.message}` });
  }
};
