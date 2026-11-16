# NAM Chain Capture

Serviço que recebe N capturas NAM (`.nam`) + IRs opcionais, encadeia em série e treina **uma única captura nova** que reproduz a cadeia inteira — para pedaleiras que só aceitam um bloco NAM por preset.

- Spec completa: [docs/SPEC.md](docs/SPEC.md)
- Integração Tone3000 (viabilidade e termos): [docs/TONE3000.md](docs/TONE3000.md)

## Status

Fase 0 — PoC (ver seção 7 e Épico 0 da spec). Código em [`poc/`](poc/README.md): CLI `python -m chaincap` + notebook do Colab.

## Estrutura planejada

```
poc/      # script Python + notebook Colab (Épico 0)
worker/   # pipeline Python de encadeamento + treino (Épico 1)
api/      # upload, jobs, perfis de dispositivo (Épico 2)
web/      # frontend React (épicos posteriores)
docs/     # spec e decisões
```
