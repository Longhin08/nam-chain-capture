# NAM Chain Capture

Serviço que recebe N capturas NAM (`.nam`) + IRs opcionais, encadeia em série e treina **uma única captura nova** que reproduz a cadeia inteira — para pedaleiras que só aceitam um bloco NAM por preset.

- Spec completa: [docs/SPEC.md](docs/SPEC.md)
- Integração Tone3000 (viabilidade e termos): [docs/TONE3000.md](docs/TONE3000.md)

## Instalar o app

Baixe o instalador na página de [Releases](../../releases):

- **Windows 10/11 (64 bits):** `NAM-Chain-Capture-<versão>-Windows-Setup.exe`. Instala só para o seu usuário, sem pedir senha de administrador.
- **macOS 12+ (Apple Silicon):** `NAM-Chain-Capture-<versão>-macOS.dmg`. Arraste o app para Aplicativos. Na primeira vez, clique com o botão direito no app → **Abrir**, porque ele ainda não é assinado pela Apple.

Ao abrir, o app baixa sozinho o sinal de treino oficial do NAM (27 MB). Depois é só enviar suas capturas `.nam` e IRs, montar a cadeia, ouvir e treinar.

- **GPU no Windows:** se você tiver placa NVIDIA, o topo do app mostra **Ativar GPU NVIDIA**. O app baixa uma vez o PyTorch CUDA oficial (~2,5 GB), e o treino passa a usar a placa. Sem isso, o treino roda na CPU, que é bem mais lento.
- **GPU no Mac:** Macs com Apple Silicon usam a GPU (Metal) automaticamente.
- **Onde ficam seus arquivos:** `%APPDATA%\NAM Chain Capture` (Windows) ou `~/Library/Application Support/NAM Chain Capture` (macOS). Desinstalar o app não apaga essa pasta.

## Status

Fase 0 — PoC (ver seção 7 e Épico 0 da spec). Código em [`poc/`](poc/README.md): app desktop (Windows/macOS), CLI `python -m chaincap` e notebook do Colab.

## Estrutura planejada

```
poc/      # script Python + notebook Colab (Épico 0)
worker/   # pipeline Python de encadeamento + treino (Épico 1)
api/      # upload, jobs, perfis de dispositivo (Épico 2)
web/      # frontend React (épicos posteriores)
docs/     # spec e decisões
```
