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

## Licença

[MIT](LICENSE).

- Os DIs de exemplo (`poc/app/assets/di/`) são gravações de guitarra elétrica direto na interface, em domínio público (CC0 1.0), do Freesound: [clean guitar loop](https://freesound.org/people/mil_dew/sounds/613906/) de mil_dew, [Clean Strumming & Arpeggio](https://freesound.org/people/loudernoises/sounds/453384/) de loudernoises e um trecho de 30 s de [Guitar Jamming](https://freesound.org/s/182803/) de unfa. Foram convertidas para mono 48 kHz, com corte de silêncio e nível de DI (`poc/packaging/make_demo_dis.py`). O unfa pede que os sons dele não sejam usados para treinar modelos; no app, os DIs servem só para ouvir o preview e nunca entram no treino. O app de celular usa os primeiros 8 s de cada um (`mobile/www/di/`).
- O IR "Caixa 4x12 genérica" é gerado pelo próprio app a partir de uma curva de resposta (`poc/chaincap/cab_ir.py`).

As capturas `.nam` e IRs que você usa continuam sob a licença de quem as criou: o modelo gerado é um derivado delas.
