# PoC — Épico 0

Script Python que encadeia N capturas `.nam` + IRs, renderiza o sinal de treino oficial do NAM (`v3_0_0.wav`) pela cadeia e treina **uma única captura** que reproduz tudo.

- **Interface (montar a cadeia, ouvir, treinar):**

```bash
cd poc && ../.venv/bin/python -m uvicorn app.server:app --port 8765
# abra http://127.0.0.1:8765
```

  - Envie `.nam` e IRs pela barra lateral. Eles ficam em `poc/workspace/`.
  - Monte a cadeia como uma pedaleira: arraste os pedais para reordenar (ou use ‹ ›). Os knobs de entrada e saída aceitam arrastar, roda do mouse e setas (Shift = passo de 3 dB); duplo clique zera.
  - O footswitch faz bypass. No IR, dá para ligar o corte em N ms e a normalização.
  - **Ouvir cadeia** processa até 30 s do áudio de teste escolhido e toca no navegador, com o volume normalizado. As opções são três gravações de guitarra crua em CC0 que vêm com o app (ver README principal). O pico real aparece abaixo do player.
  - **Treinar captura** cria um job em fila (um por vez, em segundo plano), com progresso por época e ESR. Os arquivos da cadeia são copiados para o job, então dá para mexer na biblioteca depois.
  - Com o job pronto, dá para **ouvir o resultado**, compará-lo com a cadeia original, baixar o `.a1.nam` e usar o modelo treinado como estágio de uma nova cadeia.
  - Sem o `v3_0_0.wav`, o topo da página mostra um botão para baixá-lo.
- **Com GPU (recomendado):** [`chain_capture_colab.ipynb`](chain_capture_colab.ipynb), abrindo no Colab.
- **Local:**

```bash
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch   # só se não tiver GPU
.venv/bin/pip install -r poc/requirements.txt
cd poc

mkdir -p data
# baixe o v3_0_0.wav oficial (md5 36cd1af62985c2fac3e654333e36431e)
curl -sSL -o data/v3_0_0.wav "https://drive.usercontent.google.com/download?id=1Pgf8PdE0rKB1TD4TRPKbpNo1ByR3IOm9&export=download&confirm=t"

python -m chaincap render data/chain.json --input data/v3_0_0.wav --out out      # só a cadeia -> out/target.wav
python -m chaincap run    data/chain.json --input data/v3_0_0.wav --out out --arch feather
```

`chain.json` (caminhos relativos ao próprio arquivo):

```json
{
  "name": "Drive + Amp + Cab",
  "arch": "standard",
  "stages": [
    {"type": "nam", "file": "drive.nam", "input_gain_db": 0, "output_gain_db": 0},
    {"type": "nam", "file": "amp.nam"},
    {"type": "ir",  "file": "cab.wav", "ir_max_ms": 20}
  ]
}
```

## App desktop (desenvolvimento)

O app é o mesmo servidor + interface dentro de uma janela nativa (pywebview: WebView2 no Windows, WebKit no macOS), empacotado com PyInstaller. Um único executável faz tudo:

| Comando | O que faz |
|---|---|
| `NAM Chain Capture` | abre a janela (servidor numa porta local aleatória, com token por sessão) |
| `… --browser` | abre no navegador padrão, se a janela nativa não estiver disponível |
| `… --self-test` | gera uma captura + IR sintéticos, processa a cadeia, treina e exporta (o CI roda isso no binário empacotado) |
| `… --serve [PORTA]` | só o servidor, sem janela |
| `… --worker run …` | (interno) treino de um job, chamado pelo próprio app |
| `… --gpu-check` | (interno) testa o PyTorch CUDA baixado |

Rodar sem empacotar: `cd poc && ../.venv/bin/python desktop_main.py` (precisa de `pip install -r requirements-desktop.txt`).

**Linux** (não tem instalador; roda a partir do código). A janela nativa usa GTK + WebKit2GTK:

```bash
sudo apt install gir1.2-webkit2-4.1 python3-gi      # geralmente já vem no Ubuntu/Mint
python3 -m venv .venv --system-site-packages         # enxerga o PyGObject do sistema
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -r poc/requirements.txt -r poc/requirements-desktop.txt
cd poc && ../.venv/bin/python desktop_main.py
```

Sem GTK/WebKit, o app abre a interface no navegador padrão. Os dados ficam em `~/.local/share/nam-chain-capture`.

Gerar os instaladores:
- **Pelo CI (recomendado):** o workflow [`.github/workflows/desktop.yml`](../.github/workflows/desktop.yml) compila no Windows e no macOS (Apple Silicon), roda os testes e o `--self-test` dentro do app empacotado e sobe o `Setup.exe` e o `.dmg` como artefatos. Um push de tag `v*` (ex.: `git tag v0.1.0 && git push --tags`) cria um **rascunho** de Release com os dois arquivos; você revisa e publica.
- **Local:** `cd poc && pyinstaller packaging/nam_chain_capture.spec --noconfirm`. No Windows, depois `iscc /DAppVersion=0.1.0 packaging\windows.iss`.

Pacote de GPU (Windows + NVIDIA), em `app/gpu_runtime.py`:
- O instalador leva o PyTorch CPU.
- Com placa NVIDIA, o app procura no índice oficial (`download.pytorch.org/whl/cuXXX`) o wheel CUDA da **mesma versão** do torch embutido, baixa, confere o sha256, extrai em `%APPDATA%\NAM Chain Capture\gpu-runtime` e testa com `--gpu-check`.
- Só o processo de treino carrega esse torch (por um import hook); a interface segue no CPU.
- Se o app for atualizado com outro torch, o pacote antigo é ignorado e o botão reaparece.

## Saídas (`--out`)

| Arquivo | O que é |
|---|---|
| `chain.a1.nam` | **Use este no pedal.** WaveNet no formato `.nam` legado 0.5.4, que firmwares antigos leem. |
| `chain.nam` | O mesmo modelo no formato atual (0.7.0), para o plugin NAM recente. |
| `target.wav` | Saída da cadeia (24-bit), que foi o alvo do treino. |
| `report.json` | ESR, veredito (`ótimo` < 0.01 ≤ `ok` ≤ 0.05 < `ruim`), tempo, custo estimado, avisos. |

O comando sai com código 2 se ESR > 0.05 (spec §9.6).

## Decisões e descobertas

- **NAM 0.13 mudou o treinador.** O `nam.train.core.train()` não tem mais seletor de arquitetura: ele treina um `PackedWaveNet` e exporta um `SlimmableContainer`, que pedaleiras não leem. O PoC treina as WaveNets clássicas (standard/lite/feather/nano) direto pelo `LightningModule`, com os mesmos splits do v3. As configs e os nºs de pesos foram conferidos com o treinador 0.10 (13802 / 6554 / 3026 / 842).
- **Formato do arquivo.** O 0.13 grava `.nam` versão 0.7.0, que o código NAM 0.10 **não carrega**. Por isso o `chain.a1.nam` é reescrito no formato 0.5.4. Conferido: carrega no NAM 0.10 com saída idêntica bit a bit, nas 4 archs.
- **Capturas de entrada.** Arquivos 0.5.x (a maioria no Tone3000) são convertidos para carregar no 0.13 (testado contra saídas de referência do 0.10, inclusive WaveNet *gated* e LSTM). De `SlimmableContainer`, usamos o submodelo de maior qualidade. No PoC só há suporte a 48 kHz.
- **IR:** normalizado por energia, convolução causal, reamostrado para 48 kHz se preciso. Gera aviso se passar de 20 ms.
- **Clipping:** se o pico passar de 0.98, a saída é atenuada (equivale a baixar o ganho de saída do último estágio) e um aviso vai para o relatório.
- **Nível:** como no treinador oficial, a saída é normalizada para −18 dBFS RMS no treino. O volume final do modelo não é o da cadeia (o plugin e os pedais normalizam por `loudness`).
- **Calibração:** o `input_level_dbu` do 1º NAM vai para o modelo final, descontando os ganhos aplicados antes dele.
- **Headless:** o `nam.train.core` importa `tkinter`. O `chaincap/_compat.py` cria um stub quando o Tk não existe.

## Testes

```bash
cd poc && ../.venv/bin/python -m pytest -q
```

As fixtures em `tests/fixtures/` foram geradas com `neural-amp-modeler==0.10.0` (formato legado + saídas de referência).

## Medições

| Data | Máquina | Cadeia | Arch | Render | Treino | ESR | Obs. |
|---|---|---|---|---|---|---|---|
| 2026-11-16 | CPU 8 núcleos, sem GPU | 2 NAM sintéticos (pesos aleatórios) + IR 50 ms → 20 ms | standard | 70 s | 123 s/época | — | Só valida o pipeline. A saída sintética era quase só DC, então o ESR não significa nada. |

## Pendente no Épico 0

- [ ] Rodar com capturas reais no Colab (GPU) e registrar ESR, tempo e custo.
- [ ] Carregar o `chain.a1.nam` no Pocket Master e validar o som.
- [ ] Conversão A2-Lite (spec §9.7): ainda não implementada; falta definir quais pedais pedem esse formato.
