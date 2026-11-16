# PoC — Épico 0

Script Python que encadeia N capturas `.nam` + IRs, renderiza o sinal de treino oficial do NAM (`v3_0_0.wav`) pela cadeia e treina **uma única captura** que reproduz tudo.

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

## Pendente no Épico 0

- [ ] Rodar com capturas reais no Colab (GPU) e registrar ESR, tempo e custo.
- [ ] Carregar o `chain.a1.nam` no Pocket Master e validar o som.
- [ ] Conversão A2-Lite (spec §9.7): ainda não implementada; falta definir quais pedais pedem esse formato.
