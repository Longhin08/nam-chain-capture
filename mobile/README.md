# App de celular (Android e navegador)

Monta a cadeia (inclusive cadeias paralelas), toca a prévia **no próprio aparelho** e gera o
pacote `.chaincap`. O treino é feito no Google Colab
([notebook](../poc/chain_capture_colab.ipynb)): não há servidor próprio.

| Pasta | O quê |
|---|---|
| `www/` | Interface, sem dependências. `nam-engine.js` é o motor da prévia |
| `wasm/` | Fonte do trecho acelerado com SIMD (`node wasm/build.js` regera `www/kernel-wasm.js`) |
| `android/` | Projeto Capacitor |

## Prévia

- Modelos WaveNet (A2 e A1 clássico) e IRs. Outros tipos, como LSTM, não têm prévia.
- Em containers A2 usa o submodelo de maior qualidade, como o app de computador.
- `poc/tests/test_mobile.py` compara a saída com a do motor Python.
- IR fora de 48 kHz é reamostrado por um método diferente do usado no treino: pequena diferença.

## Rodar no navegador

```bash
python -m http.server -d mobile/www 8000
```

## APK de teste

O workflow **App Android** (GitHub Actions) gera o APK como artefato. Para compilar no seu
computador: JDK 21 e Android SDK; depois `npm ci && npx cap sync android` e, em `android/`,
`./gradlew assembleDebug`. É um APK de depuração: para instalar uma versão nova pode ser
preciso desinstalar a anterior.
