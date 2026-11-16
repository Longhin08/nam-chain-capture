# NAM Chain Capture — Spec inicial

> Produto: serviço que recebe N capturas NAM (`.nam`) + IRs opcionais, encadeia em série, e treina **uma única captura nova** que reproduz a cadeia inteira. Alvo: pedaleiras que só aceitam um bloco NAM por preset.

---

## 1. Problema

- Pedaleiras baratas (ex.: Sonicake Pocket Master) rodam NAM, mas **só em um bloco** da cadeia, por limite de CPU do DSP.
- O firmware é fechado; não dá para habilitar mais blocos.
- Usuários têm capturas separadas (pedal de drive, amp, cab/IR) e querem usar tudo junto no pedal.
- Hoje a solução é manual: Reaper + plugin NAM + render + treino. Exige DAW, GPU ou Colab, e cuidado com alinhamento/níveis. A maioria não faz.

## 2. Solução

Automatizar a "recaptura virtual":

1. Usuário envia os `.nam` (e IRs `.wav`), define ordem e ganhos de cada estágio.
2. Backend passa o **sinal de treino padrão do NAM** pela cadeia (inferência determinística, sem áudio real).
3. Treina um modelo novo: entrada = sinal padrão, saída = resultado da cadeia.
4. Devolve um `.nam` na arquitetura que o pedal do usuário aguenta.

Não há latência de interface, ruído ou alinhamento a resolver: é o cenário mais simples de captura possível.

## 3. Por que é viável

- Stack 100% open source:
  - Inferência: `NeuralAmpModelerCore` (C++) ou o pacote Python `neural-amp-modeler` (PyPI).
  - Treino: `neural-amp-modeler` (PyTorch).
  - IR = convolução simples (scipy/numpy ou torch).
- Sinal de treino padrão: `v3_0_0.wav` (ou o mais recente que o treinador exigir). Vem junto com o treinador.
- Treino de um modelo standard: minutos numa GPU (T4/L4); centavos por captura em spot. Em CPU é lento demais para produção.

## 4. O que já existe (concorrência / risco)

| Quem | O que faz | Risco |
|---|---|---|
| Tone3000 | Biblioteca + treinador online gratuito. Onde a comunidade inteira está. | Pode adicionar "chain capture" como feature a qualquer momento. |
| NAM Janus Parametric | App de navegador; empacota vários `.nam` num container "slimmable" (controle Slim escolhe o modelo). | Não serializa; outra abordagem. |
| Soundshed (NAM Blend) | Roda modelos em paralelo e mistura saídas. | Blend paralelo, não cadeia em série. |
| Comunidade | Faz na mão no DAW e publica como "full rig". | Público do tipo "faço de graça". |

**Público pagante:** quem tem pedaleira de um bloco só e não quer aprender DAW. Real, mas pequeno.

## 5. Onde está o valor (posicionamento)

Não vender "juntar capturas". Vender **"monte seu preset para a sua pedaleira"**:

- **Perfis por dispositivo**: usuário escolhe o pedal; o sistema já sabe arquitetura máxima, sample rate, se precisa A2-Lite, se o cab fica embutido. Esse mapeamento de hardware é o diferencial e ninguém fez direito.
- **Preview antes de treinar**: inferência de um trecho de DI pela cadeia (segundos, barato) para ouvir no navegador antes de gastar GPU.
- **Integração com Tone3000**: buscar capturas direto da biblioteca, sem o usuário baixar arquivos.

## 6. Riscos a resolver antes de escalar

- **Licença**: muitas capturas do Tone3000 restringem redistribuição/derivados. Uso próprio OK; compartilhar o resultado é zona cinzenta. Termos de uso claros desde o início.
- **Dependência de plataforma** (Tone3000).
- **Custo variável de GPU**: modelo de negócio por crédito ou assinatura, nunca "grátis ilimitado".

## 7. Plano de validação (antes do SaaS)

1. **PoC**: script Python (~200 linhas) + Colab: cadeia de N `.nam` + IR → treina → exporta.
2. Testar com os próprios arquivos; medir ESR, tempo e custo de GPU.
3. Postar em fórum/Reddit de NAM; ver se alguém usa.
4. Só então: backend, fila, billing.

## 8. Arquitetura proposta (MVP)

```
[Web/React] ──► [API] ──► [Fila] ──► [Worker Python + GPU]
                   │                        │
                   ▼                        ▼
              [PostgreSQL]           [Object storage]
           jobs, usuários,         .nam de entrada,
           perfis de pedal         IRs, .nam de saída
```

- **API**: C#/.NET ou Python (FastAPI). Responsável por upload, jobs, auth, perfis de dispositivo.
- **Worker**: obrigatoriamente Python (treinador é PyTorch). Serviço separado; não portar.
- **Fila**: Redis/RabbitMQ ou tabela de jobs no Postgres (suficiente no MVP).
- **GPU**: instância spot sob demanda (RunPod, Lambda, GCP L4). Escala para zero quando ocioso.
- **Storage**: S3-compatível.

## 9. Pipeline do worker (detalhe)

```
entrada: job { stages: [ {type: nam|ir, file, input_gain_db, output_gain_db} ], arch, device_profile }

1. carregar v3_0_0.wav (48 kHz, mono, float32)
2. x = sinal
   para cada stage na ordem:
     se nam: x = nam_infer(model, x * gain_in) * gain_out
     se ir:  x = convolve(x, ir) (normalizar IR, truncar a ~20 ms se perfil exigir)
3. checar clipping (|x| < 1.0); se clipar, reduzir ganho do último estágio e logar aviso
4. salvar y = x como WAV 24-bit, mesmo comprimento do input, sem fade/normalização
5. treinar: nam.train(input=v3_0_0.wav, output=y.wav, arch=arch, epochs=default)
6. validar ESR: < 0.01 ótimo; 0.01–0.05 ok; > 0.05 falhar o job com diagnóstico
7. exportar .nam; se device_profile exigir A2-Lite, converter
8. anexar metadados: cadeia original, ganhos, arch, ESR, perfil
```

Regras:
- Ordem importa (drive → amp → IR). Nunca reordenar automaticamente.
- IRs longos perdem cauda na captura; avisar o usuário se IR > ~20 ms.
- Manter referência de calibração de entrada (dBu) se as capturas originais tiverem; propagar para o `.nam` final.
- Nada de efeitos com lookahead na cadeia (não suportado no MVP).

## 10. Modelo de dados (mínimo)

- `users`
- `device_profiles`: nome, fabricante, max_arch, sample_rate, requires_a2lite (bool), cab_disabled_with_nam (bool), notas
- `captures`: arquivo, dono, origem (upload|tone3000), licença, metadados do header `.nam`
- `jobs`: user_id, device_profile_id, stages (jsonb), arch, status, esr, output_capture_id, custo_gpu, timestamps
- `credits` / `subscriptions`

## 11. Arquiteturas NAM (para o seletor)

| Arch | CPU relativa | Uso |
|---|---|---|
| standard | alta | pedaleiras potentes, plugin |
| lite | média | maioria das pedaleiras intermediárias |
| feather | baixa | pedais baratos |
| nano | muito baixa | quando nada mais roda liso |

Perfil de dispositivo define o máximo; usuário pode escolher abaixo.

## 12. Backlog inicial (SDD)

### Épico 0 — PoC (sem produto)
- [ ] Script: carregar `.nam`, inferir sinal padrão, encadear N modelos + IR
- [ ] Script: treinar e exportar
- [ ] Testar com arquivos próprios; registrar ESR, tempo, custo
- [ ] Carregar o resultado no pedal e validar som

### Épico 1 — Core do worker
- [ ] Pipeline da seção 9 como módulo Python testável (TDD: cada estágio puro, com WAVs de fixture)
- [ ] Validação de entrada (`.nam` válido, sample rate, tamanho)
- [ ] Detecção de clipping e ajuste
- [ ] Exportação + conversão A2-Lite

### Épico 2 — API + fila
- [ ] Upload de capturas e IRs
- [ ] Criar job, consultar status, baixar resultado
- [ ] Fila + worker com GPU sob demanda

### Épico 3 — Perfis de dispositivo
- [ ] Tabela inicial (Pocket Master, GP-200, Ampero II, HeadRush, Nano Cortex…)
- [ ] Regras: arch máxima, A2-Lite, cab embutido

### Épico 4 — Preview
- [ ] Inferência de DI curto pela cadeia, streaming para o navegador

### Épico 5 — Integração Tone3000
- [ ] Busca e importação via API
- [ ] Exibir licença da captura e bloquear derivados quando a licença proibir

### Épico 6 — Billing
- [ ] Créditos por captura; custo real de GPU por job

## 13. Critérios de aceite do MVP

- Usuário envia 2 `.nam` + 1 IR, escolhe um perfil de pedal, recebe 1 `.nam` em < 10 min.
- ESR do modelo final < 0.02 em 90% dos jobs de teste.
- Arquivo carrega e roda no pedal-alvo sem conversão manual.
- Custo de GPU por job registrado e visível no admin.

## 14. Referências

- NAM core: github.com/sdatkinson/NeuralAmpModelerCore
- NAM trainer: github.com/sdatkinson/neural-amp-modeler
- Plugin: neuralampmodeler.com
- Tone3000: tone3000.com (biblioteca, treinador online, API)
