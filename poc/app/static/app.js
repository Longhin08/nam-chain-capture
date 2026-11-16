"use strict";

const $ = (s, el = document) => el.querySelector(s);
const STORE_KEY = "chaincap.chain.v1";
const ARCH_INFO = {
  standard: "plugin e pedais fortes",
  lite: "maioria das pedaleiras",
  feather: "pedais baratos",
  nano: "quando nada mais roda",
};
const KIND_LABEL = { nam: "Captura NAM", ir: "IR de caixa" };

// Como desenhar cada estágio, a partir do gear_type gravado no .nam.
//   pedal  -> pedal de chão     head -> cabeçote
//   combo  -> amp + caixa       cab  -> gabinete (IR)
const GEAR_FORM = {
  pedal: ["pedal", "Pedal"],
  amp: ["head", "Amplificador"],
  preamp: ["head", "Pré-amplificador"],
  pedal_amp: ["head", "Pedal + amp"],
  studio: ["head", "Equipamento de estúdio"],
  amp_cab: ["combo", "Amp + caixa"],
  amp_pedal_cab: ["combo", "Pedal + amp + caixa"],
};

// Cor de esmalte de cada pedal: fixa por nome, para reconhecer de relance.
const ENAMELS = [
  "#e3b23c", // mostarda
  "#f2d03b", // amarelo
  "#e07a2f", // laranja
  "#d9572b", // laranja queimado
  "#c8392b", // vermelho
  "#8e1f2b", // vinho
  "#e05a8a", // rosa
  "#9c3d8a", // magenta
  "#7a4fa3", // roxo
  "#4b3f8f", // índigo
  "#3b6fb6", // azul
  "#2a9fd6", // azul-claro
  "#1f4e79", // azul-marinho
  "#2f8f8a", // petróleo
  "#3fb8a0", // turquesa
  "#3f8f5a", // verde
  "#8bbf3f", // verde-limão
  "#3d5a2e", // verde-oliva
  "#8a5a3b", // marrom
  "#c9a06a", // caramelo
  "#e9dfc7", // creme
  "#f4f1ea", // branco
  "#b8bcc2", // prata
  "#2b2c30", // preto
];
function enamelFor(name) {
  let h = 2166136261;
  for (const ch of name || "") h = Math.imul(h ^ ch.charCodeAt(0), 16777619) >>> 0;
  const color = ENAMELS[h % ENAMELS.length];
  // texto claro em esmalte escuro (luminância relativa WCAG)
  const lin = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  const [r, g, b] = [1, 3, 5].map((i) => lin(parseInt(color.slice(i, i + 2), 16)));
  const L = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return [color, L < 0.18];
}

function stageForm(st) {
  if (st.job) return ["combo", "Modelo treinado"];
  if (st.type === "ir") return ["cab", "Gabinete (IR)"];
  const info = fileInfo("nam", st.file);
  return (info && GEAR_FORM[info.gear_type]) || ["pedal", "Captura NAM"];
}
const DROP_TEXT = {
  nam: "Solte arquivos .nam aqui ou clique para escolher",
  ir: "Solte IRs .wav aqui ou clique para escolher",
  di: "Solte gravações de guitarra crua (.wav) aqui ou clique para escolher",
};

const state = {
  kind: "nam",
  files: { nam: [], ir: [], di: [] },
  chain: [],
  status: null,
  jobs: [],
  openLogs: new Set(),
  audioUrl: null,
};

// ------------------------------------------------------------ utilidades

// O app desktop abre a página com ?token=…; guardamos e limpamos a URL.
const TOKEN = (() => {
  const q = new URLSearchParams(location.search);
  let t = q.get("token");
  try {
    if (t) sessionStorage.setItem("chaincap.token", t);
    else t = sessionStorage.getItem("chaincap.token");
  } catch {}
  if (q.has("token")) history.replaceState(null, "", location.pathname);
  return t || "";
})();
const withToken = (url) => (TOKEN ? url + (url.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(TOKEN) : url);
const desktopApi = () => (window.pywebview && window.pywebview.api) || null;

async function api(path, opts = {}) {
  opts.headers = { ...(opts.headers || {}), ...(TOKEN ? { "X-Chaincap-Token": TOKEN } : {}) };
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const j = await res.json(); if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch {}
    throw new Error(msg);
  }
  return res;
}
const getJSON = (p) => api(p).then((r) => r.json());
const postJSON = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

let toastTimer;
function toast(msg, isErr = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.toggle("err", isErr);
  t.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove("show"), isErr ? 6000 : 3000);
}

function saveChain() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(state.chain)); } catch {}
}
function loadChain() {
  try { state.chain = JSON.parse(localStorage.getItem(STORE_KEY) || "[]"); } catch { state.chain = []; }
}
const uid = () => Math.random().toString(36).slice(2, 10);
const fmtDb = (v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} dB`;
const stem = (name) => name.replace(/\.(nam|wav)$/i, "");

function fileInfo(kind, name) {
  return state.files[kind].find((f) => f.name === name);
}

// ------------------------------------------------------------ status

async function refreshStatus() {
  state.status = await getJSON("/api/status");
  renderStatus();
}

function renderStatus() {
  const s = state.status;
  const el = $("#status");
  el.innerHTML = "";
  const v3 = document.createElement("span");
  v3.className = "chip " + (s.v3.valid ? "ok" : "bad");
  if (s.v3.valid) {
    v3.innerHTML = `<span class="dot"></span>Sinal de treino pronto`;
  } else if (s.v3.downloading) {
    v3.innerHTML = `<span class="dot"></span>Baixando sinal de treino…`;
  } else {
    v3.innerHTML = `<span class="dot"></span>${s.v3.error ? s.v3.error : "Falta o sinal de treino"} `;
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = "Baixar (27 MB)";
    b.onclick = async () => {
      await api("/api/v3/download", { method: "POST" });
      await refreshStatus();
      pollStatus();
    };
    v3.append(b);
  }
  el.append(v3, gpuChip(s));
  renderTrainNote();
  renderDiSelect();
}

function gpuChip(s) {
  const r = s.gpu_runtime || {};
  const chip = document.createElement("span");
  const dot = `<span class="dot"></span>`;
  if (r.installing) {
    chip.className = "chip bad";
    const pct = r.total_bytes ? ` ${Math.floor((100 * r.done_bytes) / r.total_bytes)}%` : "";
    const phase = { procurando: "Procurando PyTorch CUDA", baixando: "Baixando pacote de GPU", instalando: "Instalando pacote de GPU", testando: "Testando a GPU" }[r.phase] || "Preparando GPU";
    chip.innerHTML = `${dot}${phase}…${pct}`;
    return chip;
  }
  if (s.gpu) {
    chip.className = "chip ok";
    chip.innerHTML = `${dot}GPU: `;
    chip.append(document.createTextNode(s.gpu));
    if (r.installed) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = "Remover pacote";
      b.title = "Apaga o PyTorch CUDA baixado (libera ~4 GB); o treino volta a usar a CPU";
      b.onclick = async () => {
        try { await api("/api/gpu/remove", { method: "POST" }); await refreshStatus(); } catch (e) { toast(e.message, true); }
      };
      chip.append(" ", b);
    }
    return chip;
  }
  chip.className = "chip bad";
  chip.innerHTML = `${dot}${r.error ? "" : "Sem GPU: treino lento"}`;
  if (r.error) chip.append(document.createTextNode(r.error));
  if (r.supported && r.nvidia_detected) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = r.error ? "Tentar de novo" : "Ativar GPU NVIDIA (~2,5 GB)";
    b.title = "Baixa o PyTorch CUDA oficial para treinar na sua placa de vídeo";
    b.onclick = async () => {
      try { await api("/api/gpu/install", { method: "POST" }); await refreshStatus(); pollStatus(); } catch (e) { toast(e.message, true); }
    };
    chip.append(" ", b);
  }
  return chip;
}

let statusPoll;
function pollStatus() {
  clearInterval(statusPoll);
  statusPoll = setInterval(async () => {
    await refreshStatus();
    const busy = state.status.v3.downloading || (state.status.gpu_runtime && state.status.gpu_runtime.installing);
    if (!busy) clearInterval(statusPoll);
  }, 1500);
}

// ------------------------------------------------------------ biblioteca

async function refreshFiles() {
  state.files = await getJSON("/api/files");
  renderFiles();
  renderDiSelect();
  renderChain();
}

function describeFile(f) {
  if (f.error) return f.error;
  if (f.kind === "nam") {
    const bits = [f.architecture];
    if (f.weights) bits.push(`${f.weights.toLocaleString("pt-BR")} pesos`);
    if (f.modeled_by) bits.push(`por ${f.modeled_by}`);
    return bits.filter(Boolean).join(", ");
  }
  const ms = f.seconds * 1000;
  return f.kind === "ir"
    ? `${ms < 1000 ? ms.toFixed(0) + " ms" : f.seconds.toFixed(2) + " s"}, ${f.sample_rate / 1000} kHz`
    : `${f.seconds.toFixed(1)} s, ${f.sample_rate / 1000} kHz`;
}

function renderFiles() {
  const ul = $("#files");
  ul.innerHTML = "";
  const list = state.files[state.kind];
  $("#drop-text").textContent = DROP_TEXT[state.kind];
  $("#file-input").accept = state.kind === "nam" ? ".nam" : ".wav,audio/wav";
  if (!list.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Nenhum arquivo ainda.";
    ul.append(li);
    return;
  }
  for (const f of list) {
    const li = document.createElement("li");
    li.className = "file" + (f.error ? " err" : "");
    li.dataset.kind = f.kind;
    const info = document.createElement("div");
    info.innerHTML = `<div class="fname"></div><div class="fmeta"></div>`;
    info.querySelector(".fname").textContent = f.title || stem(f.name);
    info.querySelector(".fmeta").textContent = describeFile(f);
    info.title = f.name;
    li.append(info);
    if (f.builtin) {
      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "Exemplo";
      tag.title = "Gravação em domínio público que vem com o app";
      li.append(tag, document.createElement("span"));
      ul.append(li);
      continue;
    }
    if (f.kind !== "di" && !f.error) {
      li.draggable = true;
      li.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("application/x-chaincap-file", JSON.stringify({ kind: f.kind, name: f.name }));
        e.dataTransfer.effectAllowed = "copy";
      });
      const add = document.createElement("button");
      add.type = "button";
      add.className = "small";
      add.textContent = "Adicionar";
      add.setAttribute("aria-label", `Adicionar ${f.name} à cadeia`);
      add.onclick = () => addStage(f.kind, f.name);
      li.append(add);
    } else {
      li.append(document.createElement("span"));
    }
    const del = document.createElement("button");
    del.type = "button";
    del.className = "icon";
    del.textContent = "×";
    del.setAttribute("aria-label", `Apagar ${f.name}`);
    del.onclick = async () => {
      try {
        await api(`/api/files/${f.kind}/${encodeURIComponent(f.name)}`, { method: "DELETE" });
        toast(`${f.name} apagado`);
        await refreshFiles();
      } catch (e) { toast(e.message, true); }
    };
    li.append(del);
    ul.append(li);
  }
}

async function uploadFiles(fileList, { target = null } = {}) {
  const added = [];
  for (const file of fileList) {
    const ext = file.name.toLowerCase().split(".").pop();
    if (ext !== "nam" && ext !== "wav") {
      toast(`${file.name}: envie capturas .nam ou IRs .wav`, true);
      continue;
    }
    // .nam vai para capturas; .wav é IR de caixa, ou DI se a aba DIs estiver aberta
    const kind = ext === "nam" ? "nam" : state.kind === "di" && !target ? "di" : "ir";
    const fd = new FormData();
    fd.append("kind", kind);
    fd.append("file", file);
    try {
      const r = await api("/api/files", { method: "POST", body: fd });
      const info = await r.json();
      toast(`${info.name} enviado`);
      added.push(info);
      if (kind === "di") {
        // DI novo vira o áudio de teste padrão.
        try { localStorage.setItem(DI_KEY, info.name); } catch {}
      }
    } catch (e) {
      toast(e.message, true);
    }
  }
  await refreshFiles();
  if (target && added.length) {
    let at = Math.min(target.at, target.arr.length);
    for (const info of added) if (info.kind !== "di") target.arr.splice(at++, 0, newStage(info.kind, info.name));
    commit();
  }
  return added;
}

function setupLibrary() {
  for (const b of document.querySelectorAll(".tabs button")) {
    b.onclick = () => {
      state.kind = b.dataset.kind;
      for (const o of document.querySelectorAll(".tabs button")) o.setAttribute("aria-selected", String(o === b));
      renderFiles();
    };
  }
  const drop = $("#drop");
  $("#file-input").onchange = (e) => { uploadFiles(e.target.files); e.target.value = ""; };
  drop.addEventListener("dragover", (e) => { if (e.dataTransfer.types.includes("Files")) { e.preventDefault(); drop.classList.add("over"); } });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", () => drop.classList.remove("over"));

  // Arquivo solto em QUALQUER lugar da janela: nunca deixar a janela abrir o
  // arquivo como página. Na cadeia, ele já entra como estágio.
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  window.addEventListener("dragover", (e) => {
    if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; }
  });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    clearDropMarks();
    const near = (sel) => (e.target.closest ? e.target.closest(sel) : null);
    const pedal = near(".pedal"), row = near(".branch-row");
    let target = null;
    if (pedal && locate(pedal.dataset.uid)) {
      const loc = locate(pedal.dataset.uid);
      const r = pedal.getBoundingClientRect();
      target = { arr: loc.arr, at: loc.i + (e.clientX > r.left + r.width / 2 ? 1 : 0) };
    } else if (row) {
      const sp = splitNode();
      const b = sp && sp.branches.find((x) => x.uid === row.dataset.branchUid);
      if (b) target = { arr: b.stages, at: b.stages.length };
    } else if (near("#board")) {
      target = { arr: state.chain, at: mainIndexAt(e.clientX) };
    }
    uploadFiles(e.dataTransfer.files, { target });
  });
}

// ------------------------------------------------------------ cadeia
//
// state.chain é a lista principal de nós. Um nó é um estágio
//   { uid, type: "nam" | "ir", file | job, input_gain_db, output_gain_db, ... }
// ou uma divisão (no máximo uma)
//   { uid, type: "split", branches: [{ uid, stages: [estágios], level_db, invert, enabled }, ...] }
// Antes da divisão: estágios em comum; depois dela: estágios aplicados à soma.

const AMPISH = ["amp", "amp_cab", "preamp", "pedal_amp", "amp_pedal_cab"];

function newStage(type, file, extra = {}) {
  return {
    uid: uid(), type, file, input_gain_db: 0, output_gain_db: 0,
    ir_max_ms: type === "ir" ? 20 : null, ir_normalize: true, enabled: true, ...extra,
  };
}

function addStage(type, file, extra = {}) {
  state.chain.push(newStage(type, file, extra));
  saveChain();
  renderChain();
}

const isSplit = (n) => n && n.type === "split";
const splitNode = () => state.chain.find(isSplit) || null;
const BRANCH_NAMES = "ABCD";

// Todas as listas que guardam estágios (principal + cadeias da divisão).
function stageLists() {
  const out = [state.chain];
  const sp = splitNode();
  if (sp) for (const b of sp.branches) out.push(b.stages);
  return out;
}

function locate(uidv) {
  for (const arr of stageLists()) {
    const i = arr.findIndex((n) => n.uid === uidv);
    if (i >= 0) return { arr, i };
  }
  return null;
}

function allStages() {
  const out = [];
  for (const n of state.chain) {
    if (isSplit(n)) n.branches.forEach((b) => out.push(...b.stages));
    else out.push(n);
  }
  return out;
}

// Caminhos que o sinal percorre (um por cadeia ligada), só com estágios ligados.
function signalPaths() {
  const sp = splitNode();
  const on = (arr) => arr.filter((s) => !isSplit(s) && s.enabled);
  if (!sp) return [on(state.chain)];
  const k = state.chain.indexOf(sp);
  const pre = on(state.chain.slice(0, k)), post = on(state.chain.slice(k + 1));
  return sp.branches.filter((b) => b.enabled).map((b) => [...pre, ...on(b.stages), ...post]);
}

function activeStageCount() {
  return new Set(signalPaths().flat().map((s) => s.uid)).size;
}

function stageMissing(st) {
  if (st.job) return !state.jobs.some((j) => j.id === st.job && j.files.includes("chain.nam"));
  return !fileInfo(st.type, st.file);
}

function makeKnob(label, value, onChange, { min = -24, max = 24 } = {}) {
  const wrap = document.createElement("div");
  wrap.className = "knob-wrap";
  const knob = document.createElement("div");
  knob.className = "knob";
  knob.tabIndex = 0;
  knob.setAttribute("role", "slider");
  knob.setAttribute("aria-label", label);
  knob.setAttribute("aria-valuemin", String(min));
  knob.setAttribute("aria-valuemax", String(max));
  const lab = document.createElement("span");
  lab.className = "knob-label";
  lab.textContent = label;
  const val = document.createElement("span");
  val.className = "knob-value";
  let v = value;
  const set = (nv, commit = true) => {
    v = Math.max(min, Math.min(max, Math.round(nv * 2) / 2));
    // 0 dB sempre no meio-dia; cada lado usa a sua faixa.
    // negativo gira para a esquerda (anti-horário), positivo para a direita
    const ang = v >= 0 ? (v / max) * 135 : (v / -min) * 135;
    knob.style.setProperty("--angle", `${ang}deg`);
    knob.setAttribute("aria-valuenow", String(v));
    knob.setAttribute("aria-valuetext", fmtDb(v));
    val.textContent = fmtDb(v);
    if (commit) onChange(v);
  };
  set(v, false);
  knob.addEventListener("keydown", (e) => {
    const step = e.shiftKey ? 3 : 0.5;
    const map = { ArrowUp: step, ArrowRight: step, ArrowDown: -step, ArrowLeft: -step, PageUp: 6, PageDown: -6 };
    if (e.key in map) { e.preventDefault(); set(v + map[e.key]); }
    if (e.key === "Home") { e.preventDefault(); set(min); }
    if (e.key === "End") { e.preventDefault(); set(max); }
    if (e.key === "0") { e.preventDefault(); set(0); }
  });
  knob.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    controlActive = true;
    knob.focus();
    knob.setPointerCapture(e.pointerId);
    const y0 = e.clientY, v0 = v;
    const move = (ev) => set(v0 + (y0 - ev.clientY) * (ev.shiftKey ? 0.05 : 0.2));
    const up = () => { knob.removeEventListener("pointermove", move); knob.removeEventListener("pointerup", up); };
    knob.addEventListener("pointermove", move);
    knob.addEventListener("pointerup", up);
  });
  knob.addEventListener("wheel", (e) => { e.preventDefault(); set(v + (e.deltaY < 0 ? 0.5 : -0.5)); }, { passive: false });
  knob.addEventListener("dblclick", () => set(0));
  knob.title = "Arraste para cima/baixo, use as setas ou dê dois cliques para zerar";
  wrap.append(knob, lab, val);
  return wrap;
}

let dragUid = null;
// Enquanto um knob ou campo é manipulado, o pedal não pode começar a ser arrastado.
let controlActive = false;
document.addEventListener("pointerup", () => { controlActive = false; });

function commit() {
  saveChain();
  renderChain();
}

// Move o estágio arrastado (ou um arquivo da biblioteca) para arr[at].
function dropInto(arr, at, e) {
  const fileData = e.dataTransfer.getData("application/x-chaincap-file");
  if (fileData) {
    const f = JSON.parse(fileData);
    arr.splice(at, 0, newStage(f.kind, f.name));
  } else if (dragUid) {
    const src = locate(dragUid);
    if (!src) return;
    if (src.arr === arr && src.i < at) at -= 1;
    const [moved] = src.arr.splice(src.i, 1);
    arr.splice(at, 0, moved);
  }
  commit();
}

function makePedal(st, arr, i) {
  const li = $("#pedal-tpl").content.firstElementChild.cloneNode(true);
  li.dataset.kind = st.type;
  li.dataset.uid = st.uid;
  if (st.job) li.dataset.trained = "";
  li.classList.toggle("bypassed", !st.enabled);
  const info = st.job ? null : fileInfo(st.type, st.file);
  const [form, formLabel] = stageForm(st);
  li.dataset.form = form;
  if (form === "pedal") {
    const [color, lightInk] = enamelFor(st.file || st.label || "");
    li.style.setProperty("--enamel", color);
    li.classList.toggle("light-ink", lightInk);
  }
  $(".kind", li).textContent = formLabel;
  $(".pedal-name", li).textContent = st.label || (info && info.title) || stem(st.file || "");
  const meta = $(".pedal-meta", li);
  if (stageMissing(st)) meta.textContent = "Arquivo não está mais na biblioteca";
  else if (info) meta.textContent = describeFile(info);
  else meta.textContent = st.arch || "";
  $(".knobs", li).append(
    makeKnob("Entrada", st.input_gain_db, (v) => { st.input_gain_db = v; saveChain(); }),
    makeKnob("Saída", st.output_gain_db, (v) => { st.output_gain_db = v; saveChain(); }),
  );
  if (st.type === "ir") {
    const opts = $(".ir-opts", li);
    opts.hidden = false;
    const on = $(".trunc-on", opts), ms = $(".trunc-ms", opts), norm = $(".norm", opts);
    on.checked = st.ir_max_ms != null;
    ms.value = st.ir_max_ms ?? 20;
    ms.disabled = !on.checked;
    norm.checked = st.ir_normalize;
    on.onchange = () => { st.ir_max_ms = on.checked ? Number(ms.value) || 20 : null; ms.disabled = !on.checked; saveChain(); };
    ms.onchange = () => { st.ir_max_ms = Math.max(1, Math.min(2000, Number(ms.value) || 20)); ms.value = st.ir_max_ms; saveChain(); };
    norm.onchange = () => { st.ir_normalize = norm.checked; saveChain(); };
  }
  const fs = $(".footswitch", li);
  fs.setAttribute("aria-pressed", String(st.enabled));
  fs.onclick = () => { st.enabled = !st.enabled; commit(); };
  $(".remove", li).onclick = () => { arr.splice(i, 1); commit(); };
  // ‹ › movem dentro da mesma lista, pulando a divisão na lista principal.
  const sp = splitNode();
  const inBranch = arr !== state.chain;
  const step = (dir) => {
    let j = i + dir;
    if (inBranch && (j < 0 || j >= arr.length)) {
      // sai da divisão: para antes dela (‹) ou depois dela (›)
      arr.splice(i, 1);
      const k = state.chain.indexOf(sp);
      state.chain.splice(dir < 0 ? k : k + 1, 0, st);
    } else if (arr === state.chain && isSplit(arr[j])) {
      // entra na cadeia A pela ponta mais próxima
      const a = arr[j].branches[0].stages;
      arr.splice(i, 1);
      if (dir > 0) a.unshift(st); else a.push(st);
    } else {
      arr.splice(i, 1);
      arr.splice(j, 0, st);
    }
    commit();
    const el = document.querySelector(`.pedal[data-uid="${st.uid}"] ${dir < 0 ? ".move-left" : ".move-right"}`);
    (el && el.style.visibility !== "hidden" ? el : document.querySelector(`.pedal[data-uid="${st.uid}"] .footswitch`)).focus();
  };
  const left = $(".move-left", li), right = $(".move-right", li);
  // Dentro de uma cadeia A/B as pontas continuam ativas (saem da divisão).
  const first = i === 0 && !inBranch, last = i === arr.length - 1 && !inBranch;
  left.disabled = first; right.disabled = last;
  left.style.visibility = first ? "hidden" : "";
  right.style.visibility = last ? "hidden" : "";
  if (inBranch) {
    if (i === 0) left.setAttribute("aria-label", "Tirar da divisão (para antes dela)");
    if (i === arr.length - 1) right.setAttribute("aria-label", "Tirar da divisão (para depois dela)");
  }
  left.onclick = () => step(-1);
  right.onclick = () => step(1);

  for (const ctl of li.querySelectorAll("input, button")) {
    ctl.addEventListener("pointerdown", () => { controlActive = true; });
  }
  li.addEventListener("dragstart", (e) => {
    if (controlActive) { e.preventDefault(); return; }
    if (e.target !== li) return;
    e.stopPropagation();
    dragUid = st.uid;
    li.classList.add("dragging");
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", st.uid);
  });
  li.addEventListener("dragend", () => { dragUid = null; li.classList.remove("dragging"); clearDropMarks(); });
  li.addEventListener("dragover", (e) => {
    const isFile = e.dataTransfer.types.includes("application/x-chaincap-file");
    if (!dragUid && !isFile) return;
    e.preventDefault();
    e.stopPropagation();
    const after = e.offsetX > li.offsetWidth / 2;
    clearDropMarks();
    li.classList.add(after ? "drop-after" : "drop-before");
  });
  li.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length) return; // tratado na janela
    e.preventDefault();
    e.stopPropagation();
    const after = li.classList.contains("drop-after");
    clearDropMarks();
    dropInto(arr, i + (after ? 1 : 0), e);
  });
  return li;
}

function makeToggle(label, pressed, title, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "toggle";
  b.textContent = label;
  b.title = title;
  b.setAttribute("aria-pressed", String(pressed));
  b.onclick = onClick;
  b.addEventListener("pointerdown", () => { controlActive = true; });
  return b;
}

function makeSplit(sp, i) {
  const li = document.createElement("li");
  li.className = "split-node";
  li.dataset.uid = sp.uid;
  const block = document.createElement("div");
  block.className = "split-block";
  block.setAttribute("role", "group");
  block.setAttribute("aria-label", "Divisão do sinal");

  const head = document.createElement("div");
  head.className = "split-head";
  const title = document.createElement("span");
  title.className = "split-title";
  title.textContent = "Divisão do sinal";
  const hint = document.createElement("span");
  hint.className = "split-hint";
  hint.textContent = "as cadeias tocam juntas e são somadas no mixer";
  const mv = (dir, label, glyph) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "icon";
    b.textContent = glyph;
    b.setAttribute("aria-label", label);
    const j = i + dir;
    b.disabled = j < 0 || j >= state.chain.length;
    b.style.visibility = b.disabled ? "hidden" : "";
    b.onclick = () => { state.chain.splice(i, 1); state.chain.splice(j, 0, sp); commit(); };
    return b;
  };
  const undo = document.createElement("button");
  undo.type = "button";
  undo.className = "small";
  undo.textContent = "Desfazer divisão";
  undo.title = "Mantém os pedais da cadeia A no lugar da divisão e remove a cadeia B";
  undo.onclick = () => unsplit(sp);
  head.append(mv(-1, "Mover divisão para antes", "‹"), title, hint, mv(1, "Mover divisão para depois", "›"), undo);
  block.append(head);

  sp.branches.forEach((b, k) => {
    const row = document.createElement("div");
    row.className = "branch-row" + (b.enabled ? "" : " muted");
    row.dataset.branchUid = b.uid;
    const badge = document.createElement("span");
    badge.className = "branch-badge";
    badge.textContent = BRANCH_NAMES[k];
    badge.setAttribute("aria-label", `Cadeia ${BRANCH_NAMES[k]}`);

    const ol = document.createElement("ol");
    ol.className = "branch-chain";
    ol.setAttribute("aria-label", `Pedais da cadeia ${BRANCH_NAMES[k]}`);
    b.stages.forEach((st, j) => ol.append(makePedal(st, b.stages, j)));
    if (!b.stages.length) {
      const empty = document.createElement("li");
      empty.className = "branch-empty";
      empty.textContent = "Vazia: passa o sinal limpo. Arraste pedais para cá.";
      ol.append(empty);
    }
    // Soltar na área da cadeia (fora de um pedal) adiciona no fim dela.
    row.addEventListener("dragover", (e) => {
      const isFile = e.dataTransfer.types.includes("application/x-chaincap-file");
      if (!dragUid && !isFile) return;
      e.preventDefault();
      e.stopPropagation();
      row.classList.add("drop-target");
    });
    row.addEventListener("dragleave", (e) => { if (!row.contains(e.relatedTarget)) row.classList.remove("drop-target"); });
    row.addEventListener("drop", (e) => {
      row.classList.remove("drop-target");
      if (e.dataTransfer.files && e.dataTransfer.files.length) return; // tratado na janela
      e.preventDefault();
      e.stopPropagation();
      dropInto(b.stages, b.stages.length, e);
    });

    const mixer = document.createElement("div");
    mixer.className = "mixer";
    mixer.append(
      makeKnob("Nível", b.level_db, (v) => { b.level_db = v; saveChain(); }, { min: -24, max: 12 }),
      makeToggle("Ø fase", b.invert, "Inverte a fase desta cadeia (use se a soma ficar magra)", () => { b.invert = !b.invert; commit(); }),
      makeToggle("Mudo", !b.enabled, "Silencia esta cadeia", () => {
        if (b.enabled && sp.branches.filter((x) => x.enabled).length === 1) {
          toast("Pelo menos uma cadeia precisa ficar ligada", true);
          return;
        }
        b.enabled = !b.enabled;
        commit();
      }),
    );
    row.append(badge, ol, mixer);
    block.append(row);
  });
  li.append(block);
  return li;
}

function splitChain() {
  if (splitNode()) return;
  // Divide a partir do 1º amp: ele e o que vem depois viram a cadeia A, e a
  // cadeia B começa como cópia da A (é só trocar o amp dela).
  const k = state.chain.findIndex((s) => {
    if (s.job) return true;
    const info = s.type === "nam" ? fileInfo("nam", s.file) : null;
    return info && AMPISH.includes(info.gear_type);
  });
  const moved = k >= 0 ? state.chain.splice(k) : [];
  const copy = moved.map((s) => ({ ...s, uid: uid() }));
  const branch = (stages) => ({ uid: uid(), stages, level_db: 0, invert: false, enabled: true });
  state.chain.push({ uid: uid(), type: "split", branches: [branch(moved), branch(copy)] });
  commit();
  toast(moved.length
    ? "Cadeia B criada como cópia da A: troque o amp dela"
    : "Divisão criada: arraste pedais para as cadeias A e B");
}

function unsplit(sp) {
  const i = state.chain.indexOf(sp);
  const backup = JSON.parse(JSON.stringify(state.chain));
  state.chain.splice(i, 1, ...sp.branches[0].stages);
  commit();
  showUndo("Divisão desfeita: a cadeia A ficou no lugar.", backup);
}

let undoTimer;
function showUndo(msg, backup) {
  const bar = $("#undo-bar");
  bar.innerHTML = "";
  const p = document.createElement("span");
  p.textContent = msg;
  const b = document.createElement("button");
  b.type = "button";
  b.className = "small";
  b.textContent = "Desfazer";
  b.onclick = () => { state.chain = backup; bar.hidden = true; commit(); };
  bar.append(p, b);
  bar.hidden = false;
  clearTimeout(undoTimer);
  undoTimer = setTimeout(() => { bar.hidden = true; }, 10000);
}

function renderChain() {
  const ol = $("#chain");
  ol.innerHTML = "";
  state.chain.forEach((n, i) => ol.append(isSplit(n) ? makeSplit(n, i) : makePedal(n, state.chain, i)));
  const total = allStages().length, active = activeStageCount();
  const sp = splitNode();
  $("#board-hint").textContent = total
    ? `${active} de ${total} estágios ligados${sp ? `, sinal dividido em ${sp.branches.length} cadeias` : ""}. O sinal passa da esquerda para a direita; arraste os pedais para reordenar.`
    : "Monte a cadeia na ordem do sinal: drive, amp, caixa.";
  const splitBtn = $("#split-chain");
  splitBtn.disabled = !!sp;
  splitBtn.title = sp ? "Já existe uma divisão nesta cadeia" : "Divide o sinal em duas cadeias paralelas (ex.: dois amps)";
  $("#play-chain").disabled = !active;
  renderCabAlert();
  renderTrainNote();
}

// Captura só de amp no fim de um caminho do sinal, sem IR depois: som áspero.
function renderCabAlert() {
  const box = $("#board-alert");
  if (!box || !state.status) return;
  let found = null;
  for (const path of signalPaths()) {
    let last = -1;
    path.forEach((st, i) => {
      const info = !st.job && st.type === "nam" ? fileInfo("nam", st.file) : null;
      if (info && (state.status.amp_only_gear || []).includes(info.gear_type)) last = i;
    });
    if (last >= 0 && !path.slice(last + 1).some((s) => s.type === "ir")) { found = path[last]; break; }
  }
  if (!found) { box.hidden = true; box.innerHTML = ""; return; }
  const ampInfo = fileInfo("nam", found.file);
  const name = (ampInfo && ampInfo.title) || stem(found.file);
  box.hidden = false;
  box.innerHTML = "";
  const p = document.createElement("p");
  p.textContent = `“${name}” é uma captura só do amplificador. Sem uma caixa (IR) depois dela, o som fica áspero e com zumbido nos agudos.`;
  box.append(p);
  const cab = state.status.builtin_cab;
  if (cab && fileInfo("ir", cab)) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "secondary";
    b.textContent = "Adicionar caixa genérica";
    b.onclick = () => {
      const loc = locate(found.uid);
      loc.arr.splice(loc.i + 1, 0, newStage("ir", cab));
      commit();
      toast("Caixa genérica adicionada depois do amp");
    };
    box.append(b);
  } else {
    const hint = document.createElement("span");
    hint.className = "note";
    hint.textContent = "Envie um IR de caixa na aba IRs.";
    box.append(hint);
  }
}

// Índice na lista principal onde algo solto em clientX deve entrar.
function mainIndexAt(clientX) {
  const kids = [...$("#chain").children];
  return kids.filter((k) => { const r = k.getBoundingClientRect(); return clientX > r.left + r.width / 2; }).length;
}

function markMainDrop(at) {
  clearDropMarks();
  const kids = [...$("#chain").children];
  if (at < kids.length) kids[at].classList.add("drop-before");
  else if (kids.length) kids[kids.length - 1].classList.add("drop-after");
}

function clearDropMarks() {
  for (const el of document.querySelectorAll(".drop-before, .drop-after")) el.classList.remove("drop-before", "drop-after");
  for (const el of document.querySelectorAll(".drop-target")) el.classList.remove("drop-target");
}

function setupBoard() {
  const board = $("#board");
  // Soltar na área livre da pedaleira (inclusive no cabeçalho/borda da divisão):
  // entra na cadeia principal na posição do cursor.
  board.addEventListener("dragover", (e) => {
    if (!e.dataTransfer.types.includes("application/x-chaincap-file") && !dragUid) return;
    e.preventDefault();
    markMainDrop(mainIndexAt(e.clientX));
  });
  board.addEventListener("dragleave", (e) => { if (!board.contains(e.relatedTarget)) clearDropMarks(); });
  board.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length) return; // tratado na janela
    if (!e.dataTransfer.getData("application/x-chaincap-file") && !dragUid) return;
    e.preventDefault();
    clearDropMarks();
    dropInto(state.chain, mainIndexAt(e.clientX), e);
  });
  $("#split-chain").onclick = splitChain;
  $("#clear-chain").onclick = () => {
    if (!state.chain.length) return;
    const backup = state.chain;
    state.chain = [];
    commit();
    showUndo("Cadeia limpa.", backup);
  };
}

function stagePayload(s) {
  return {
    type: s.type, file: s.job ? null : s.file, job: s.job || null,
    input_gain_db: s.input_gain_db, output_gain_db: s.output_gain_db,
    ir_max_ms: s.type === "ir" ? s.ir_max_ms : null, ir_normalize: s.ir_normalize, enabled: s.enabled,
  };
}

function stagesPayload(nodes = state.chain) {
  return nodes.map((n) => isSplit(n)
    ? {
        type: "split",
        branches: n.branches.map((b) => ({
          level_db: b.level_db, invert: b.invert, enabled: b.enabled,
          stages: b.stages.map(stagePayload),
        })),
      }
    : stagePayload(n));
}

// ------------------------------------------------------------ preview

const DI_KEY = "chaincap.di";

function renderDiSelect() {
  const sel = $("#di-select");
  // A escolha do usuário fica salva (o listener de change grava); se o arquivo
  // ainda existir, ela vale. Senão, mantém o que está selecionado.
  let cur = sel.value;
  try { const saved = localStorage.getItem(DI_KEY); if (saved !== null) cur = saved; } catch {}
  sel.innerHTML = "";
  for (const f of state.files.di) {
    const o = document.createElement("option");
    o.value = f.name;
    o.textContent = stem(f.name);
    sel.append(o);
  }
  const builtin = state.status && state.status.builtin_di;
  if ([...sel.options].some((x) => x.value === cur)) sel.value = cur;
  else if (builtin && [...sel.options].some((x) => x.value === builtin)) sel.value = builtin;
}

function playBlob(blob) {
  if (state.audioUrl) URL.revokeObjectURL(state.audioUrl);
  state.audioUrl = URL.createObjectURL(blob);
  const p = $("#player");
  p.src = state.audioUrl;
  p.play().catch(() => {});
}

async function preview(stages, label, btn) {
  const note = $("#listen-note");
  const original = btn ? btn.textContent : null;
  if (btn) { btn.disabled = true; btn.textContent = "Processando…"; }
  note.className = "note";
  note.textContent = `Processando ${label}…`;
  try {
    const res = await postJSON("/api/preview", {
      stages: stagesPayload(stages), di: $("#di-select").value || null, seconds: Number($("#di-seconds").value),
    });
    const peak = res.headers.get("X-Chain-Peak-Dbfs");
    const warns = JSON.parse(res.headers.get("X-Chain-Warnings") || "[]");
    playBlob(await res.blob());
    note.textContent = `Tocando ${label}. Pico real: ${peak} dBFS.` + (warns.length ? " " + warns.join(" ") : "");
    note.className = warns.length ? "note warn" : "note";
  } catch (e) {
    note.textContent = e.message;
    note.className = "note err";
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = original; }
  }
}

function setupPreview() {
  $("#di-select").addEventListener("change", (e) => {
    try { localStorage.setItem(DI_KEY, e.target.value); } catch {}
  });
  $("#play-chain").onclick = (e) => preview(state.chain, "a cadeia", e.currentTarget);
  $("#play-di").onclick = async () => {
    const note = $("#listen-note");
    try {
      const q = new URLSearchParams({ seconds: $("#di-seconds").value, di: $("#di-select").value });
      const res = await api(`/api/di-clip?${q}`);
      playBlob(await res.blob());
      note.className = "note";
      note.textContent = "Tocando a guitarra crua, sem a cadeia.";
    } catch (e) {
      note.textContent = e.message;
      note.className = "note err";
    }
  };
}

// ------------------------------------------------------------ treino

function setupTrain() {
  const fs = $("#arch");
  for (const a of state.status.archs) {
    const l = document.createElement("label");
    l.innerHTML = `<input type="radio" name="arch" value="${a}">${a}<small>${ARCH_INFO[a] || ""}</small>`;
    fs.append(l);
  }
  const saved = (() => { try { return JSON.parse(localStorage.getItem("chaincap.train") || "{}"); } catch { return {}; } })();
  const arch = saved.arch && state.status.archs.includes(saved.arch) ? saved.arch : "standard";
  fs.querySelector(`input[value="${arch}"]`).checked = true;
  if (saved.name) $("#job-name").value = saved.name;
  if (saved.epochs) $("#epochs").value = saved.epochs;
  if (saved.gpu) $("#gpu-price").value = saved.gpu;
  const persist = () => {
    try {
      localStorage.setItem("chaincap.train", JSON.stringify({
        arch: fs.querySelector("input:checked").value, name: $("#job-name").value,
        epochs: $("#epochs").value, gpu: $("#gpu-price").value,
      }));
    } catch {}
  };
  fs.addEventListener("change", persist);
  for (const id of ["#job-name", "#epochs", "#gpu-price"]) $(id).addEventListener("change", persist);

  $("#train").onclick = async (e) => {
    const btn = e.currentTarget;
    btn.disabled = true;
    try {
      const body = {
        name: $("#job-name").value.trim() || "Minha cadeia",
        arch: fs.querySelector("input:checked").value,
        epochs: Number($("#epochs").value) || 100,
        gpu_usd_per_hour: $("#gpu-price").value ? Number($("#gpu-price").value) : null,
        stages: stagesPayload(),
      };
      await postJSON("/api/jobs", body);
      toast("Treino na fila");
      await refreshJobs();
    } catch (err) {
      toast(err.message, true);
    } finally {
      btn.disabled = false;
      renderTrainNote();
    }
  };
}

function renderTrainNote() {
  const note = $("#train-note");
  const btn = $("#train");
  if (!note || !state.status) return;
  const enabled = signalPaths().flat();
  const missing = enabled.filter(stageMissing);
  let msg = "", cls = "note";
  if (!state.status.v3.valid) { msg = "Baixe o sinal de treino no topo da página para poder treinar."; cls = "note warn"; }
  else if (!enabled.length) msg = "Adicione pelo menos um estágio ligado à cadeia.";
  else if (missing.length) { msg = "Um dos estágios aponta para um arquivo que foi apagado."; cls = "note err"; }
  else if (!state.status.gpu) { msg = "Sem GPU, cada época leva alguns minutos. Para 100 épocas, use o Colab."; cls = "note warn"; }
  else msg = "O treino roda em segundo plano; você pode continuar mexendo na cadeia.";
  note.textContent = msg;
  note.className = cls;
  btn.disabled = !state.status.v3.valid || !enabled.length || missing.length > 0;
}

// ------------------------------------------------------------ jobs

const STATUS_LABEL = {
  queued: "Na fila", running: "Treinando", canceling: "Cancelando", done: "Pronto",
  failed: "Falhou", rejected: "Reprovado", canceled: "Cancelado",
};

async function refreshJobs() {
  state.jobs = await getJSON("/api/jobs");
  // A lâmpada piloto "respira" enquanto um treino está rodando.
  const pilot = $("#pilot");
  if (pilot) pilot.classList.toggle("busy", state.jobs.some((j) => j.status === "running"));
  await renderJobs();
  renderTrainNote();
}

async function renderJobs() {
  const ul = $("#jobs");
  ul.innerHTML = "";
  if (!state.jobs.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "Nenhum treino ainda. Monte a cadeia, ouça e clique em Treinar captura.";
    ul.append(li);
    return;
  }
  for (const j of state.jobs) {
    const li = document.createElement("li");
    li.className = "job";
    const r = j.report, p = j.progress;
    const head = document.createElement("div");
    head.innerHTML = `<div class="job-title"></div><div class="job-sub"></div>`;
    head.querySelector(".job-title").textContent = j.name;
    const subs = [j.arch, `${j.stages.length} estágios`, `${j.epochs} épocas`];
    if (r) {
      subs.push(`${Math.round(r.train_seconds / 60)} min`);
      if (r.gpu_cost_usd != null) subs.push(`US$ ${r.gpu_cost_usd.toFixed(3)}`);
    }
    head.querySelector(".job-sub").textContent = subs.join(", ");
    const st = document.createElement("div");
    st.className = `job-state ${j.status}`;
    st.textContent = STATUS_LABEL[j.status] || j.status;
    if (r) {
      const e = document.createElement("span");
      e.className = `esr ${r.esr_verdict}`;
      e.textContent = `  ESR ${r.esr.toExponential(2)} (${r.esr_verdict})`;
      st.append(e);
    } else if (p && p.best_val_loss != null && j.status === "running") {
      st.textContent += `, ESR ${p.best_val_loss.toExponential(2)}`;
    }
    li.append(head, st);

    if (j.status === "running" || j.status === "queued" || j.status === "canceling") {
      const bar = document.createElement("div");
      bar.className = "progress";
      const pct = p && p.max_epochs ? Math.min(100, (100 * (p.epoch + (p.val_loss != null ? 1 : 0))) / p.max_epochs) : 0;
      bar.innerHTML = `<span style="width:${pct}%"></span>`;
      bar.setAttribute("role", "progressbar");
      bar.setAttribute("aria-valuenow", String(Math.round(pct)));
      bar.setAttribute("aria-label", `Progresso de ${j.name}`);
      li.append(bar);
    }
    if (r && r.warnings && r.warnings.length) {
      const w = document.createElement("div");
      w.className = "job-warn";
      w.textContent = r.warnings.join(" ");
      li.append(w);
    }
    if (j.error) {
      const w = document.createElement("div");
      w.className = "job-warn";
      w.textContent = j.error;
      li.append(w);
    }

    const actions = document.createElement("div");
    actions.className = "job-actions";
    const btn = (label, fn, cls = "small") => {
      const b = document.createElement("button");
      b.type = "button"; b.className = cls; b.textContent = label; b.onclick = fn;
      actions.append(b);
      return b;
    };
    const link = (label, file) => {
      if (!j.files.includes(file)) return;
      const a = document.createElement("a");
      a.className = "small"; a.textContent = label;
      a.href = withToken(`/api/jobs/${j.id}/files/${file}`); a.download = file;
      a.addEventListener("click", async (e) => {
        const d = desktopApi();
        if (!d) return; // navegador: download normal
        e.preventDefault();
        const r = await d.save_job_file(j.id, file);
        if (r && r.saved) toast(`Salvo em ${r.saved}`);
        else if (r && r.error) toast(r.error, true);
      });
      actions.append(a);
    };
    if (j.status === "done" || j.status === "rejected") {
      btn("Ouvir resultado", (e) => preview([{ type: "nam", job: j.id, input_gain_db: 0, output_gain_db: 0, enabled: true }], `o modelo “${j.name}”`, e.currentTarget));
      btn("Ouvir cadeia original", (e) => preview(j.stages.map((s) => ({ ...s })), `a cadeia original de “${j.name}”`, e.currentTarget));
      link("Baixar para o pedal (.a1.nam)", "chain.a1.nam");
      link("Baixar .nam atual", "chain.nam");
      link("Relatório", "report.json");
      btn("Usar na cadeia", () => { addStage("nam", null, { job: j.id, label: j.name, arch: j.arch }); toast(`“${j.name}” adicionado à cadeia`); });
    }
    if (j.status === "queued" || j.status === "running") {
      btn("Cancelar", async () => {
        try { await api(`/api/jobs/${j.id}/cancel`, { method: "POST" }); await refreshJobs(); } catch (e) { toast(e.message, true); }
      }, "small danger");
    }
    const logOpen = state.openLogs.has(j.id);
    btn(logOpen ? "Esconder log" : "Ver log", () => {
      logOpen ? state.openLogs.delete(j.id) : state.openLogs.add(j.id);
      renderJobs();
    });
    if (!["queued", "running", "canceling"].includes(j.status)) {
      btn("Apagar", async () => {
        try { await api(`/api/jobs/${j.id}`, { method: "DELETE" }); await refreshJobs(); } catch (e) { toast(e.message, true); }
      }, "small danger");
    }
    li.append(actions);
    if (logOpen) {
      const pre = document.createElement("pre");
      pre.className = "job-log";
      pre.textContent = "Carregando…";
      li.append(pre);
      getJSON(`/api/jobs/${j.id}/log`).then((d) => {
        pre.textContent = d.lines.join("\n") || "(vazio)";
        pre.scrollTop = pre.scrollHeight;
      });
    }
    ul.append(li);
  }
}

// ------------------------------------------------------------ boot

async function boot() {
  loadChain();
  setupLibrary();
  setupBoard();
  setupPreview();
  try {
    await refreshStatus();
    if (state.status.v3.downloading) pollStatus();
    setupTrain();
    await Promise.all([refreshFiles(), refreshJobs()]);
    renderChain();
  } catch (e) {
    toast(`Não consegui falar com o servidor: ${e.message}`, true);
  }
  setInterval(() => {
    if (state.jobs.some((j) => ["queued", "running", "canceling"].includes(j.status))) refreshJobs();
  }, 3000);
  // Avisa o app que a página está aberta (no modo navegador ele se encerra sem isso).
  const ping = () => api("/api/ping", { method: "POST" }).catch(() => {});
  ping();
  setInterval(ping, 10000);
}

boot();
