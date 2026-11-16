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
const DROP_TEXT = {
  nam: "Solte arquivos .nam aqui ou clique para escolher",
  ir: "Solte IRs .wav aqui ou clique para escolher",
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
    if (!f.error) {
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

async function uploadFiles(fileList, { addToChainAt = null } = {}) {
  const added = [];
  for (const file of fileList) {
    const ext = file.name.toLowerCase().split(".").pop();
    if (ext !== "nam" && ext !== "wav") {
      toast(`${file.name}: envie capturas .nam ou IRs .wav`, true);
      continue;
    }
    // .nam vai para capturas; .wav é IR de caixa
    const kind = ext === "nam" ? "nam" : "ir";
    const fd = new FormData();
    fd.append("kind", kind);
    fd.append("file", file);
    try {
      const r = await api("/api/files", { method: "POST", body: fd });
      const info = await r.json();
      toast(`${info.name} enviado`);
      added.push(info);
    } catch (e) {
      toast(e.message, true);
    }
  }
  await refreshFiles();
  if (addToChainAt != null && added.length) {
    let at = Math.min(addToChainAt, state.chain.length);
    for (const info of added) {
      addStage(info.kind, info.name);
      state.chain.splice(at++, 0, state.chain.pop());
    }
    saveChain();
    renderChain();
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
    const pedal = e.target.closest && e.target.closest(".pedal");
    let at = null;
    if (pedal) {
      const i = state.chain.findIndex((s) => s.uid === pedal.dataset.uid);
      const r = pedal.getBoundingClientRect();
      at = i + (e.clientX > r.left + r.width / 2 ? 1 : 0);
    } else if (e.target.closest && e.target.closest("#board")) {
      at = state.chain.length;
    }
    uploadFiles(e.dataTransfer.files, { addToChainAt: at });
  });
}

// ------------------------------------------------------------ cadeia

function addStage(type, file, extra = {}) {
  state.chain.push({
    uid: uid(), type, file, input_gain_db: 0, output_gain_db: 0,
    ir_max_ms: type === "ir" ? 20 : null, ir_normalize: true, enabled: true, ...extra,
  });
  saveChain();
  renderChain();
}

function stageMissing(st) {
  if (st.job) return !state.jobs.some((j) => j.id === st.job && j.files.includes("chain.nam"));
  return !fileInfo(st.type, st.file);
}

function makeKnob(label, value, onChange) {
  const wrap = document.createElement("div");
  wrap.className = "knob-wrap";
  const knob = document.createElement("div");
  knob.className = "knob";
  knob.tabIndex = 0;
  knob.setAttribute("role", "slider");
  knob.setAttribute("aria-label", label);
  knob.setAttribute("aria-valuemin", "-24");
  knob.setAttribute("aria-valuemax", "24");
  const lab = document.createElement("span");
  lab.className = "knob-label";
  lab.textContent = label;
  const val = document.createElement("span");
  val.className = "knob-value";
  let v = value;
  const set = (nv, commit = true) => {
    v = Math.max(-24, Math.min(24, Math.round(nv * 2) / 2));
    knob.style.setProperty("--angle", `${(v / 24) * 135}deg`);
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
    if (e.key === "Home") { e.preventDefault(); set(-24); }
    if (e.key === "End") { e.preventDefault(); set(24); }
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

function renderChain() {
  const ol = $("#chain");
  ol.innerHTML = "";
  const tpl = $("#pedal-tpl");
  state.chain.forEach((st, i) => {
    const li = tpl.content.firstElementChild.cloneNode(true);
    li.dataset.kind = st.type;
    li.dataset.uid = st.uid;
    if (st.job) li.dataset.trained = "";
    li.classList.toggle("bypassed", !st.enabled);
    const info = st.job ? null : fileInfo(st.type, st.file);
    $(".kind", li).textContent = st.job ? "Modelo treinado" : KIND_LABEL[st.type];
    $(".pedal-name", li).textContent = st.label || (info && info.title) || stem(st.file || "");
    const meta = $(".pedal-meta", li);
    if (stageMissing(st)) {
      meta.textContent = "Arquivo não está mais na biblioteca";
    } else if (info) {
      meta.textContent = describeFile(info);
    } else {
      meta.textContent = st.arch || "";
    }
    const knobs = $(".knobs", li);
    knobs.append(
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
    fs.onclick = () => { st.enabled = !st.enabled; saveChain(); renderChain(); };
    $(".remove", li).onclick = () => { state.chain.splice(i, 1); saveChain(); renderChain(); };
    const left = $(".move-left", li), right = $(".move-right", li);
    left.disabled = i === 0;
    right.disabled = i === state.chain.length - 1;
    left.style.visibility = i === 0 ? "hidden" : "";
    right.style.visibility = i === state.chain.length - 1 ? "hidden" : "";
    left.onclick = () => moveStage(i, i - 1, ".move-left");
    right.onclick = () => moveStage(i, i + 1, ".move-right");

    for (const ctl of li.querySelectorAll("input, button")) {
      ctl.addEventListener("pointerdown", () => { controlActive = true; });
    }
    li.addEventListener("dragstart", (e) => {
      if (controlActive) { e.preventDefault(); return; }
      if (e.target !== li) return;
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
      let target = i + (after ? 1 : 0);
      const fileData = e.dataTransfer.getData("application/x-chaincap-file");
      if (fileData) {
        const f = JSON.parse(fileData);
        addStage(f.kind, f.name);
        const added = state.chain.pop();
        state.chain.splice(target, 0, added);
      } else if (dragUid) {
        const from = state.chain.findIndex((s) => s.uid === dragUid);
        if (from < target) target -= 1;
        const [moved] = state.chain.splice(from, 1);
        state.chain.splice(target, 0, moved);
      }
      saveChain();
      renderChain();
    });
    ol.append(li);
  });
  const enabled = state.chain.filter((s) => s.enabled).length;
  $("#board-hint").textContent = state.chain.length
    ? `${enabled} de ${state.chain.length} estágios ligados. O sinal passa da esquerda para a direita; arraste os pedais para reordenar.`
    : "Monte a cadeia na ordem do sinal: drive, amp, caixa.";
  $("#play-chain").disabled = !enabled;
  renderCabAlert();
  renderTrainNote();
}

// Captura só de amp no fim da cadeia, sem IR depois: o som fica áspero.
function renderCabAlert() {
  const box = $("#board-alert");
  if (!box || !state.status) return;
  const on = state.chain.filter((s) => s.enabled);
  let lastAmp = -1;
  on.forEach((st, i) => {
    const info = !st.job && st.type === "nam" ? fileInfo("nam", st.file) : null;
    if (info && (state.status.amp_only_gear || []).includes(info.gear_type)) lastAmp = i;
  });
  const hasIrAfter = lastAmp >= 0 && on.slice(lastAmp + 1).some((s) => s.type === "ir");
  if (lastAmp < 0 || hasIrAfter) { box.hidden = true; box.innerHTML = ""; return; }
  const ampInfo = fileInfo("nam", on[lastAmp].file);
  const name = (ampInfo && ampInfo.title) || stem(on[lastAmp].file);
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
      const at = state.chain.indexOf(on[lastAmp]) + 1;
      addStage("ir", cab);
      const added = state.chain.pop();
      state.chain.splice(at, 0, added);
      saveChain();
      renderChain();
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

function moveStage(from, to, focusSel) {
  const [s] = state.chain.splice(from, 1);
  state.chain.splice(to, 0, s);
  saveChain();
  renderChain();
  const el = document.querySelector(`.pedal[data-uid="${s.uid}"] ${focusSel}`);
  (el && el.style.visibility !== "hidden" ? el : document.querySelector(`.pedal[data-uid="${s.uid}"] .footswitch`)).focus();
}

function clearDropMarks() {
  for (const el of document.querySelectorAll(".drop-before, .drop-after")) el.classList.remove("drop-before", "drop-after");
}

function setupBoard() {
  const board = $("#board");
  board.addEventListener("dragover", (e) => {
    if (e.dataTransfer.types.includes("application/x-chaincap-file")) e.preventDefault();
  });
  board.addEventListener("drop", (e) => {
    const fileData = e.dataTransfer.getData("application/x-chaincap-file");
    if (!fileData) return;
    e.preventDefault();
    const f = JSON.parse(fileData);
    addStage(f.kind, f.name);
  });
  $("#clear-chain").onclick = () => {
    if (!state.chain.length) return;
    const backup = state.chain;
    state.chain = [];
    saveChain();
    renderChain();
    toast("Cadeia limpa");
    state.lastCleared = backup;
  };
}

function stagesPayload(stages = state.chain) {
  return stages.map((s) => ({
    type: s.type, file: s.job ? null : s.file, job: s.job || null,
    input_gain_db: s.input_gain_db, output_gain_db: s.output_gain_db,
    ir_max_ms: s.type === "ir" ? s.ir_max_ms : null, ir_normalize: s.ir_normalize, enabled: s.enabled,
  }));
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
  const enabled = state.chain.filter((s) => s.enabled);
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
}

boot();
