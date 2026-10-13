"use strict";

const $ = (id) => document.getElementById(id);
const MODE_NAMES = { detect: "🎯 Erkennen", ask: "💬 Fragen", watch: "🚨 Überwachen" };
const PALETTE = ["#58a6ff", "#3fb950", "#f778ba", "#d29922", "#a371f7", "#39c5cf", "#ff7b72", "#e3b341"];

const state = {
  config: null,
  status: null,
  mode: "detect",
  lastResult: null,
  streaming: "",
  history: [],
  shownResult: null, // im Analysebild-Modus angezeigtes Ergebnis
  labelColors: new Map(),
};

// ------------------------------------------------------------------ Hilfsfunktionen
async function api(path, body) {
  const res = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const msg = await res.text();
    toast(`Fehler: ${msg}`);
    throw new Error(msg);
  }
  return res.json();
}

let toastTimer;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 5000);
}

function colorFor(label, attr) {
  if (attr === true) return "#3fb950";
  if (!state.labelColors.has(label)) {
    state.labelColors.set(label, PALETTE[state.labelColors.size % PALETTE.length]);
  }
  return state.labelColors.get(label);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

// ------------------------------------------------------------------ Zeichnen
function drawBoxes(ctx, w, h, result, alpha = 1) {
  if (!result || !result.boxes) return;
  const scale = Math.max(1, w / 900);
  ctx.globalAlpha = alpha;
  ctx.lineWidth = 3 * scale;
  ctx.font = `600 ${15 * scale}px system-ui, sans-serif`;
  ctx.textBaseline = "top";
  for (const b of result.boxes) {
    const [x1, y1, x2, y2] = b.box;
    const X = x1 * w, Y = y1 * h, W = (x2 - x1) * w, H = (y2 - y1) * h;
    const col = colorFor(b.label, b.attr);
    ctx.strokeStyle = col;
    ctx.beginPath();
    ctx.roundRect(X, Y, W, H, 6 * scale);
    ctx.stroke();
    const pad = 4 * scale, th = 20 * scale;
    const tw = ctx.measureText(b.label).width + 2 * pad;
    const ty = Y - th >= 30 * scale ? Y - th : Y; // oben im Bild Label in die Box legen (HUD)
    ctx.fillStyle = col;
    ctx.fillRect(X - ctx.lineWidth / 2, ty, tw, th);
    ctx.fillStyle = "#0d1117";
    ctx.fillText(b.label, X - ctx.lineWidth / 2 + pad, ty + 2.5 * scale);
  }
  ctx.globalAlpha = 1;
}

function redrawOverlay() {
  const cv = $("overlay");
  const wrap = $("videoWrap");
  const w = wrap.clientWidth, h = wrap.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const frozen = $("optFrozen").checked;
  const r = frozen ? state.shownResult : state.lastResult;
  if (!r) return;
  // Im Livebild werden alte Ergebnisse langsam ausgeblendet
  let alpha = 1;
  if (!frozen) {
    const age = Date.now() / 1000 - r.ts;
    const ttl = state.status?.continuous ? 4 + r.latency_ms / 1000 : 30;
    alpha = Math.max(0.15, Math.min(1, 1 - (age - ttl) / 4));
  }
  drawBoxes(ctx, w, h, r, alpha);
}

function loop() {
  redrawOverlay();
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ Darstellung der Ergebnisse
function renderCounts(r) {
  const el = $("counts");
  el.innerHTML = "";
  if (!r || r.mode !== "detect") return;
  const entries = Object.entries(r.counts || {});
  if (!entries.length) {
    el.innerHTML = `<span class="count-chip" style="border-color:#8b949e">nichts gefunden</span>`;
    return;
  }
  for (const [label, n] of entries) {
    const attr = r.boxes.find((b) => b.label === label)?.attr;
    el.insertAdjacentHTML("beforeend",
      `<span class="count-chip" style="border-color:${colorFor(label, attr)}">${n}× ${escapeHtml(label)}</span>`);
  }
}

function renderAnswer(r) {
  const a = $("answer");
  a.classList.remove("muted");
  if (r.mode === "ask") {
    const text = r.text || "(leere Antwort)";
    // Nicht gestreamte Antworten (z. B. Finger zählen) nur bei Änderung neu schreiben
    if (a.dataset.text !== text || a.dataset.mode !== "ask") {
      a.textContent = text;
      a.classList.toggle("big-answer", text.length <= 24);
      if (state.status?.task?.stream === false) flash(a);
    }
    a.dataset.text = text;
    a.dataset.mode = "ask";
  } else if (r.mode === "watch") {
    a.dataset.mode = r.mode;
    a.classList.remove("big-answer");
    const txt = r.answer === true ? "JA" : r.answer === false ? "nein" : "unklar";
    a.innerHTML = `<div class="big" style="color:${r.answer ? "var(--bad)" : "var(--muted)"}">${txt}</div>` +
      `<div class="muted">${escapeHtml(r.title)}</div>`;
  } else {
    a.dataset.mode = r.mode;
    a.classList.remove("big-answer");
    const parts = Object.entries(r.counts || {}).map(([l, n]) => `${n}× ${escapeHtml(l)}`);
    a.innerHTML = parts.length ? `<div class="big">${parts.join(" · ")}</div>` : `<span class="muted">Keine Objekte gefunden.</span>`;
  }
  $("raw").textContent = r.raw ?? "";
  $("promptUsed").textContent = r.prompt || "(Klassen-Prompt)";
  let meta = `${r.model} · ${r.latency_ms} ms`;
  if (r.stage_ms) meta += ` (Detektion ${r.stage_ms.detect} + Attribut ${r.stage_ms.attribute})`;
  $("resultMeta").textContent = meta;
}

function flash(el) {
  el.classList.remove("changed");
  void el.offsetWidth; // Animation neu starten
  el.classList.add("changed");
}

function renderWatch(r) {
  const banner = $("watchBanner");
  if (r.mode !== "watch") {
    banner.hidden = true;
    return;
  }
  banner.hidden = false;
  banner.className = "watch-banner " + (r.answer ? "yes" : "no");
  banner.textContent = r.answer ? `🚨 ${r.title}` : `${r.title} – nein`;
  if (r.triggered) {
    const w = $("videoWrap");
    w.classList.remove("alarm");
    void w.offsetWidth; // Animation neu starten
    w.classList.add("alarm");
    setTimeout(() => w.classList.remove("alarm"), 2500);
    if ($("optSound").checked) beep();
    if ($("optSpeak").checked) speak(r.title.replace("?", "") + " erkannt");
  }
}

function addHistory(r, important = false) {
  const item = { ...r };
  state.history.unshift(item);
  if (state.history.length > 40) state.history.pop();
  const div = document.createElement("div");
  div.className = "hist" + (r.triggered ? " trigger" : "");
  const cv = document.createElement("canvas");
  const img = new Image();
  img.onload = () => {
    cv.width = img.width;
    cv.height = img.height;
    const ctx = cv.getContext("2d");
    ctx.drawImage(img, 0, 0);
    drawBoxes(ctx, img.width, img.height, r);
  };
  img.src = "data:image/jpeg;base64," + r.thumb;
  let body = "";
  if (r.mode === "ask") body = r.text;
  else if (r.mode === "watch") body = r.answer ? "🚨 JA" : "nein";
  else body = Object.entries(r.counts || {}).map(([l, n]) => `${n}× ${l}`).join(", ") || "nichts gefunden";
  const time = new Date(r.ts * 1000).toLocaleTimeString("de-DE");
  div.innerHTML = `<div></div><div><div class="h-title">${escapeHtml(r.title)} <span class="muted">${time}</span></div>` +
    `<div class="h-body">${escapeHtml(body || "")}</div></div>`;
  div.firstChild.appendChild(cv);
  div.onclick = () => showFrozen(item);
  const list = $("historyList");
  list.prepend(div);
  while (list.children.length > 40) list.lastChild.remove();
}

function showFrozen(r) {
  state.shownResult = r;
  $("frozen").src = "data:image/jpeg;base64," + r.thumb;
  $("optFrozen").checked = true;
  applyFrozen();
  renderAnswer(r);
}

function applyFrozen() {
  const on = $("optFrozen").checked;
  if (on && !state.shownResult) state.shownResult = state.lastResult;
  if (on && state.shownResult) $("frozen").src = "data:image/jpeg;base64," + state.shownResult.thumb;
  $("frozen").hidden = !on || !state.shownResult;
  $("video").hidden = !$("frozen").hidden;
}

// In Dauerschleife nur bei geänderter Antwort in den Verlauf schreiben
function historyKey(r) {
  if (r.mode === "ask") return "a:" + r.text;
  if (r.mode === "watch") return "w:" + r.answer;
  return "d:" + JSON.stringify(r.counts);
}
let lastHistoryKey = null;

function onResult(r) {
  state.lastResult = r;
  if ($("optFrozen").checked) {
    state.shownResult = r;
    applyFrozen();
  }
  renderAnswer(r);
  renderCounts(r);
  renderWatch(r);
  $("hudLatency").textContent = `${r.latency_ms} ms`;
  const key = r.title + historyKey(r);
  if (!state.status?.continuous || key !== lastHistoryKey || r.triggered) {
    addHistory(r);
    if (r.mode === "ask" && $("optSpeak").checked) speak(r.text);
  }
  lastHistoryKey = key;
}

function onStatus(s) {
  const prev = state.status;
  state.status = s;
  const pill = $("modelState");
  const model = state.config?.models.find((m) => m.id === s.model_id);
  if (s.loading) {
    pill.className = "pill loading";
    pill.textContent = `⏳ lade ${s.loading} …`;
  } else if (s.error) {
    pill.className = "pill err";
    pill.textContent = "⚠️ Fehler";
    pill.title = s.error;
    if (s.error !== prev?.error) toast(s.error);
  } else if (model) {
    pill.className = "pill ok";
    pill.textContent = `✔ ${model.name} bereit`;
    pill.title = model.description;
  } else {
    pill.className = "pill";
    pill.textContent = "kein Modell";
  }
  if (s.model_id && $("model").value !== s.model_id && !s.loading) $("model").value = s.model_id;
  $("vramText").textContent = `${s.vram_used.toFixed(1)} / ${s.vram_total.toFixed(1)} GB · Modell ${s.vram_model.toFixed(1)}`;
  $("vramBarAll").style.width = `${(100 * s.vram_used) / s.vram_total}%`;
  $("vramBarModel").style.width = `${(100 * s.vram_model) / s.vram_total}%`;
  $("busyDot").classList.toggle("on", s.busy);
  $("hudRate").textContent = s.continuous ? `${s.runs_per_sec.toFixed(1)} /s` : "Einzelbild";
  const live = $("btnLive");
  live.textContent = s.continuous ? "⏸ Live stoppen" : "▶ Live starten";
  live.className = s.continuous ? "danger" : "primary";
  $("hudMode").textContent = MODE_NAMES[s.task.mode];
  $("hudTitle").textContent = s.task.title;
  if (s.camera !== null) $("camera").value = s.camera;
  $("optMirror").checked = s.mirror;
  document.querySelectorAll(".preset").forEach((b) => b.classList.toggle("active", b.dataset.id === s.preset_id));
  $("customForm").classList.toggle("active", !s.preset_id && s.task.mode === state.mode);
}

// ------------------------------------------------------------------ WebSocket
function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => $("conn").classList.add("on");
  ws.onclose = () => {
    $("conn").classList.remove("on");
    setTimeout(connect, 1500);
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    switch (msg.type) {
      case "status": onStatus(msg); break;
      case "result": onResult(msg); break;
      case "stream_start":
        state.streaming = "";
        $("answer").dataset.text = "";
        $("answer").classList.remove("big-answer");
        $("answer").classList.remove("muted");
        $("answer").innerHTML = '<span class="cursor"></span>';
        break;
      case "token":
        state.streaming += msg.text;
        $("answer").textContent = state.streaming;
        $("answer").insertAdjacentHTML("beforeend", '<span class="cursor"></span>');
        break;
      case "error": toast(msg.message); break;
    }
  };
}

// ------------------------------------------------------------------ Audio & Sprache
let audioCtx;
function beep() {
  audioCtx ??= new AudioContext();
  const t = audioCtx.currentTime;
  for (const [i, f] of [880, 1320].entries()) {
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.frequency.value = f;
    g.gain.setValueAtTime(0.0001, t + i * 0.15);
    g.gain.exponentialRampToValueAtTime(0.3, t + i * 0.15 + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + i * 0.15 + 0.14);
    o.connect(g).connect(audioCtx.destination);
    o.start(t + i * 0.15);
    o.stop(t + i * 0.15 + 0.15);
  }
}

function speak(text) {
  if (!text || !window.speechSynthesis) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "de-DE";
  speechSynthesis.speak(u);
}

// ------------------------------------------------------------------ Presets & Formular
function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  document.querySelectorAll("#customForm [data-for]").forEach((d) => (d.hidden = d.dataset.for !== mode));
  const box = $("presets");
  box.innerHTML = "";
  for (const p of state.config.presets.filter((p) => p.task.mode === mode)) {
    const b = document.createElement("button");
    b.className = "preset";
    b.dataset.id = p.id;
    b.title = p.task.prompt || p.description;
    b.innerHTML = `<span class="ico">${p.icon}</span><span><div class="t">${escapeHtml(p.task.title)}</div>` +
      (p.description ? `<div class="d">${escapeHtml(p.description)}</div>` : "") + `</span>`;
    b.onclick = () => {
      api("/api/task", { preset_id: p.id, run: true });
      fillCustom(p.task);
    };
    box.appendChild(b);
  }
  if (state.status) onStatus(state.status);
}

// Preset-Inhalt ins Formular übernehmen, damit man ihn als Vorlage abwandeln kann
function fillCustom(t) {
  if (t.mode === "detect") {
    $("cClasses").value = t.classes.map((c) => c.query).join(", ");
    $("cAttr").value = t.attribute?.question.replace(" Answer only with yes or no.", "") ?? "";
    $("cYes").value = t.attribute?.yes_label ?? "";
    $("cNo").value = t.attribute?.no_label ?? "";
  } else if (t.mode === "ask") {
    $("cAsk").value = t.prompt;
  } else {
    $("cWatch").value = t.prompt;
  }
}

function customTask() {
  const mode = state.mode;
  if (mode === "detect") {
    const classes = $("cClasses").value.split(",").map((s) => s.trim()).filter(Boolean).map((q) => ({ query: q }));
    const q = $("cAttr").value.trim();
    const task = { mode, title: classes.map((c) => c.query).join(", ") || "Alles erkennen", classes };
    if (q) {
      task.attribute = {
        question: /yes or no/i.test(q) ? q : q + " Answer only with yes or no.",
        yes_label: $("cYes").value.trim() || "✔ " + (classes[0]?.query ?? "ja"),
        no_label: $("cNo").value.trim() || null,
      };
      task.title += " + " + q;
    }
    return task;
  }
  if (mode === "ask") {
    const prompt = $("cAsk").value.trim();
    return prompt && { mode, title: prompt.length > 40 ? prompt.slice(0, 40) + "…" : prompt, prompt, max_new_tokens: 300 };
  }
  const prompt = $("cWatch").value.trim();
  return prompt && { mode, title: prompt, prompt };
}

// ------------------------------------------------------------------ Init
async function init() {
  state.config = await api("/api/config");
  const sel = $("model");
  for (const m of state.config.models) {
    sel.insertAdjacentHTML("beforeend", `<option value="${m.id}">${escapeHtml(m.name)} · ~${m.vram_gb} GB</option>`);
  }
  sel.onchange = () => api("/api/model", { id: sel.value });
  for (const c of state.config.cameras) {
    $("camera").insertAdjacentHTML("beforeend", `<option value="${c.index}">/dev/video${c.index} – ${escapeHtml(c.name)}</option>`);
  }
  $("camera").onchange = () => api("/api/settings", { camera: +$("camera").value });
  $("optMirror").onchange = () => api("/api/settings", { mirror: $("optMirror").checked });
  $("maxSide").value = state.config.status.max_side;
  $("maxSide").onchange = () => api("/api/settings", { max_side: +$("maxSide").value });
  $("interval").oninput = () => ($("intervalOut").textContent = `${$("interval").value} s`);
  $("interval").onchange = () => api("/api/settings", { interval: +$("interval").value });

  document.querySelectorAll("#tabs button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  const initialTask = state.config.status.task;
  setMode(initialTask.mode);
  fillCustom(initialTask);
  onStatus(state.config.status);

  $("customForm").onsubmit = (e) => {
    e.preventDefault();
    const task = customTask();
    if (!task) return toast("Bitte erst einen Prompt eingeben.");
    api("/api/task", { task, run: true });
  };
  $("cAsk").onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $("customForm").requestSubmit();
    }
  };
  $("btnLive").onclick = () => api("/api/run", { action: state.status?.continuous ? "stop" : "start" });
  $("btnOnce").onclick = () => api("/api/run", { action: "once" });
  $("btnSnap").onclick = snapshot;
  $("btnClear").onclick = () => {
    state.history = [];
    $("historyList").innerHTML = "";
  };
  $("optFrozen").onchange = applyFrozen;

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea, select")) return;
    if (e.code === "Space") {
      e.preventDefault();
      $("btnOnce").click();
    } else if (e.key === "l") {
      $("btnLive").click();
    } else if (["1", "2", "3"].includes(e.key)) {
      setMode(["detect", "ask", "watch"][+e.key - 1]);
    }
  });

  connect();
  loop();
}

// Livebild bzw. Analysebild samt Boxen als PNG speichern
function snapshot() {
  const frozen = $("optFrozen").checked;
  const src = frozen ? $("frozen") : $("video");
  const cv = document.createElement("canvas");
  cv.width = src.naturalWidth;
  cv.height = src.naturalHeight;
  const ctx = cv.getContext("2d");
  ctx.drawImage(src, 0, 0);
  drawBoxes(ctx, cv.width, cv.height, frozen ? state.shownResult : state.lastResult);
  const a = document.createElement("a");
  a.download = `vlm-snapshot-${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
  a.href = cv.toDataURL("image/png");
  a.click();
}

init();
