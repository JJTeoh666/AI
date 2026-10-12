import {AudioTransport, formatTime} from "/transport.js";

const $ = id => document.getElementById(id);
const hosted = document.body.dataset.sites === "true";
let engineBase = hosted ? "http://127.0.0.1:8765" : window.location.origin;
const engineURL = path => new URL(path, engineBase).href;
const state = {token: null, connected: false, audio: null, result: null, resultID: null, renderedURL: null,
  job: null, busy: false, tab: "staff", page: 0, pages: 1, views: {}, urls: {}, viewRequest: 0};
let modelRequest = 0;

function showError(error) { $("error").textContent = error.message || String(error); $("error").hidden = false; }
function clearError() { $("error").hidden = true; }
function message(text) { $("status").textContent = text; }
async function api(path, body, method = "GET") {
  const options = {method, headers: {}};
  if (method !== "GET") {
    options.headers["X-Piano-Token"] = state.token;
    if (body !== undefined) { options.headers["Content-Type"] = "application/json"; options.body = JSON.stringify(body); }
  }
  const response = await fetch(engineURL(path), {...options, credentials: "omit"});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}

function controls() {
  for (const id of ["model", "refreshModels", "browseModel", "chooseAudio", "openResult", "device", "overlap", "blend", "modelThresholds"]) $(id).disabled = state.busy || !state.connected;
  for (const head of ["frame", "onset", "offset"]) $(head + "Threshold").disabled = state.busy || $("modelThresholds").checked;
  $("analyze").disabled = state.busy || !state.audio || !$("model").value;
  $("cancelTask").hidden = !state.busy || !state.job;
  $("taskProgress").hidden = !state.busy;
  $("statusDot").classList.toggle("busy", state.busy);
  $("sourceResult").disabled = !state.result;
  $("sourceOriginal").disabled = !state.audio;
  const playable = transport.source === "result" ? !!state.result : !!state.audio;
  for (const id of ["playPause", "stop", "seek"]) $(id).disabled = !playable;
  for (const id of ["exportJSON", "exportView", "exportWAV"]) $(id).disabled = !state.result || state.busy;
}

function setBusy(value, text) {
  state.busy = value;
  if (text) message(text);
  if (value) $("taskProgress").removeAttribute("value");
  controls();
}

async function upload(path, file) {
  const response = await fetch(engineURL(path), {method: "POST", credentials: "omit", headers: {"X-Piano-Token": state.token,
    "X-Filename": encodeURIComponent(file.name), "Content-Type": "application/octet-stream"}, body: file});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Upload failed.");
  return data;
}

async function waitJob(job, label) {
  state.job = job.id;
  setBusy(true, label);
  try {
    while (true) {
      const status = await api(`/api/jobs/${job.id}`);
      if (status.total > 0) {
        $("taskProgress").max = status.total;
        $("taskProgress").value = status.completed;
        message(`${label} ${Math.round(100 * status.completed / status.total)}%`);
      }
      if (status.status === "complete") return status;
      if (status.status === "cancelled") throw new Error("Task cancelled.");
      if (status.status === "error") throw new Error(status.error);
      await new Promise(resolve => setTimeout(resolve, 650));
    }
  } finally { state.job = null; setBusy(false); }
}

function fillModels(models, preferred = $("model").value) {
  $("model").replaceChildren();
  if (!models.length) $("model").add(new Option("No checkpoints found — browse a .pt file", ""));
  for (const model of models) $("model").add(new Option(model.name, model.id));
  if (models.some(item => item.id === preferred)) $("model").value = preferred;
  previewModel();
  controls();
}

async function previewModel() {
  const request = ++modelRequest;
  if (!$("model").value) { $("modelDetails").textContent = "Train a model first, or browse an existing checkpoint."; return; }
  $("modelDetails").textContent = "Reading model…";
  try {
    const model = await api(`/api/models/${$("model").value}`);
    if (request !== modelRequest) return;
    $("modelBadge").textContent = model.version ? `V${model.version}` : "MODEL";
    $("modelDetails").textContent = `${(model.parameters / 1e6).toFixed(2)}M parameters · ${model.pitch_dependent ? "Pitch-dependent thresholds" : `Frame ${model.thresholds.frame.toFixed(3)} / onset ${model.thresholds.onset.toFixed(3)} / offset ${model.thresholds.offset.toFixed(3)}`}`;
    for (const head of ["frame", "onset", "offset"]) $(head + "Threshold").value = model.thresholds[head].toFixed(6);
  } catch (error) { if (request === modelRequest) { $("modelDetails").textContent = "Checkpoint unavailable"; showError(error); } }
}

async function chooseAudio(file) {
  if (!file || state.busy) return;
  clearError();
  transport.pause();
  setBusy(true, "Uploading recording…");
  try {
    state.audio = await upload("/api/audio", file);
    state.audio.url = engineURL(state.audio.url);
    $("audioName").textContent = file.name;
    $("audioDescription").textContent = `${formatTime(state.audio.duration)} · ready for analysis`;
    if (!state.result || transport.source === "original") selectSource("original", true);
    message("Recording ready. Choose Analyze, or listen to the original.");
  } catch (error) { showError(error); message("Recording upload failed."); }
  finally { setBusy(false); }
}

async function analyze() {
  if (!state.audio || !$("model").value || state.busy) return;
  clearError(); transport.pause(); setBusy(true, "Loading model and preparing analysis…");
  try {
    const options = {audio_id: state.audio.id, model_id: $("model").value, device: $("device").value,
      overlap_seconds: Number($("overlap").value), overlap_blend: $("blend").value,
      use_model_thresholds: $("modelThresholds").checked};
    for (const head of ["frame", "onset", "offset"]) options[head + "_threshold"] = Number($(head + "Threshold").value);
    const job = await api("/api/analyze", options, "POST");
    const complete = await waitJob(job, "Analyzing recording…");
    await showResult(await api(`/api/results/${complete.result_id}`));
    message("Analysis complete. Play the piano result or explore the score.");
  } catch (error) { showError(error); message(error.message); }
  finally { setBusy(false); }
}

async function showResult(data) {
  transport.stop();
  state.result = data.result; state.resultID = data.id; state.renderedURL = null;
  state.page = 0; state.views = {}; state.viewRequest++;
  $("timelineStart").value = 0;
  $("timelineSpan").value = state.result.duration;
  for (const url of Object.values(state.urls)) URL.revokeObjectURL(url);
  state.urls = {};
  $("resultTitle").textContent = state.result.audio ? state.result.audio.split(/[\\/]/).pop() : "Piano transcription";
  $("noteCount").textContent = state.result.notes.length.toLocaleString();
  $("chordCount").textContent = state.result.chords.length.toLocaleString();
  $("recordingDuration").textContent = formatTime(state.result.duration);
  $("resultVersion").textContent = `${state.result.model_version ? "V" + state.result.model_version + " · " : ""}${state.result.model ? state.result.model.split(/[\\/]/).pop() : "Saved result"}`;
  $("resultStats").hidden = false; $("inspection").hidden = false; $("empty").hidden = true;
  const config = state.result.inference_config || {};
  if ([0, 1, 2, 3, 4].includes(config.overlap_seconds)) $("overlap").value = config.overlap_seconds;
  $("blend").value = ["crop", "weighted", "equal"].includes(config.overlap_blend) ? config.overlap_blend : "crop";
  const savedModel = state.result.model?.split(/[\\/]/).pop();
  const matchingOption = Array.from($("model").options).find(option => option.text === savedModel);
  if (matchingOption) { $("model").value = matchingOption.value; await previewModel(); }
  for (const head of ["frame", "onset", "offset"]) {
    const value = state.result.thresholds?.[head];
    if (typeof value === "number") $(head + "Threshold").value = value.toFixed(6);
  }
  selectSource("result", true);
  renderTables(); selectTab("staff"); controls();
}

function viewQuery(kind = state.tab) {
  return new URLSearchParams({view: kind === "timeline" ? "timeline" : "staff", bpm: $("bpm").value,
    page: state.page, names: $("noteNames").checked, start: $("timelineStart").value, span: $("timelineSpan").value});
}

async function loadView(kind) {
  if (!state.result || !["staff", "timeline"].includes(kind)) return;
  const request = ++state.viewRequest;
  const resultID = state.resultID;
  $(kind + "Image").style.opacity = "0.4";
  try {
    const view = await api(`/api/results/${resultID}/view?${viewQuery(kind)}`);
    if (request !== state.viewRequest || resultID !== state.resultID) return;
    if (state.urls[kind]) URL.revokeObjectURL(state.urls[kind]);
    state.urls[kind] = URL.createObjectURL(new Blob([view.svg], {type: "image/svg+xml"}));
    $(kind + "Image").src = state.urls[kind];
    state.views[kind] = view;
    if (kind === "staff") { state.page = view.page; state.pages = view.pages; updatePages(); }
    else {
      $("timelineStart").value = view.start;
      $("timelineSpan").value = Number((view.end - view.start).toFixed(6));
      $("previousWindow").disabled = view.start <= 0;
      $("nextWindow").disabled = view.end >= state.result.duration;
    }
    updateCursor(transport.position, false);
  } catch (error) { showError(error); }
  finally { if (request === state.viewRequest) $(kind + "Image").style.opacity = "1"; }
}

function updatePages() {
  $("pageLabel").textContent = `Page ${state.page + 1} / ${state.pages}`;
  $("previousPage").disabled = state.page === 0;
  $("nextPage").disabled = state.page >= state.pages - 1;
}

function selectTab(kind) {
  state.tab = kind;
  for (const button of document.querySelectorAll("[data-tab]")) {
    const active = button.dataset.tab === kind;
    button.classList.toggle("active", active); button.setAttribute("aria-selected", active);
    button.tabIndex = active ? 0 : -1;
    $("pane-" + button.dataset.tab).hidden = !state.result || !active;
  }
  if (state.result && ["staff", "timeline"].includes(kind)) loadView(kind);
}

function renderTables() {
  if (!state.result) return;
  const search = $("noteSearch").value.trim().toLowerCase().replaceAll("#", "♯");
  $("notesBody").replaceChildren(); $("chordsBody").replaceChildren();
  let count = 0;
  const row = (container, values, time) => {
    const tr = document.createElement("tr"); tr.dataset.time = time; tr.tabIndex = 0;
    tr.setAttribute("aria-label", `Seek to ${time.toFixed(2)} seconds`);
    for (const value of values) { const td = document.createElement("td"); td.textContent = value; tr.append(td); }
    container.append(tr);
  };
  for (const note of state.result.notes) {
    if (search && !note.name.toLowerCase().replaceAll("#", "♯").includes(search) && !String(note.pitch).includes(search.replace("midi ", ""))) continue;
    row($("notesBody"), [note.name, note.pitch, note.start.toFixed(3) + " s", note.end.toFixed(3) + " s",
      (note.end - note.start).toFixed(3) + " s", ((note.confidence ?? 1) * 100).toFixed(1) + "%", note.velocity ?? "—"], note.start);
    count++;
  }
  for (const chord of state.result.chords) row($("chordsBody"), [chord.name, chord.start.toFixed(3) + " s", chord.end.toFixed(3) + " s", (chord.end - chord.start).toFixed(3) + " s"], chord.start);
  $("notesShown").textContent = `${count.toLocaleString()} / ${state.result.notes.length.toLocaleString()} notes · select a row to seek`;
}

function updateCursor(seconds, follow = true) {
  if (!state.result) return;
  const time = Math.max(0, Math.min(seconds, state.result.duration));
  $("inspectTime").textContent = time.toFixed(2) + " s";
  $("activeNotes").textContent = state.result.notes.filter(note => note.start <= time && time < note.end).map(note => note.name).join(", ") || "—";
  $("activeChords").textContent = state.result.chords.filter(chord => chord.start <= time && time < chord.end).map(chord => chord.name).join(", ") || "—";
  for (const kind of ["staff", "timeline"]) {
    const view = state.views[kind]; if (!view) continue;
    const x = kind === "staff" ? (time - view.start) * view.bpm / 60 : time;
    const percent = view.bounds[0] + (x - view.xlim[0]) / (view.xlim[1] - view.xlim[0]) * view.bounds[2];
    $(kind + "Cursor").style.left = `${percent * 100}%`;
    $(kind + "Cursor").hidden = time < view.start || time > view.end;
  }
  if (follow && state.tab === "staff") {
    const page = Math.min(Math.max(0, Math.ceil(state.result.duration / (480 / Number($("bpm").value))) - 1), Math.floor(time / (480 / Number($("bpm").value))));
    if (Number.isFinite(page) && page !== state.page) { state.page = page; loadView("staff"); }
  }
}

function plotSeek(event, kind) {
  const view = state.views[kind]; if (!view) return;
  const rect = $(kind + "Image").getBoundingClientRect();
  const fraction = (event.clientX - rect.left) / rect.width;
  const x = view.xlim[0] + (fraction - view.bounds[0]) / view.bounds[2] * (view.xlim[1] - view.xlim[0]);
  transport.seek(kind === "staff" ? view.start + Math.max(0, x) * 60 / view.bpm : x);
}

async function pianoURL() {
  if (state.renderedURL) return state.renderedURL;
  const resultID = state.resultID;
  if (!resultID) throw new Error("Analyze a recording or open a result first.");
  clearError(); setBusy(true, "Preparing sampled piano playback…");
  try {
    let response = await api(`/api/results/${resultID}/render`, undefined, "POST");
    if (!response.url) response = await waitJob(response, "Preparing piano audio…");
    if (resultID !== state.resultID) throw new Error("The selected result changed.");
    state.renderedURL = engineURL(response.url);
    message("Piano audio ready. Drag the progress bar to move through the recording.");
    return state.renderedURL;
  } finally { setBusy(false); }
}

const transport = new AudioTransport($("audio"), {
  getURL: source => source === "result" ? pianoURL() : Promise.resolve(state.audio.url),
  onError: error => { showError(error); $("playerStatus").textContent = error.message; },
  onChange: snapshot => {
    $("playPause").textContent = snapshot.playing ? "Ⅱ" : "▶";
    $("playPause").setAttribute("aria-label", snapshot.playing ? "Pause" : "Play");
    $("currentTime").textContent = formatTime(snapshot.position);
    $("totalTime").textContent = formatTime(snapshot.duration);
    $("seek").max = snapshot.duration || 0;
    if (!$("seek").matches(":active")) $("seek").value = snapshot.position;
    $("seek").setAttribute("aria-valuetext", `${formatTime(snapshot.position)} of ${formatTime(snapshot.duration)}`);
    $("playerStatus").textContent = snapshot.loading ? "Preparing playback…" : snapshot.playing ? "Playing" : snapshot.duration > 0 ? "Paused · drag to seek" : "No audio loaded";
    updateCursor(snapshot.position);
  }
});
transport.setLevels(50, 0);

function selectSource(source, reset = false) {
  if (source === "result" && !state.result || source === "original" && !state.audio) return;
  transport.select(source, source === "result" ? state.result.duration : state.audio.duration, reset);
  $("sourceResult").classList.toggle("selected", source === "result");
  $("sourceOriginal").classList.toggle("selected", source === "original");
  controls();
}

$("chooseAudio").addEventListener("click", () => $("audioFile").click());
$("audioFile").addEventListener("change", event => { chooseAudio(event.target.files[0]); event.target.value = ""; });
for (const type of ["dragover", "dragenter"]) $("chooseAudio").addEventListener(type, event => { event.preventDefault(); if (!state.busy) $("chooseAudio").classList.add("drag-over"); });
$("chooseAudio").addEventListener("dragleave", () => $("chooseAudio").classList.remove("drag-over"));
$("chooseAudio").addEventListener("drop", event => { event.preventDefault(); $("chooseAudio").classList.remove("drag-over"); chooseAudio(event.dataTransfer.files[0]); });
$("browseModel").addEventListener("click", () => $("modelFile").click());
$("modelFile").addEventListener("change", async event => {
  const file = event.target.files[0]; event.target.value = ""; if (!file) return;
  clearError(); setBusy(true, "Loading checkpoint…");
  try { const model = await upload("/api/models/upload", file); fillModels((await api("/api/models")).models, model.id); message("Model ready."); }
  catch (error) { showError(error); message("Checkpoint upload failed."); }
  finally { setBusy(false); }
});
$("openResult").addEventListener("click", () => $("resultFile").click());
$("resultFile").addEventListener("change", async event => {
  const file = event.target.files[0]; event.target.value = ""; if (!file) return;
  clearError(); transport.pause(); setBusy(true, "Opening saved result…");
  try { await showResult(await upload("/api/results/import", file)); message("Saved result loaded. Select Play to hear the piano transcription."); }
  catch (error) { showError(error); message("Could not open that result."); }
  finally { setBusy(false); }
});
$("refreshModels").addEventListener("click", async () => { try { fillModels((await api("/api/models")).models); } catch (error) { showError(error); } });
$("model").addEventListener("change", () => { previewModel(); controls(); });
$("modelThresholds").addEventListener("change", controls);
$("analyze").addEventListener("click", analyze);
$("cancelTask").addEventListener("click", async () => { if (state.job) { try { await api(`/api/jobs/${state.job}/cancel`, undefined, "POST"); message("Cancelling after the current section…"); } catch (error) { showError(error); } } });
$("playPause").addEventListener("click", () => transport.toggle());
$("stop").addEventListener("click", () => transport.stop());
$("sourceResult").addEventListener("click", () => selectSource("result"));
$("sourceOriginal").addEventListener("click", () => selectSource("original"));
$("seek").addEventListener("input", event => transport.seek(Number(event.target.value)));
for (const id of ["volume", "gain"]) $(id).addEventListener("input", () => {
  $("volumeLabel").textContent = $("volume").value + "%";
  $("gainLabel").textContent = "+" + $("gain").value + " dB";
  transport.setLevels($("volume").value, $("gain").value);
});
for (const button of document.querySelectorAll("[data-tab]")) {
  button.addEventListener("click", () => selectTab(button.dataset.tab));
  button.addEventListener("keydown", event => {
    const kinds = ["staff", "timeline", "notes", "chords"];
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault(); const next = kinds[(kinds.indexOf(state.tab) + (event.key === "ArrowRight" ? 1 : 3)) % 4];
    selectTab(next); $("tab-" + next).focus();
  });
}
$("previousPage").addEventListener("click", () => { state.page = Math.max(0, state.page - 1); loadView("staff"); });
$("nextPage").addEventListener("click", () => { state.page = Math.min(state.pages - 1, state.page + 1); loadView("staff"); });
for (const id of ["timelineStart", "timelineSpan"]) $(id).addEventListener("change", () => {
  if (!Number.isFinite(Number($("timelineStart").value)) || Number($("timelineStart").value) < 0 || !Number.isFinite(Number($("timelineSpan").value)) || Number($("timelineSpan").value) <= 0) {
    showError(new Error("Timeline start must be nonnegative and the visible duration must be positive.")); return;
  }
  loadView("timeline");
});
$("fullTimeline").addEventListener("click", () => { $("timelineStart").value = 0; $("timelineSpan").value = state.result.duration; loadView("timeline"); });
for (const [id, direction] of [["previousWindow", -1], ["nextWindow", 1]]) $(id).addEventListener("click", () => {
  $("timelineStart").value = Math.max(0, Number($("timelineStart").value) + direction * Number($("timelineSpan").value)); loadView("timeline");
});
for (const id of ["bpm", "noteNames"]) $(id).addEventListener("change", () => {
  const bpm = Number($("bpm").value);
  if (!Number.isFinite(bpm) || bpm < 30 || bpm > 300) { showError(new Error("Notation tempo must be 30–300 BPM.")); return; }
  state.page = 0; loadView("staff");
});
$("zoom").addEventListener("input", () => { $("zoomLabel").textContent = $("zoom").value + "%"; $("staffFigure").style.width = $("zoom").value + "%"; });
$("staffFigure").style.width = "125%";
$("noteSearch").addEventListener("input", renderTables);
for (const kind of ["staff", "timeline"]) $(kind + "Image").addEventListener("click", event => plotSeek(event, kind));
for (const id of ["notesBody", "chordsBody"]) {
  const select = event => { const row = event.target.closest("tr[data-time]"); if (row) transport.seek(Number(row.dataset.time)); };
  $(id).addEventListener("click", select);
  $(id).addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(event); } });
}
async function download(path, filename) {
  try {
    const response = await fetch(engineURL(path), {credentials: "omit"});
    if (!response.ok) throw new Error("Download failed.");
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a"); link.href = url; link.download = filename;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  } catch (error) { showError(error); }
}
$("exportJSON").addEventListener("click", () => { if (state.resultID) download(`/api/results/${state.resultID}/export`, "prediction.json"); });
$("exportView").addEventListener("click", () => { if (state.resultID) download(`/api/results/${state.resultID}/view.png?${viewQuery()}`, "piano-view.png"); });
$("exportWAV").addEventListener("click", async () => { try { const url = await pianoURL(); download(url, "piano-result.wav"); } catch (error) { showError(error); } });
document.addEventListener("keydown", event => {
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName) || event.ctrlKey || event.altKey || event.metaKey) return;
  if (event.code === "Space" && !$("playPause").disabled) { event.preventDefault(); transport.toggle(); }
  if (["ArrowLeft", "ArrowRight"].includes(event.key) && !$("seek").disabled) { event.preventDefault(); transport.seek(transport.position + (event.key === "ArrowRight" ? 5 : -5)); }
});
// Native timeupdate is coarse; this keeps the score cursor smooth during playback.
let lastCursor = 0;
function tick(time) {
  if (!$("audio").paused && time - lastCursor > 100) {
    transport.position = $("audio").currentTime; transport.publish(); lastCursor = time;
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

async function connectEngine() {
  clearError(); $("connectEngine").disabled = true;
  try {
    if (hosted) {
      const port = Number($("enginePort").value);
      if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error("Choose a port between 1 and 65535.");
      engineBase = `http://127.0.0.1:${port}`;
    }
    message("Connecting to the piano engine…");
    const bootstrap = await api("/api/bootstrap"); state.token = bootstrap.token; state.connected = true;
    $("soundName").textContent = bootstrap.piano_sound;
    $("device").querySelector("[value=cuda]").disabled = !bootstrap.cuda;
    fillModels(bootstrap.models); controls();
    $("connectionTitle").textContent = "Piano engine connected";
    $("connectionHelp").textContent = "Keep your Python terminal open. Recordings, checkpoints and analysis stay on this computer. Reload this page after restarting the engine.";
    $("connectEngine").textContent = "Connected"; $("enginePort").disabled = true;
    message("Ready. V8.1 is the default when its checkpoint is available.");
  } catch (error) {
    state.connected = false; controls(); $("connectEngine").disabled = false;
    showError(new Error(hosted ? "Could not connect. Run python web_app.py --sites on this computer, check the port, and allow this website's local network access in your browser. Then try Connect again." : error.message));
    message("Piano engine unavailable.");
  }
}
$("connectEngine").addEventListener("click", connectEngine);
controls();
if (hosted) { $("engineConnection").hidden = false; message("Connect the piano engine to begin."); document.querySelector(".local-badge").lastChild.textContent = "Sites · local piano engine"; }
else await connectEngine();

// Optional browser tools use the same player and result state as the controls.
if (document.modelContext?.registerTool) {
  const lifecycle = new AbortController();
  const tools = [
    {name: "get_piano_status", description: "Read the connection, selected model, transcription summary and playback position.",
      inputSchema: {type: "object", properties: {}, additionalProperties: false},
      annotations: {readOnlyHint: true, untrustedContentHint: true},
      execute: () => ({connected: state.connected, model: $("model").selectedOptions[0]?.text || null,
        notes: state.result?.notes.length || 0, chords: state.result?.chords.length || 0,
        duration: state.result?.duration || state.audio?.duration || 0, position: transport.position,
        playing: !$("audio").paused})},
    {name: "seek_piano_playback", description: "Move the visible piano player and score cursor to a time in seconds. Keeps the current play or pause state.",
      inputSchema: {type: "object", properties: {seconds: {type: "number", minimum: 0}}, required: ["seconds"], additionalProperties: false},
      annotations: {readOnlyHint: false, untrustedContentHint: false},
      execute: input => {
        if (!input || typeof input.seconds !== "number" || !Number.isFinite(input.seconds) || input.seconds < 0 || Object.keys(input).some(key => key !== "seconds")) throw new Error("Provide a nonnegative time in seconds.");
        if (!state.result && !state.audio) throw new Error("Load a recording or transcription first.");
        transport.seek(input.seconds);
        return {position: transport.position, playing: !$("audio").paused};
      }}
  ];
  for (const tool of tools) {
    try { Promise.resolve(document.modelContext.registerTool(tool, {signal: lifecycle.signal})).catch(console.warn); }
    catch (error) { console.warn(error); }
  }
  window.addEventListener("pagehide", () => lifecycle.abort(), {once: true});
}
