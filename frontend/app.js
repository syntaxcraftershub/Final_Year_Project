/* ShadowTrace frontend controller.
   - Sample + custom (paste) modes
   - Unified SSE reader over fetch (works for GET sample and POST custom)
   - Live chart, metric cards, sparklines, animated step log, detail drawer
   - Theme toggle, stop control, random flagged example
   All offline, no dependencies. */

const $ = (id) => document.getElementById(id);
const chart = new LiveChart($("chart"));
let controller = null;      // AbortController for the active stream
let steps = [];             // steps of the current run (for the drawer)
const driftHist = [];
const deltaHist = [];

const VERDICT = {
  benign:  { big: "BENIGN",  foot: "on-task",              cls: "benign" },
  gray:    { big: "GRAY",    foot: "uncertain → LLM gate", cls: "gray" },
  flagged: { big: "FLAGGED", foot: "diverged from task",   cls: "flagged" },
};

/* ---------- theme ---------- */
$("themeBtn").addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  document.documentElement.setAttribute("data-theme", cur === "dark" ? "light" : "dark");
  chart.draw();
});
document.documentElement.setAttribute("data-theme",
  window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

/* ---------- mode switch ---------- */
document.querySelectorAll(".seg-btn").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    const custom = b.dataset.mode === "custom";
    $("customPanel").hidden = !custom;
    $("samplePanel").hidden = custom;
  }));

/* ---------- data load ---------- */
async function loadCategories() {
  const cats = await fetch("/api/categories").then((r) => r.json());
  const sel = $("category"); sel.innerHTML = "";
  cats.forEach((c) => sel.add(new Option(`${c.label} (${c.count})`, c.id)));
  await loadTrajectories();
}
async function loadTrajectories() {
  const cat = $("category").value;
  const list = await fetch(`/api/trajectories?category=${encodeURIComponent(cat)}`).then((r) => r.json());
  const sel = $("trajectory"); sel.innerHTML = "";
  list.forEach((t) => sel.add(new Option(`${t.task_id}  ·  ${t.n_steps} steps`, t.task_id)));
}
$("category").addEventListener("change", loadTrajectories);

$("surprise").addEventListener("click", async () => {
  // pick a random privacy-leak / prompt-injection trajectory (delta shines there)
  const cats = ["PRIVACY_LEAKAGE", "PROMPT_INJECTION"];
  const cat = cats[Math.floor(Math.random() * cats.length)];
  const list = await fetch(`/api/trajectories?category=${cat}`).then((r) => r.json());
  const pick = list[Math.floor(Math.random() * list.length)];
  document.querySelector('[data-mode="sample"]').click();
  $("category").value = cat;
  await loadTrajectories();
  $("trajectory").value = pick.task_id;
  run();
});

/* ---------- sliders ---------- */
$("driftHigh").addEventListener("input", (e) => ($("driftOut").textContent = (+e.target.value).toFixed(2)));
$("deltaSpike").addEventListener("input", (e) => ($("deltaOut").textContent = (+e.target.value).toFixed(2)));
$("delay").addEventListener("input", (e) => ($("delayOut").textContent = (+e.target.value).toFixed(2) + "s"));

/* ---------- helpers ---------- */
function setStatus(text, cls) { $("statusText").textContent = text; $("statusPill").className = "status-pill " + (cls || ""); }
function setVerdict(v) {
  const info = VERDICT[v]; const card = $("verdictCard");
  card.className = "card verdict-card " + info.cls;
  $("verdictBig").textContent = info.big;
  $("verdictFoot").textContent = info.foot;
}
function sparkline(el, arr) {
  el.innerHTML = "";
  const tail = arr.slice(-16);
  tail.forEach((v) => { const b = document.createElement("i"); b.style.height = Math.max(2, v * 20) + "px"; el.appendChild(b); });
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ---------- SSE over fetch (GET or POST) ---------- */
async function streamSSE(url, opts, onEvent) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error("HTTP " + res.status);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx); buf = buf.slice(idx + 2);
      let ev = "message", data = "";
      frame.split("\n").forEach((ln) => {
        if (ln.startsWith("event:")) ev = ln.slice(6).trim();
        else if (ln.startsWith("data:")) data += ln.slice(5).trim();
      });
      if (data) onEvent(ev, JSON.parse(data));
    }
  }
}

/* ---------- run ---------- */
function resetView() {
  $("log").querySelector("tbody").innerHTML = "";
  driftHist.length = 0; deltaHist.length = 0; steps = [];
  $("mStep").innerHTML = "0<small>/0</small>";
  $("mDrift").textContent = "0.000";
  $("mDelta").textContent = "0.000";
  $("mSaved").innerHTML = "0<small>/0</small>";
  sparkline($("sparkDrift"), []); sparkline($("sparkDelta"), []);
  setVerdict("benign");
}

async function run() {
  if (controller) controller.abort();
  controller = new AbortController();
  resetView();

  const mode = document.querySelector(".seg-btn.active").dataset.mode;
  const driftHigh = parseFloat($("driftHigh").value);
  const deltaSpike = parseFloat($("deltaSpike").value);
  const delay = parseFloat($("delay").value);

  let url, opts;
  if (mode === "custom") {
    let parsed;
    try { parsed = JSON.parse($("customSteps").value); }
    catch (e) { $("customErr").textContent = "Invalid steps JSON: " + e.message; return; }
    $("customErr").textContent = "";
    url = "/api/stream_custom";
    opts = {
      method: "POST", signal: controller.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        intent: $("customIntent").value, steps: parsed,
        thresholds: { drift_high: driftHigh, delta_spike: deltaSpike, gray_band: 0.05 }, delay,
      }),
    };
  } else {
    const taskId = $("trajectory").value;
    if (!taskId) return;
    const p = new URLSearchParams({ task_id: taskId, drift_high: driftHigh, delta_spike: deltaSpike, gray_band: 0.05, delay });
    url = `/api/stream?${p}`; opts = { signal: controller.signal };
  }

  $("run").disabled = true; $("stop").hidden = false;
  setStatus("streaming…", "running");
  let grayCount = 0, nSteps = 0;

  try {
    await streamSSE(url, opts, (ev, d) => {
      if (ev === "meta") {
        nSteps = d.n_steps;
        const gt = d.label;
        const chip = $("gtChip");
        chip.textContent = gt ? "ground truth: " + gt : "no label";
        chip.className = "chip" + (gt ? " gt-" + gt : "");
        $("metaChip").textContent = `${d.category} · ${d.n_steps} steps`;
        $("intent").textContent = d.instruction;
        chart.reset(d.n_steps, driftHigh, deltaSpike);
        $("mStep").innerHTML = `0<small>/${d.n_steps}</small>`;
        $("mSaved").innerHTML = `0<small>/${d.n_steps}</small>`;
      } else if (ev === "step") {
        steps.push(d);
        chart.push(d.cum_drift, d.delta);
        driftHist.push(d.cum_drift); deltaHist.push(d.delta);
        $("mStep").innerHTML = `${d.step_no}<small>/${nSteps}</small>`;
        $("mDrift").textContent = d.cum_drift.toFixed(3);
        $("mDelta").textContent = d.delta.toFixed(3);
        sparkline($("sparkDrift"), driftHist); sparkline($("sparkDelta"), deltaHist);
        if (d.verdict === "gray") grayCount++;
        const saved = nSteps - grayCount;
        $("mSaved").innerHTML = `${saved}<small>/${nSteps}</small>`;
        setVerdict(d.flagged_so_far ? "flagged" : d.verdict);

        const tr = document.createElement("tr");
        tr.className = "v-" + d.verdict;
        tr.innerHTML =
          `<td>${d.step_no}</td>` +
          `<td><code>${escapeHtml(d.action)}</code></td>` +
          `<td>${d.cum_drift.toFixed(3)}</td>` +
          `<td>${d.delta.toFixed(3)}</td>` +
          `<td><span class="badge ${d.verdict}">${d.verdict}</span></td>` +
          `<td class="reason">${escapeHtml(d.reason)}</td>`;
        tr.addEventListener("click", () => openDrawer(d));
        const tb = $("log").querySelector("tbody");
        tb.appendChild(tr);
        tb.parentElement.parentElement.scrollTop = tb.parentElement.parentElement.scrollHeight;
      } else if (ev === "summary") {
        setVerdict(d.final_verdict);
        const gt = d.ground_truth ? ` · truth ${d.ground_truth}` : "";
        const mark = d.correct === true ? " ✓ correct" : d.correct === false ? " ✗ wrong" : "";
        setStatus(`done · ${d.final_verdict.toUpperCase()}${gt}${mark} · peak Δ ${d.peak_delta}`, "done");
      }
    });
  } catch (e) {
    if (e.name !== "AbortError") setStatus("stream error — backend running?", "error");
  } finally {
    $("run").disabled = false; $("stop").hidden = true; controller = null;
  }
}

/* ---------- drawer ---------- */
function openDrawer(d) {
  $("drawerTitle").textContent = `Step ${d.step_no} · ${d.action}`;
  $("drawerContent").innerHTML =
    row("Verdict", `<span class="badge ${d.verdict}">${d.verdict}</span>`) +
    row("Why", escapeHtml(d.reason)) +
    row("Cumulative drift", d.cum_drift.toFixed(4)) +
    row("Step-to-step delta", d.delta.toFixed(4)) +
    row("What the agent did", `<pre>${escapeHtml(d.sentence)}</pre>`) +
    (d.result ? row("Tool result", `<pre>${escapeHtml(d.result)}</pre>`) : "");
  $("drawer").hidden = false;
}
function row(k, v) { return `<div class="d-row"><div class="k">${k}</div><div class="v">${v}</div></div>`; }
$("drawerClose").addEventListener("click", () => ($("drawer").hidden = true));
$("drawer").addEventListener("click", (e) => { if (e.target.id === "drawer") $("drawer").hidden = true; });

/* ---------- wire ---------- */
$("run").addEventListener("click", run);
$("stop").addEventListener("click", () => { if (controller) controller.abort(); setStatus("stopped", ""); });

loadCategories().catch((e) => setStatus("failed to load data: " + e.message, "error"));
