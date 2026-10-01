// IDGuardian dashboard UI logic. Talks to GET /api/lookup for scoring.
// History and Saved Reports go through the /api/history and /api/saved
// endpoints (backed by MongoDB) when the server has a database connection
// (window.DB_AVAILABLE); otherwise they fall back to localStorage so the
// app keeps working without MongoDB configured. Theme/Threshold are always
// localStorage -- they're per-browser UI preferences, not data worth
// persisting server-side.

const THEME_KEY = "idguardian_theme";
const THRESHOLD_KEY = "idguardian_threshold_v1";
const CLIENT_ID_KEY = "idguardian_client_id";
const HISTORY_KEY = "idguardian_history_v1";
const SAVED_KEY = "idguardian_saved_v1";
const HISTORY_MAX = 50;
const DB_AVAILABLE = !!window.DB_AVAILABLE;

function getClientId() {
  let id;
  try { id = localStorage.getItem(CLIENT_ID_KEY); } catch { /* ignore */ }
  if (!id) {
    id = window.crypto && crypto.randomUUID ? crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    try { localStorage.setItem(CLIENT_ID_KEY, id); } catch { /* ignore */ }
  }
  return id;
}
const CLIENT_ID = getClientId();

const PLATFORM_GRADIENTS = {
  instagram: "linear-gradient(135deg, #f58529, #dd2a7b, #8134af, #515bd4)",
  facebook: "linear-gradient(135deg, #1877f2, #0a58ca)",
  linkedin: "linear-gradient(135deg, #0a66c2, #004182)",
  twitter: "linear-gradient(135deg, #111827, #374151)",
};

const QUOTES_GENUINE = [
  "“Trust is built in drops and lost in buckets — this one earned its drops.”",
  "“Real accounts don't need to fake their history — the numbers just fit.”",
  "“Consistency across every layer is the strongest signature of a real person.”",
  "“The quiet, unremarkable normalcy of this profile is exactly what genuine looks like.”",
];
const QUOTES_FAKE = [
  "“Fake profiles are easy to build but hard to make consistent — the cracks show here.”",
  "“When every layer disagrees with real-world norms at once, that's not a coincidence.”",
  "“Synthetic accounts optimize for looking good, not for making sense.”",
  "“The pattern here is engineered, not lived — that's what the models are picking up.”",
];

// The gauge path (`M20,110 A80,80 0 0,1 180,110`) is a true semicircle
// (chord 160 == 2 * radius 80), so its length is exactly pi*r -- but it's
// still measured directly rather than assumed, so the dash-offset math stays
// correct even if the path's `d` ever changes. getTotalLength() works even
// while the element is display:none since path length is pure geometry,
// independent of layout.
function getArcLength() {
  const path = $("trustArc");
  try {
    const len = path.getTotalLength();
    if (len > 0) return len;
  } catch (e) { /* fall through to the precomputed fallback below */ }
  return Math.PI * 80; // radius-80 semicircle, precomputed
}

let currentPlatform = "instagram";
let currentResult = null;
let threshold = 50;

const $ = (id) => document.getElementById(id);
const escapeHtml = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));
const capitalize = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const platformLabel = (p) => (p === "twitter" ? "X (Twitter)" : capitalize(p));
const fmtNum = (n) => {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (Math.abs(n) >= 1000) return Math.round(n).toLocaleString();
  const r = Math.round(n * 100) / 100;
  return String(r);
};
const fmtPct = (frac) => `${Math.round(frac * 100)}%`;

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const icon = theme === "dark" ? "☀️" : "🌙";
  if ($("themeToggle")) $("themeToggle").textContent = icon;
}
function toggleTheme() {
  const cur = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
  const next = cur === "dark" ? "light" : "dark";
  localStorage.setItem(THEME_KEY, next);
  applyTheme(next);
}

// ---------------------------------------------------------------------------
// Toast (lightweight, replaces the removed "view on real platform" link)
// ---------------------------------------------------------------------------
function showToast(msg) {
  let toast = $("__toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "__toast";
    toast.style.cssText = "position:fixed;left:50%;bottom:28px;transform:translateX(-50%) translateY(20px);" +
      "background:#1e293b;color:#fff;padding:11px 18px;border-radius:10px;font-size:.83rem;font-weight:600;" +
      "z-index:100;opacity:0;transition:opacity .25s ease, transform .25s ease;max-width:420px;text-align:center;" +
      "box-shadow:0 8px 24px rgba(0,0,0,.25);";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  requestAnimationFrame(() => {
    toast.style.opacity = "1";
    toast.style.transform = "translateX(-50%) translateY(0)";
  });
  clearTimeout(toast.__timer);
  toast.__timer = setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(-50%) translateY(20px)";
  }, 3200);
}

// ---------------------------------------------------------------------------
// View switching
// ---------------------------------------------------------------------------
function showView(name) {
  document.querySelectorAll(".view").forEach((el) => { el.hidden = el.id !== `view-${name}`; });
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === name);
  });
  // Fire-and-forget: each render function is async (may fetch from the
  // server) and updates the DOM once its own data resolves.
  if (name === "dashboard") renderDashboard();
  if (name === "history") renderHistory();
  if (name === "saved") renderSaved();
  if (name === "settings") renderSettings();
}

// ---------------------------------------------------------------------------
// Platform selection (landing page only -- lookup itself is cross-platform)
// ---------------------------------------------------------------------------
function setPlatform(p) {
  currentPlatform = p;
  document.querySelectorAll(".platform-toggle").forEach((btn) => {
    btn.classList.toggle("ring-2", btn.dataset.platform === p);
    btn.classList.toggle("ring-blue-400", btn.dataset.platform === p);
  });
  document.querySelectorAll(".platform-pill").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.platform === p);
  });
  const iconEl = $("platformIcon");
  if (iconEl) {
    iconEl.innerHTML = window.PLATFORM_ICONS[p] || "";
    iconEl.style.setProperty("--hero-grad", PLATFORM_GRADIENTS[p]);
  }
  if ($("platformTitle")) $("platformTitle").textContent = platformLabel(p);
  renderExampleChips();
  if (!$("sampleUsernamesPanel").hidden) renderExpandedChips();
}

// ---------------------------------------------------------------------------
// Sample username chips
// ---------------------------------------------------------------------------
function chipHtml(entry) {
  const cls = entry.is_fake ? "chip chip-fake" : "chip chip-genuine";
  return `<button type="button" class="${cls}" data-username="${escapeHtml(entry.username)}">@${escapeHtml(entry.username)}</button>`;
}
function renderExampleChips() {
  const box = $("exampleChips");
  const entries = window.SAMPLE_USERNAMES.filter((s) => s.platform === currentPlatform).slice(0, 6);
  box.innerHTML = entries.map(chipHtml).join("") || `<span class="text-xs text-slate-400">No samples for this platform.</span>`;
}
function renderExpandedChips() {
  const box = $("expandedChips");
  const entries = window.SAMPLE_USERNAMES.filter((s) => s.platform === currentPlatform);
  $("sampleUsernamesPlatform").textContent = platformLabel(currentPlatform);
  box.innerHTML = entries.map(chipHtml).join("");
}

// ---------------------------------------------------------------------------
// Lookup flow
// ---------------------------------------------------------------------------
async function doLookup(query) {
  const btn = $("lookupBtn");
  const originalLabel = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Analyzing…";
  $("errorBox").hidden = true;
  try {
    const res = await fetch(`/api/lookup?query=${encodeURIComponent(query)}`);
    const data = await res.json();
    if (!res.ok) {
      $("errorBox").className = "error-box";
      $("errorBox").textContent = data.error || "Something went wrong.";
      $("errorBox").hidden = false;
      return;
    }
    currentResult = data;
    setPlatform(data.platform);
    renderResult(data);
    historyAdd(data).catch(() => { /* history save failure shouldn't break scoring */ });
    $("searchLanding").hidden = true;
    $("resultWrap").hidden = false;
    $("resultWrap").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    $("errorBox").className = "error-box";
    $("errorBox").textContent = "Network error — is the server running?";
    $("errorBox").hidden = false;
  } finally {
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
}

// ---------------------------------------------------------------------------
// Result rendering
// ---------------------------------------------------------------------------
function verdictFor(fusionScore) {
  return fusionScore * 100 >= threshold;
}

function renderResult(data) {
  const isFake = verdictFor(data.fusion_score);
  const trustScore = Math.round((1 - data.fusion_score) * 100);

  // --- Profile header ---
  $("avatarInitial").textContent = data.username.charAt(0).toUpperCase();
  $("avatarInitial").style.setProperty("--hero-grad", PLATFORM_GRADIENTS[data.platform]);
  $("resUsername").textContent = `@${data.username}`;
  const archBadge = $("resArchetypeBadge");
  archBadge.textContent = data.ground_truth_archetype.replace(/_/g, " ");
  archBadge.className = `badge ${data.ground_truth_is_fake ? "badge-red" : "badge-green"}`;
  archBadge.title = "Ground-truth label from the synthetic dataset";
  $("resBio").textContent = data.profile_data.text || "(no bio provided)";

  const statFields = data.profile_data.behavioral.filter((f) => !f.is_bool).slice(0, 3);
  $("resStats").innerHTML = statFields.map((f) => (
    `<div><div class="text-sm font-bold">${fmtNum(f.value)}</div><div class="text-[11px] text-slate-400">${escapeHtml(f.label)}</div></div>`
  )).join("");

  const verdictCard = $("resVerdictCard");
  verdictCard.classList.remove("genuine", "fake");
  verdictCard.classList.add(isFake ? "fake" : "genuine");
  const verdictIcon = $("resVerdictIcon");
  verdictIcon.classList.remove("genuine", "fake");
  verdictIcon.classList.add(isFake ? "fake" : "genuine");
  verdictIcon.textContent = isFake ? "✕" : "✓";
  $("resVerdictLabel").textContent = isFake ? "Likely Fake" : "Likely Genuine";
  $("resVerdictSub").textContent = `${fmtPct(data.fusion_score)} fake probability · ${platformLabel(data.platform)}`;
  $("resAnalyzedOn").textContent = new Date().toLocaleString();

  // --- Trust score gauge ---
  let arcColor = "#EF4444", trustLabel = "Low Trust";
  if (trustScore >= 75) { arcColor = "#10B981"; trustLabel = "High Trust"; }
  else if (trustScore >= 45) { arcColor = "#F59E0B"; trustLabel = "Moderate Trust"; }
  const arc = $("trustArc");
  const arcLen = getArcLength();
  arc.style.transition = "stroke-dashoffset .6s ease, stroke .3s ease";
  arc.style.strokeDasharray = `${arcLen}`;
  arc.style.strokeDashoffset = `${arcLen * (1 - trustScore / 100)}`;
  arc.setAttribute("stroke", arcColor);
  $("trustScoreNum").textContent = trustScore;
  $("trustScoreNum").style.color = arcColor;
  $("trustScoreLabel").textContent = trustLabel;

  const summaryBox = $("trustSummaryBox");
  summaryBox.classList.remove("trust-summary-genuine", "trust-summary-fake");
  summaryBox.classList.add(isFake ? "trust-summary-fake" : "trust-summary-genuine");
  const summaryIcon = $("trustSummaryIcon");
  summaryIcon.classList.remove("trust-summary-genuine", "trust-summary-fake");
  summaryIcon.classList.add(isFake ? "trust-summary-fake" : "trust-summary-genuine");
  summaryIcon.textContent = isFake ? "⚠️" : "✅";
  $("trustSummaryTitle").textContent = isFake
    ? "Multiple risk signals detected" : "Strong authenticity signals";
  $("trustSummarySub").textContent = isFake
    ? "This profile diverges from genuine-account norms across one or more layers."
    : "This profile's text, meaning, and behavior are all consistent with real accounts.";

  // --- Model layer scores ---
  const layers = [
    { name: "Lexical", tag: "TF-IDF + LogReg", score: data.lexical_score, available: true },
    { name: "Semantic", tag: "SBERT + XGBoost", score: data.semantic_score, available: data.semantic_available },
    { name: "Behavioral", tag: "XGBoost", score: data.behavioral_score, available: true },
  ];
  $("layerScoreRows").innerHTML = layers.map((l) => {
    if (!l.available) {
      return `<div>
        <div class="flex items-center justify-between text-sm mb-1">
          <span class="font-semibold">${l.name} <span class="text-xs font-normal text-slate-400">${l.tag}</span></span>
          <span class="badge badge-slate">Unavailable</span>
        </div>
        <div class="bar-track"><div class="layer-bar-fill" style="width:0%;background:#cbd5e1;"></div></div>
      </div>`;
    }
    const pct = Math.round(l.score * 100);
    const color = pct >= 50 ? "#EF4444" : "#10B981";
    return `<div>
      <div class="flex items-center justify-between text-sm mb-1">
        <span class="font-semibold">${l.name} <span class="text-xs font-normal text-slate-400">${l.tag}</span></span>
        <span class="font-bold" style="color:${color}">${pct}%</span>
      </div>
      <div class="bar-track"><div class="layer-bar-fill" style="width:${pct}%;background:${color};"></div></div>
    </div>`;
  }).join("");
  $("fakeProbVal").textContent = fmtPct(data.fusion_score);
  $("layerInfoDot").title = "Each layer's own fake-probability score, before Trust Fusion learns how much to weight it.";

  // --- Classification ---
  const clsBadge = $("classificationBadge");
  clsBadge.textContent = isFake ? "✗ Fake" : "✓ Genuine";
  clsBadge.className = `badge ${isFake ? "badge-red" : "badge-green"}`;

  const confidence = Math.max(data.fusion_score, 1 - data.fusion_score);
  const riskLevel = data.fusion_score < 0.3 ? ["Low", "#10B981"] : data.fusion_score < 0.7 ? ["Medium", "#F59E0B"] : ["High", "#EF4444"];
  const verifiedField = data.profile_data.behavioral.find((f) => f.label === "Verified");
  const rows = [
    ["Category", capitalize(data.ground_truth_archetype.replace(/_/g, " "))],
    ["Platform", platformLabel(data.platform)],
    ["Confidence", fmtPct(confidence)],
    ["Risk Level", `<span class="inline-flex items-center gap-1.5"><span style="width:7px;height:7px;border-radius:50%;background:${riskLevel[1]};display:inline-block;"></span>${riskLevel[0]}</span>`],
    ["Verified Account", verifiedField ? (verifiedField.value ? "Yes" : "No") : "—"],
    ["Posts Analyzed", String(data.profile_data.captions.length)],
    ["Analysis Time", `${data.analysis_time_seconds}s`],
  ];
  $("classificationList").innerHTML = rows.map(([label, value]) => (
    `<div><span class="cl-label">${escapeHtml(label)}</span><span class="cl-value">${value}</span></div>`
  )).join("");

  // --- Lexical card ---
  const lex = data.explanations.lexical;
  $("lexicalSupportBadge").className = `badge ${data.lexical_score >= 0.5 ? "badge-red" : "badge-green"}`;
  $("lexicalSupportBadge").textContent = data.lexical_score >= 0.5 ? "Supports Fake" : "Supports Genuine";
  $("genuineWords").innerHTML = lex.top_genuine_words.length
    ? lex.top_genuine_words.map((w) => `<span class="chip chip-genuine" style="cursor:default;">${escapeHtml(w)}</span>`).join("")
    : `<span class="text-xs text-slate-400">None detected</span>`;
  $("fakeWords").innerHTML = lex.top_fake_words.length
    ? lex.top_fake_words.map((w) => `<span class="chip chip-fake" style="cursor:default;">${escapeHtml(w)}</span>`).join("")
    : `<span class="text-xs text-slate-400">None detected</span>`;
  $("lexicalFooter").textContent = lex.summary;

  // --- Semantic card ---
  const sem = data.explanations.semantic;
  if (data.semantic_available) {
    $("semanticSupportBadge").className = `badge ${data.semantic_score >= 0.5 ? "badge-red" : "badge-green"}`;
    $("semanticSupportBadge").textContent = data.semantic_score >= 0.5 ? "Supports Fake" : "Supports Genuine";
    const simPct = Math.round((1 - data.semantic_score) * 100);
    $("semanticSimVal").textContent = `${simPct}%`;
    $("semanticSimBar").style.width = `${simPct}%`;
  } else {
    $("semanticSupportBadge").className = "badge badge-slate";
    $("semanticSupportBadge").textContent = "Unavailable";
    $("semanticSimVal").textContent = "—";
    $("semanticSimBar").style.width = "0%";
  }
  $("semanticSummary").textContent = sem.summary;

  // --- Behavioral card ---
  const beh = data.explanations.behavioral;
  $("behavioralSupportBadge").className = `badge ${data.behavioral_score >= 0.5 ? "badge-red" : "badge-green"}`;
  $("behavioralSupportBadge").textContent = data.behavioral_score >= 0.5 ? "Supports Fake" : "Supports Genuine";
  $("behavioralTable").innerHTML = beh.top_features.length
    ? beh.top_features.map((f) => `
        <tr>
          <td>${escapeHtml(f.feature)}</td>
          <td>${fmtNum(f.value)}</td>
          <td class="text-slate-400">${fmtNum(f.genuine_typical)}</td>
          <td class="text-right">${f.direction === "fake" ? '<span style="color:#EF4444">🚩 Fake</span>' : '<span style="color:#10B981">✓ Genuine</span>'}</td>
        </tr>`).join("")
    : `<tr><td colspan="4" class="text-slate-400">No standout behavioral signals.</td></tr>`;

  // --- Final conclusion + quote ---
  const lead = isFake
    ? "This profile shows a pattern consistent with synthetic or fake accounts."
    : "This profile shows a pattern consistent with genuine, real accounts.";
  $("finalConclusion").textContent = `${lead} ${data.explanations.fusion.summary}`;
  const quotes = isFake ? QUOTES_FAKE : QUOTES_GENUINE;
  const qIdx = Math.abs(hashCode(data.username)) % quotes.length;
  $("quoteText").textContent = quotes[qIdx];
}

function hashCode(str) {
  let h = 0;
  for (let i = 0; i < str.length; i++) h = (h << 5) - h + str.charCodeAt(i) | 0;
  return h;
}

// ---------------------------------------------------------------------------
// History -- MongoDB via /api/history when available, localStorage fallback
// ---------------------------------------------------------------------------
const isValidEntry = (e) => e && e.result && e.result.username && e.platform && typeof e.fusion_score === "number";

function localList(key) {
  try { return (JSON.parse(localStorage.getItem(key)) || []).filter(isValidEntry); } catch { return []; }
}
function localSave(key, list) { try { localStorage.setItem(key, JSON.stringify(list)); } catch { /* ignore */ } }
function makeLocalEntry(result) {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    platform: result.platform, username: result.username,
    archetype: result.ground_truth_archetype, ground_truth_is_fake: result.ground_truth_is_fake,
    fusion_score: result.fusion_score, ts: Date.now(), result,
  };
}

async function historyGetAll() {
  if (DB_AVAILABLE) {
    try {
      const res = await fetch(`/api/history?client_id=${encodeURIComponent(CLIENT_ID)}`);
      if (res.ok) return await res.json();
    } catch { /* fall through to localStorage */ }
  }
  return localList(HISTORY_KEY);
}
async function historyAdd(result) {
  if (DB_AVAILABLE) {
    const res = await fetch("/api/history", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: CLIENT_ID, result }),
    });
    if (res.ok) return;
  }
  const list = localList(HISTORY_KEY);
  list.unshift(makeLocalEntry(result));
  localSave(HISTORY_KEY, list.slice(0, HISTORY_MAX));
}
async function historyRemove(id) {
  if (DB_AVAILABLE) {
    const res = await fetch(`/api/history/${encodeURIComponent(id)}?client_id=${encodeURIComponent(CLIENT_ID)}`, { method: "DELETE" });
    if (res.ok) return;
  }
  localSave(HISTORY_KEY, localList(HISTORY_KEY).filter((e) => e.id !== id));
}
async function historyClear() {
  if (DB_AVAILABLE) {
    const res = await fetch(`/api/history?client_id=${encodeURIComponent(CLIENT_ID)}`, { method: "DELETE" });
    if (res.ok) return;
  }
  localSave(HISTORY_KEY, []);
}

function historyRowHtml(entry, removable, showPdf) {
  const isFake = verdictFor(entry.fusion_score);
  return `<div class="history-row" data-id="${entry.id}">
    <div class="history-icon">${window.PLATFORM_ICONS[entry.platform] ? `<span class="pill-icon pill-icon-${entry.platform}">${window.PLATFORM_ICONS[entry.platform]}</span>` : ""}</div>
    <div class="history-main">
      <div class="history-name">@${escapeHtml(entry.username)}</div>
      <div class="history-sub">${platformLabel(entry.platform)} · ${new Date(entry.ts).toLocaleString()}</div>
    </div>
    <span class="badge ${isFake ? "badge-red" : "badge-green"}">${isFake ? "Fake" : "Genuine"}</span>
    ${showPdf ? `<button type="button" class="history-pdf" data-pdf="${entry.id}" title="Download PDF">⬇</button>` : ""}
    ${removable ? `<button type="button" class="history-remove" data-remove="${entry.id}">✕</button>` : ""}
  </div>`;
}

// History & Analytics are one merged view: a platform filter (pills, plus
// clicking "Total Analyses" resets to "All") drives both the stat grid and
// the list below, so they always agree with each other.
let historyFilterPlatform = "all";

async function renderHistory() {
  const all = await historyGetAll();
  $("historyEmpty").hidden = all.length > 0;
  $("historyBody").hidden = all.length === 0;
  if (!all.length) return;

  const platformsPresent = [...new Set(all.map((e) => e.platform))];
  if (historyFilterPlatform !== "all" && !platformsPresent.includes(historyFilterPlatform)) {
    historyFilterPlatform = "all"; // the filtered platform's only entries were just removed
  }
  const pills = ["all", ...platformsPresent];
  $("historyPlatformFilter").innerHTML = pills.map((p) => {
    const active = p === historyFilterPlatform;
    const label = p === "all" ? "All Platforms" : platformLabel(p);
    const style = active ? "background:#2563eb;border-color:#2563eb;color:#fff;" : "";
    return `<button type="button" class="chip" data-filter-platform="${p}" style="${style}">${label}</button>`;
  }).join("");
  $("historyPlatformFilter").querySelectorAll("[data-filter-platform]").forEach((btn) => {
    btn.addEventListener("click", () => { historyFilterPlatform = btn.dataset.filterPlatform; renderHistory(); });
  });

  const list = historyFilterPlatform === "all" ? all : all.filter((e) => e.platform === historyFilterPlatform);
  const fake = list.filter((e) => verdictFor(e.fusion_score)).length;
  const avgConf = list.length
    ? Math.round(list.reduce((s, e) => s + Math.max(e.fusion_score, 1 - e.fusion_score), 0) / list.length * 100)
    : 0;
  $("statGrid").innerHTML = [
    `<div class="stat-card" id="totalAnalysesCard" style="cursor:pointer;" title="Click to clear the platform filter">`
      + `<div class="stat-value">${list.length}</div><div class="stat-label">Total Analyses</div></div>`,
    statCardHtml(list.length ? fmtPct(fake / list.length) : "—", "Flagged Fake"),
    statCardHtml(list.length ? fmtPct((list.length - fake) / list.length) : "—", "Flagged Genuine"),
    statCardHtml(`${avgConf}%`, "Avg Confidence"),
  ].join("");
  $("totalAnalysesCard").addEventListener("click", () => { historyFilterPlatform = "all"; renderHistory(); });

  $("historyFilterEmpty").hidden = list.length > 0;
  $("historyList").innerHTML = list.map((e) => historyRowHtml(e, true)).join("");
  $("historyList").querySelectorAll(".history-row").forEach((row) => {
    row.addEventListener("click", (ev) => {
      if (ev.target.closest("[data-remove]")) return;
      const entry = list.find((e) => e.id === row.dataset.id);
      if (!entry) return;
      currentResult = entry.result;
      setPlatform(entry.platform);
      renderResult(entry.result);
      $("searchLanding").hidden = true;
      $("resultWrap").hidden = false;
      showView("home");
    });
  });
  $("historyList").querySelectorAll("[data-remove]").forEach((btn) => {
    btn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      await historyRemove(btn.dataset.remove);
      renderHistory();
    });
  });
}

// ---------------------------------------------------------------------------
// Saved Reports -- MongoDB via /api/saved when available (deduped
// server-side by client_id+platform+username), localStorage fallback.
// ---------------------------------------------------------------------------
async function savedGetAll() {
  if (DB_AVAILABLE) {
    try {
      const res = await fetch(`/api/saved?client_id=${encodeURIComponent(CLIENT_ID)}`);
      if (res.ok) return await res.json();
    } catch { /* fall through to localStorage */ }
  }
  return localList(SAVED_KEY);
}
async function savedAdd(result) {
  if (DB_AVAILABLE) {
    const res = await fetch("/api/saved", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ client_id: CLIENT_ID, result }),
    });
    if (res.ok) return;
  }
  const list = localList(SAVED_KEY).filter((e) => !(e.platform === result.platform && e.username === result.username));
  list.unshift(makeLocalEntry(result));
  localSave(SAVED_KEY, list);
}
async function savedRemove(id) {
  if (DB_AVAILABLE) {
    const res = await fetch(`/api/saved/${encodeURIComponent(id)}?client_id=${encodeURIComponent(CLIENT_ID)}`, { method: "DELETE" });
    if (res.ok) return;
  }
  localSave(SAVED_KEY, localList(SAVED_KEY).filter((e) => e.id !== id));
}
async function savedClear() {
  if (DB_AVAILABLE) {
    const res = await fetch(`/api/saved?client_id=${encodeURIComponent(CLIENT_ID)}`, { method: "DELETE" });
    if (res.ok) return;
  }
  localSave(SAVED_KEY, []);
}

// Renders a saved result as a downloadable PDF -- the same fields shown on
// the Check Profile result page, laid out as plain text/tables rather than
// a screenshot so it stays readable and small. Client-side only (jsPDF via
// CDN): no server-side rendering dependency to add to an already
// memory-constrained deploy.
function downloadReportPdf(data) {
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF();
  const marginL = 18, pageW = 210, maxY = 280;
  let y = 20;

  const ensureSpace = (needed) => {
    if (y + needed > maxY) { doc.addPage(); y = 20; }
  };
  const heading = (text) => {
    ensureSpace(10);
    doc.setFont("helvetica", "bold").setFontSize(12).setTextColor(15, 23, 32);
    doc.text(text, marginL, y);
    y += 7;
  };
  const row = (label, value) => {
    ensureSpace(6);
    doc.setFont("helvetica", "normal").setFontSize(10).setTextColor(71, 83, 107);
    doc.text(label, marginL, y);
    doc.setTextColor(15, 23, 32);
    doc.text(String(value), marginL + 55, y);
    y += 6;
  };
  const paragraph = (text, color = [71, 83, 107]) => {
    doc.setFont("helvetica", "normal").setFontSize(10).setTextColor(...color);
    const lines = doc.splitTextToSize(text, pageW - marginL * 2);
    ensureSpace(lines.length * 5 + 2);
    doc.text(lines, marginL, y);
    y += lines.length * 5 + 2;
  };
  const divider = () => {
    ensureSpace(6);
    doc.setDrawColor(226, 230, 236);
    doc.line(marginL, y, pageW - marginL, y);
    y += 8;
  };

  const isFake = verdictFor(data.fusion_score);
  const trustScore = Math.round((1 - data.fusion_score) * 100);
  const confidence = Math.max(data.fusion_score, 1 - data.fusion_score);
  const riskLevel = data.fusion_score < 0.3 ? "Low" : data.fusion_score < 0.7 ? "Medium" : "High";
  const verifiedField = data.profile_data.behavioral.find((f) => f.label === "Verified");

  // --- Header ---
  doc.setFont("helvetica", "bold").setFontSize(18).setTextColor(37, 99, 235);
  doc.text("IDGuardian", marginL, y);
  doc.setFont("helvetica", "normal").setFontSize(10).setTextColor(100, 116, 139);
  doc.text("Trust Report", marginL, y + 6);
  y += 18;

  doc.setFont("helvetica", "bold").setFontSize(15).setTextColor(15, 23, 32);
  doc.text(`@${data.username}`, marginL, y);
  y += 7;
  doc.setFont("helvetica", "normal").setFontSize(10).setTextColor(100, 116, 139);
  doc.text(`${platformLabel(data.platform)}  ·  Analyzed ${new Date().toLocaleString()}`, marginL, y);
  y += 12;

  // --- Verdict + trust score ---
  const verdictColor = isFake ? [239, 68, 68] : [16, 185, 129];
  doc.setFont("helvetica", "bold").setFontSize(13).setTextColor(...verdictColor);
  doc.text(isFake ? "LIKELY FAKE" : "LIKELY GENUINE", marginL, y);
  doc.setFont("helvetica", "normal").setFontSize(10).setTextColor(71, 83, 107);
  doc.text(`${fmtPct(data.fusion_score)} fake probability`, marginL, y + 6);
  doc.setFont("helvetica", "bold").setFontSize(22).setTextColor(...verdictColor);
  doc.text(`${trustScore}/100`, pageW - marginL, y, { align: "right" });
  doc.setFont("helvetica", "normal").setFontSize(9).setTextColor(100, 116, 139);
  doc.text("Trust Score", pageW - marginL, y + 6, { align: "right" });
  y += 16;
  divider();

  // --- Bio ---
  if (data.profile_data.text) {
    heading("Bio");
    paragraph(data.profile_data.text);
    y += 2;
  }

  // --- Model layer scores ---
  heading("Model Layer Scores");
  row("Lexical (TF-IDF + LogReg)", fmtPct(data.lexical_score));
  row("Semantic (SBERT + XGBoost)", data.semantic_available ? fmtPct(data.semantic_score) : "Unavailable");
  row("Behavioral (XGBoost)", fmtPct(data.behavioral_score));
  row("Final Fake Probability", fmtPct(data.fusion_score));
  y += 4;

  // --- Classification ---
  heading("Classification");
  row("Category", capitalize(data.ground_truth_archetype.replace(/_/g, " ")));
  row("Platform", platformLabel(data.platform));
  row("Confidence", fmtPct(confidence));
  row("Risk Level", riskLevel);
  row("Verified Account", verifiedField ? (verifiedField.value ? "Yes" : "No") : "—");
  row("Posts Analyzed", String(data.profile_data.captions.length));
  row("Analysis Time", `${data.analysis_time_seconds}s`);
  y += 4;
  divider();

  // --- Lexical explanation ---
  heading("Lexical Indicators");
  const lex = data.explanations.lexical;
  row("Genuine-leaning words", lex.top_genuine_words.join(", ") || "None detected");
  row("Fake-leaning words", lex.top_fake_words.join(", ") || "None detected");
  paragraph(lex.summary);
  y += 2;

  // --- Semantic explanation ---
  if (data.semantic_available) {
    heading("Semantic Analysis");
    paragraph(data.explanations.semantic.summary);
    y += 2;
  }

  // --- Behavioral explanation ---
  heading("Behavioral Indicators");
  const beh = data.explanations.behavioral;
  if (beh.top_features.length) {
    ensureSpace(8);
    doc.setFont("helvetica", "bold").setFontSize(9).setTextColor(100, 116, 139);
    doc.text("Feature", marginL, y);
    doc.text("Value", marginL + 75, y);
    doc.text("Typical", marginL + 105, y);
    doc.text("Signal", marginL + 140, y);
    y += 5;
    beh.top_features.forEach((f) => {
      ensureSpace(6);
      doc.setFont("helvetica", "normal").setFontSize(9).setTextColor(15, 23, 32);
      doc.text(f.feature, marginL, y);
      doc.text(fmtNum(f.value), marginL + 75, y);
      doc.setTextColor(100, 116, 139);
      doc.text(fmtNum(f.genuine_typical), marginL + 105, y);
      doc.setTextColor(...(f.direction === "fake" ? [239, 68, 68] : [16, 185, 129]));
      doc.text(f.direction === "fake" ? "Fake" : "Genuine", marginL + 140, y);
      y += 6;
    });
  } else {
    paragraph("No standout behavioral signals.");
  }
  y += 2;
  divider();

  // --- Final conclusion ---
  heading("Final Conclusion");
  const lead = isFake
    ? "This profile shows a pattern consistent with synthetic or fake accounts."
    : "This profile shows a pattern consistent with genuine, real accounts.";
  paragraph(`${lead} ${data.explanations.fusion.summary}`, [15, 23, 32]);

  ensureSpace(14);
  y = Math.max(y, maxY - 10);
  doc.setFont("helvetica", "normal").setFontSize(8).setTextColor(148, 163, 184);
  doc.text("Generated by IDGuardian from a synthetic demo profile -- not a real identity verification.", marginL, y);

  doc.save(`idguardian-${data.platform}-${data.username}.pdf`);
}

async function renderSaved() {
  const list = await savedGetAll();
  $("savedEmpty").hidden = list.length > 0;
  $("savedList").innerHTML = list.map((e) => historyRowHtml(e, true, true)).join("");
  $("savedList").querySelectorAll(".history-row").forEach((row) => {
    row.addEventListener("click", (ev) => {
      if (ev.target.closest("[data-remove]") || ev.target.closest("[data-pdf]")) return;
      const entry = list.find((e) => e.id === row.dataset.id);
      if (!entry) return;
      currentResult = entry.result;
      setPlatform(entry.platform);
      renderResult(entry.result);
      $("searchLanding").hidden = true;
      $("resultWrap").hidden = false;
      showView("home");
    });
  });
  $("savedList").querySelectorAll("[data-remove]").forEach((btn) => {
    btn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      await savedRemove(btn.dataset.remove);
      renderSaved();
    });
  });
  $("savedList").querySelectorAll("[data-pdf]").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      const entry = list.find((e) => e.id === btn.dataset.pdf);
      if (entry) downloadReportPdf(entry.result);
    });
  });
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------
function statCardHtml(value, label) {
  return `<div class="stat-card"><div class="stat-value">${value}</div><div class="stat-label">${escapeHtml(label)}</div></div>`;
}

async function renderDashboard() {
  const list = await historyGetAll();
  $("dashboardEmpty").hidden = list.length > 0;
  $("dashboardBody").hidden = list.length === 0;
  if (!list.length) return;
  const fake = list.filter((e) => verdictFor(e.fusion_score)).length;
  const avgTrust = Math.round(list.reduce((s, e) => s + (1 - e.fusion_score) * 100, 0) / list.length);
  $("dashStatGrid").innerHTML = [
    statCardHtml(list.length, "Total Analyses"),
    statCardHtml(fake, "Likely Fake"),
    statCardHtml(list.length - fake, "Likely Genuine"),
    statCardHtml(`${avgTrust}`, "Avg Trust Score"),
  ].join("");
  $("dashRecent").innerHTML = list.slice(0, 5).map((e) => historyRowHtml(e, false)).join("");
}

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------
async function renderSettings() {
  $("thresholdSlider").value = threshold;
  $("thresholdValue").textContent = `${threshold}%`;
  const list = await historyGetAll();
  $("historyCountLabel").textContent = `${list.length} saved lookups`
    + (DB_AVAILABLE ? " (MongoDB)." : " (this browser only).");
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
function init() {
  applyTheme(localStorage.getItem(THEME_KEY) === "dark" ? "dark" : "light");
  threshold = parseInt(localStorage.getItem(THRESHOLD_KEY), 10) || 50;

  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });

  document.querySelectorAll(".platform-toggle, .platform-pill").forEach((btn) => {
    btn.addEventListener("click", () => setPlatform(btn.dataset.platform));
  });

  $("themeToggle").addEventListener("click", toggleTheme);
  $("settingsThemeToggle").addEventListener("click", toggleTheme);

  $("lookupForm").addEventListener("submit", (ev) => {
    ev.preventDefault();
    const q = $("usernameInput").value.trim();
    if (q) doLookup(q);
  });

  document.body.addEventListener("click", (ev) => {
    const chip = ev.target.closest("[data-username]");
    if (chip) {
      $("usernameInput").value = chip.dataset.username;
      doLookup(chip.dataset.username);
    }
  });

  $("sampleUsernamesToggle").addEventListener("click", () => {
    const panel = $("sampleUsernamesPanel");
    panel.hidden = !panel.hidden;
    if (!panel.hidden) renderExpandedChips();
  });

  $("backToSearch").addEventListener("click", () => {
    $("resultWrap").hidden = true;
    $("searchLanding").hidden = false;
    $("usernameInput").value = "";
    $("usernameInput").focus();
  });

  $("saveReportBtn").addEventListener("click", async () => {
    if (!currentResult) return;
    const btn = $("saveReportBtn");
    const original = btn.textContent;
    try {
      await savedAdd(currentResult);
      btn.textContent = "✓ Saved";
    } catch {
      btn.textContent = "Save failed";
    }
    setTimeout(() => { btn.textContent = original; }, 1500);
  });

  $("platformLinkBtn").addEventListener("click", () => {
    showToast("This is a synthetic demo profile from IDGuardian's research dataset — it doesn't correspond to a real account.");
  });

  $("clearHistoryBtn").addEventListener("click", async () => {
    const list = await historyGetAll();
    if (list.length && !confirm("Clear all analysis history? This can't be undone.")) return;
    await historyClear();
    renderHistory();
  });
  $("settingsClearHistory").addEventListener("click", async () => {
    const list = await historyGetAll();
    if (list.length && !confirm("Clear all analysis history? This can't be undone.")) return;
    await historyClear();
    renderSettings();
  });
  $("clearSavedBtn").addEventListener("click", async () => {
    const list = await savedGetAll();
    if (list.length && !confirm("Clear all saved reports? This can't be undone.")) return;
    await savedClear();
    renderSaved();
  });

  $("thresholdSlider").addEventListener("input", () => {
    threshold = parseInt($("thresholdSlider").value, 10);
    $("thresholdValue").textContent = `${threshold}%`;
    localStorage.setItem(THRESHOLD_KEY, String(threshold));
    if (currentResult && !$("resultWrap").hidden) renderResult(currentResult);
  });

  setPlatform("instagram");
  showView("home");
}

document.addEventListener("DOMContentLoaded", init);
