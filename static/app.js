const PLATFORM_META = {
  instagram: { grad: "linear-gradient(135deg, #f58529, #dd2a7b, #8134af, #515bd4)", label: "Instagram" },
  twitter: { grad: "linear-gradient(135deg, #333333, #000000)", label: "Twitter / X" },
  facebook: { grad: "linear-gradient(135deg, #4f8cff, #1877f2)", label: "Facebook" },
  linkedin: { grad: "linear-gradient(135deg, #2a8fd8, #0a66c2)", label: "LinkedIn" },
};

const platformInput = document.getElementById("platformInput");
const form = document.getElementById("scoreForm");
const submitBtn = document.getElementById("submitBtn");
const lookupForm = document.getElementById("lookupForm");
const lookupBtn = document.getElementById("lookupBtn");
const usernameInput = document.getElementById("usernameInput");
const resultsBox = document.getElementById("results");
const profileCard = document.getElementById("profileCard");
const errorBox = document.getElementById("errorBox");
const groundTruthBadge = document.getElementById("groundTruthBadge");
const manualEntryCard = document.getElementById("manualEntryCard");

// ---------------- Sidebar + theme ----------------
document.getElementById("menuBtn").addEventListener("click", () => {
  document.querySelector(".sidebar").classList.toggle("collapsed");
});

const themeToggle = document.getElementById("themeToggle");
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeToggle.textContent = theme === "dark" ? "☀️" : "🌙";
  localStorage.setItem("idguardian-theme", theme);
}
applyTheme(localStorage.getItem("idguardian-theme") || "light");
themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  applyTheme(current);
});

document.querySelectorAll(".nav-item").forEach((item) => {
  item.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((i) => i.classList.remove("active"));
    item.classList.add("active");
    if (item.dataset.view !== "home") {
      showToast(`"${item.textContent.trim()}" isn't wired up in this research demo — only the lookup flow on Home is implemented.`, false);
    }
  });
});

document.getElementById("manualEntryToggle").addEventListener("click", () => {
  manualEntryCard.hidden = !manualEntryCard.hidden;
  if (!manualEntryCard.hidden) manualEntryCard.scrollIntoView({ behavior: "smooth", block: "start" });
});

document.getElementById("downloadListLink").addEventListener("click", () => {
  const row = document.querySelector(".examples-row");
  row.scrollIntoView({ behavior: "smooth", block: "center" });
  row.classList.add("pulse");
  setTimeout(() => row.classList.remove("pulse"), 900);
});

// ---------------- Platform switching ----------------
function setManualPlatform(platform) {
  platformInput.value = platform;
  document.querySelectorAll(".platform-block").forEach((el) => {
    el.hidden = el.dataset.platform !== platform;
  });
}

function renderExampleChips(platform) {
  const container = document.getElementById("exampleChips");
  const items = (window.SAMPLE_USERNAMES || []).filter((s) => s.platform === platform);
  container.innerHTML = items.map((s) =>
    `<button type="button" class="chip ${s.is_fake ? "chip-fake" : ""}" data-username="${s.username}">@${s.username}</button>`
  ).join("");
  container.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      usernameInput.value = chip.dataset.username;
      lookupForm.requestSubmit();
    });
  });
}

function selectPlatform(platform) {
  const meta = PLATFORM_META[platform];
  if (!meta) return;
  document.getElementById("platformIcon").innerHTML = (window.PLATFORM_ICONS || {})[platform] || "";
  document.getElementById("platformTitle").textContent = meta.label;
  document.documentElement.style.setProperty("--platform-grad", meta.grad);
  usernameInput.placeholder = `Enter a ${meta.label} username or paste a profile URL`;
  document.querySelectorAll(".platform-pill").forEach((b) => b.classList.toggle("active", b.dataset.platform === platform));
  renderExampleChips(platform);
  setManualPlatform(platform);
}

document.querySelectorAll(".platform-pill").forEach((btn) => {
  btn.addEventListener("click", () => selectPlatform(btn.dataset.platform));
});

const firstPillPlatform = document.querySelector(".platform-pill")?.dataset.platform;
if (firstPillPlatform) selectPlatform(firstPillPlatform);

// ---------------- Helpers ----------------
function collectBehavioral(platform) {
  const block = document.querySelector(`.behavioral-block[data-platform="${platform}"]`);
  const out = {};
  block.querySelectorAll("[data-field]").forEach((el) => {
    if (el.type === "checkbox") {
      out[el.dataset.field] = el.checked;
    } else {
      out[el.dataset.field] = parseFloat(el.value) || 0;
    }
  });
  return out;
}

function splitLines(text) {
  return text.split("\n").map((s) => s.trim()).filter(Boolean);
}

function splitHashtags(text) {
  return text.split(/[\s,]+/).map((s) => s.trim()).filter(Boolean);
}

function scoreColor(v) {
  if (v < 0.35) return "var(--safe)";
  if (v < 0.65) return "var(--warn)";
  return "var(--danger)";
}

function setBar(id, value) {
  const bar = document.getElementById(`bar-${id}`);
  const val = document.getElementById(`val-${id}`);
  bar.style.width = `${Math.round(value * 100)}%`;
  bar.style.background = scoreColor(value);
  val.textContent = `${Math.round(value * 100)}%`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function formatNumber(v) {
  if (typeof v !== "number") return v;
  return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(3);
}

function showToast(msg, isError) {
  errorBox.textContent = msg;
  errorBox.classList.toggle("info", !isError);
  errorBox.hidden = false;
  errorBox.scrollIntoView({ behavior: "smooth", block: "center" });
}

function renderProfileData(pd) {
  if (!pd) {
    profileCard.hidden = true;
    return;
  }
  document.getElementById("profileText").innerHTML =
    `<label>${escapeHtml(pd.text_label)}</label><p class="readout-text">${escapeHtml(pd.text) || "(none provided)"}</p>`;

  document.getElementById("profileCaptions").innerHTML = (pd.captions && pd.captions.length)
    ? `<label>Recent captions (${pd.captions.length} used)</label><ul class="caption-list">${
        pd.captions.map((c) => `<li>${escapeHtml(c)}</li>`).join("")}</ul>`
    : `<label>Recent captions</label><p class="readout-text">(none provided)</p>`;

  document.getElementById("profileHashtags").innerHTML = (pd.hashtags && pd.hashtags.length)
    ? `<label>Hashtags</label><p class="readout-text">${
        pd.hashtags.map((h) => `#${escapeHtml(h.replace(/^#/, ""))}`).join(" ")}</p>`
    : "";

  document.getElementById("profileBehavioral").innerHTML = pd.behavioral.map((f) => {
    const display = f.is_bool ? (f.value ? "Yes" : "No") : formatNumber(f.value);
    return `<div class="readout-field"><span>${escapeHtml(f.label)}</span><strong>${display}</strong></div>`;
  }).join("");

  profileCard.hidden = false;
}

function renderResult(data) {
  setBar("lexical", data.lexical_score);
  setBar("semantic", data.semantic_score);
  setBar("behavioral", data.behavioral_score);
  setBar("fusion", data.fusion_score);

  const ex = data.explanations || {};
  document.getElementById("explain-lexical").textContent = ex.lexical ? ex.lexical.summary : "";
  document.getElementById("explain-semantic").textContent = ex.semantic ? ex.semantic.summary : "";
  document.getElementById("explain-behavioral").textContent = ex.behavioral ? ex.behavioral.summary : "";
  document.getElementById("explain-fusion").textContent = ex.fusion ? ex.fusion.summary : "";

  const badge = document.getElementById("verdictBadge");
  const isFake = data.verdict === "Likely Fake";
  const namePrefix = data.username ? `@${data.username} — ` : "";
  badge.textContent = `${namePrefix}${data.verdict} (${Math.round(data.fusion_score * 100)}% fake probability)`;
  badge.className = "verdict-badge " + (isFake ? "fake" : "genuine");

  if (data.ground_truth_archetype) {
    const truth = data.ground_truth_is_fake ? "Fake" : "Genuine";
    groundTruthBadge.textContent = `Ground truth: ${truth} — ${data.ground_truth_archetype}`;
    groundTruthBadge.hidden = false;
  } else {
    groundTruthBadge.hidden = true;
  }

  if (data.platform) selectPlatform(data.platform);
  renderProfileData(data.profile_data);

  resultsBox.hidden = false;
  (data.profile_data ? profileCard : resultsBox).scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------- Form submissions ----------------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.hidden = true;
  const platform = platformInput.value;

  const textBlock = document.querySelector(`.text-block[data-platform="${platform}"] textarea`);
  const payload = {
    platform,
    text: textBlock ? textBlock.value : "",
    captions: splitLines(form.querySelector('textarea[name="captions"]').value),
    hashtags: splitHashtags(form.querySelector('input[name="hashtags"]').value),
    behavioral: collectBehavioral(platform),
  };

  submitBtn.disabled = true;
  submitBtn.textContent = "Scoring...";
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    renderResult(data);
  } catch (err) {
    showToast(`Error: ${err.message}`, true);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Score this profile";
  }
});

lookupForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.hidden = true;
  const query = usernameInput.value.trim();
  if (!query) return;

  lookupBtn.disabled = true;
  lookupBtn.innerHTML = "Looking up...";
  try {
    const res = await fetch(`/api/lookup?query=${encodeURIComponent(query)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    renderResult(data);
  } catch (err) {
    showToast(`Error: ${err.message}`, true);
  } finally {
    lookupBtn.disabled = false;
    lookupBtn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Account';
  }
});
