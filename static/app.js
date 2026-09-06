const platformInput = document.getElementById("platformInput");
const tabButtons = document.querySelectorAll(".tab-btn");
const form = document.getElementById("scoreForm");
const submitBtn = document.getElementById("submitBtn");
const lookupForm = document.getElementById("lookupForm");
const lookupBtn = document.getElementById("lookupBtn");
const usernameInput = document.getElementById("usernameInput");
const resultsBox = document.getElementById("results");
const errorBox = document.getElementById("errorBox");
const groundTruthBadge = document.getElementById("groundTruthBadge");

function setPlatform(platform) {
  platformInput.value = platform;
  tabButtons.forEach((b) => b.classList.toggle("active", b.dataset.platform === platform));
  document.querySelectorAll(".platform-block").forEach((el) => {
    el.hidden = el.dataset.platform !== platform;
  });
}

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => setPlatform(btn.dataset.platform));
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    usernameInput.value = chip.dataset.username;
    lookupForm.requestSubmit();
  });
});

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
  if (v < 0.65) return "#f4c95d";
  return "var(--danger)";
}

function setBar(id, value) {
  const bar = document.getElementById(`bar-${id}`);
  const val = document.getElementById(`val-${id}`);
  bar.style.width = `${Math.round(value * 100)}%`;
  bar.style.background = scoreColor(value);
  val.textContent = `${Math.round(value * 100)}%`;
}

function renderResult(data) {
  setBar("lexical", data.lexical_score);
  setBar("semantic", data.semantic_score);
  setBar("behavioral", data.behavioral_score);
  setBar("fusion", data.fusion_score);

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

  resultsBox.hidden = false;
  resultsBox.scrollIntoView({ behavior: "smooth", block: "start" });
}

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
    errorBox.textContent = `Error: ${err.message}`;
    errorBox.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Score this profile";
  }
});

lookupForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.hidden = true;
  const username = usernameInput.value.trim();
  if (!username) return;

  lookupBtn.disabled = true;
  lookupBtn.textContent = "Looking up...";
  try {
    const res = await fetch(`/api/lookup?username=${encodeURIComponent(username)}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    renderResult(data);
  } catch (err) {
    errorBox.textContent = `Error: ${err.message}`;
    errorBox.hidden = false;
  } finally {
    lookupBtn.disabled = false;
    lookupBtn.textContent = "Look up & score";
  }
});
