const metricsEls = {
  test: document.getElementById("test-accuracy"),
  train: document.getElementById("train-accuracy"),
  size: document.getElementById("dataset-size"),
  status: document.getElementById("status-pill"),
};

const form = document.getElementById("prediction-form");
const resultStage = document.getElementById("result-stage");
const submitBtn = document.getElementById("submit-btn");
const chartTest = document.getElementById("chart-test");
const chartTrain = document.getElementById("chart-train");
const historyList = document.getElementById("history-list");
const modeButtons = Array.from(document.querySelectorAll(".mode-toggle"));
const urlFields = document.getElementById("url-fields");
const manualFields = document.getElementById("manual-fields");
const themeToggle = document.getElementById("theme-toggle");
const inputs = {
  url: document.getElementById("url"),
  title: document.getElementById("title"),
  author: document.getElementById("author"),
  text: document.getElementById("text"),
};
const INDIA_LOCALE = "en-IN";
const INDIA_TIMEZONE = "Asia/Kolkata";

let activeMode = "url";

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => {
    const entities = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return entities[character] || character;
  });
}

function safeUrl(value) {
  return typeof value === "string" && /^https?:\/\//i.test(value) ? value : "";
}

function toFiniteNumber(value) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function formatMetric(value) {
  const numericValue = toFiniteNumber(value);
  if (numericValue !== null) {
    return `${numericValue}%`;
  }
  return typeof value === "string" && value.trim() ? value : "--";
}

function formatCount(value) {
  const numericValue = toFiniteNumber(value);
  if (numericValue !== null) {
    return numericValue.toLocaleString(INDIA_LOCALE);
  }
  return typeof value === "string" && value.trim() ? value : "--";
}

function formatDate(value) {
  if (!value) {
    return "";
  }

  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return String(value);
  }

  return `${parsed.toLocaleString(INDIA_LOCALE, {
    timeZone: INDIA_TIMEZONE,
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  })} IST`;
}

function buildMetricsStatus(metrics) {
  if (metrics.runtime_mode === "saved_snapshot") {
    return "Saved model snapshot";
  }
  if (metrics.runtime_mode === "heuristic_fallback") {
    return "Heuristic fallback mode";
  }
  if (metrics.dataset_file) {
    return `${metrics.dataset_file} ready`;
  }
  if (metrics.model) {
    return metrics.model;
  }
  return "Model ready";
}

function initializeTheme() {
  if (window.VerifiJinTheme) {
    window.VerifiJinTheme.initializeThemeToggle(themeToggle);
  }
}

function setMode(mode) {
  activeMode = mode === "manual" ? "manual" : "url";
  form.dataset.mode = activeMode;

  modeButtons.forEach((button) => {
    const isActive = button.dataset.mode === activeMode;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });

  urlFields.classList.toggle("is-active", activeMode === "url");
  manualFields.classList.toggle("is-active", activeMode === "manual");

  inputs.url.disabled = activeMode !== "url";
  inputs.url.required = activeMode === "url";
  inputs.title.disabled = activeMode !== "manual";
  inputs.author.disabled = activeMode !== "manual";
  inputs.text.disabled = activeMode !== "manual";
  inputs.text.required = activeMode === "manual";

  submitBtn.textContent = activeMode === "url" ? "Analyze URL" : "Analyze Article";
}

async function loadMetrics() {
  try {
    const response = await fetch("/api/metrics");
    if (!response.ok) {
      throw new Error("Model metrics are unavailable.");
    }

    const metrics = await response.json();
    metricsEls.test.textContent = formatMetric(metrics.test_accuracy);
    metricsEls.train.textContent = formatMetric(metrics.train_accuracy);
    metricsEls.size.textContent = formatCount(metrics.dataset_rows);
    metricsEls.status.textContent = buildMetricsStatus(metrics);

    const testAccuracy = toFiniteNumber(metrics.test_accuracy) ?? 0;
    const trainAccuracy = toFiniteNumber(metrics.train_accuracy) ?? 0;

    if (chartTest) {
      chartTest.style.width = `${testAccuracy}%`;
    }
    if (chartTrain) {
      chartTrain.style.width = `${trainAccuracy}%`;
    }
  } catch (error) {
    metricsEls.status.textContent = "Unable to load model snapshot";
  }
}

async function loadHistory() {
  if (!historyList) {
    return;
  }

  try {
    const response = await fetch("/api/history");
    if (!response.ok) {
      throw new Error("History is unavailable.");
    }

    const data = await response.json();
    const items = data.items || [];
    if (!items.length) {
      historyList.innerHTML = '<p class="placeholder">Recent predictions will appear here after analysis.</p>';
      return;
    }

    historyList.innerHTML = items
      .map((item) => {
        const sourceLine = [item.source || item.author || "Unknown source", item.input_mode || "manual"]
          .filter(Boolean)
          .join(" - ");
        return `
          <article class="history-entry">
            <strong>${escapeHtml(item.label || "Unknown")} - ${escapeHtml(item.confidence)}%</strong>
            <span>${escapeHtml(item.title || "Untitled entry")}</span>
            <span>${escapeHtml(sourceLine)} - ${escapeHtml(formatDate(item.created_at || "") || "Unknown time")}</span>
          </article>
        `;
      })
      .join("");
  } catch (error) {
    historyList.innerHTML = '<p class="placeholder">Unable to load prediction history.</p>';
  }
}

function renderSignalGroup(label, items, tone = "positive") {
  if (!items.length) {
    return "";
  }

  const toneClass = tone === "warning" ? " warning" : tone === "neutral" ? " neutral" : "";
  return `
    <div class="signal-group">
      <strong>${escapeHtml(label)}</strong>
      <div class="signal-list">${items
        .map((item) => `<span class="signal-chip${toneClass}">${escapeHtml(item)}</span>`)
        .join("")}</div>
    </div>
  `;
}

function renderVerificationLinks(links) {
  if (!links.length) {
    return "";
  }

  return `
    <div class="verification-block">
      <strong>Verify further</strong>
      <div class="verification-grid">${links
        .map((link) => {
          const href = safeUrl(link.url);
          if (!href) {
            return "";
          }
          return `<a class="verify-link" href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${escapeHtml(link.label || href)}</a>`;
        })
        .join("")}</div>
    </div>
  `;
}

function renderSourceProfile(profile, article) {
  const score = Number(profile.score);
  const tone = score >= 85 ? "high" : score >= 65 ? "medium" : score >= 45 ? "neutral" : "low";
  const publishedAt = formatDate(article.published_at || "");
  const details = [article.domain, publishedAt, article.input_mode === "manual" ? "Manual input" : article.input_mode === "url" ? "Live URL" : ""]
    .filter(Boolean)
    .map((item) => `<span>${escapeHtml(item)}</span>`)
    .join("");

  return `
    <div class="source-card">
      <div class="source-card-head">
        <div>
          <p class="eyebrow">Source Profile</p>
          <h4>${escapeHtml(profile.label || article.source || "Unknown source")}</h4>
        </div>
        <span class="credibility-pill ${tone}">${escapeHtml(profile.tier || "Unverified source")}</span>
      </div>
      <div class="source-details">${details}</div>
      <div class="source-grid">
        <div class="breakdown-card">
          <span>Credibility score</span>
          <strong>${Number.isFinite(score) ? `${score}/100` : "--"}</strong>
        </div>
        <div class="breakdown-card">
          <span>Source type</span>
          <strong>${escapeHtml(profile.kind || "unknown")}</strong>
        </div>
        <div class="breakdown-card">
          <span>Why it matters</span>
          <strong>${escapeHtml(profile.reason || "No source explanation available.")}</strong>
        </div>
      </div>
    </div>
  `;
}

function renderResult(data) {
  let verdictClass = "fake";
  if (data.label === "Real") {
    verdictClass = "real";
  } else if (data.label === "Needs More Context") {
    verdictClass = "neutral";
  }

  const sourceCues = data.analysis?.source_cues || [];
  const reportingCues = data.analysis?.reporting_cues || [];
  const riskCues = data.analysis?.risk_cues || [];
  const metadataCues = data.analysis?.metadata_cues || [];
  const breakdown = data.model_breakdown || {};
  const article = data.article || {};
  const sourceProfile = data.source_profile || {};
  const sourceLink = safeUrl(article.url);
  const preview = article.preview ? `<p class="supporting-text">${escapeHtml(article.preview)}</p>` : "";
  const articleMeta = [article.source, article.domain, formatDate(article.published_at || "")]
    .filter(Boolean)
    .map((item) => `<span>${escapeHtml(item)}</span>`)
    .join("");

  const articleMarkup = article.title || sourceLink
    ? `
      <div class="article-summary">
        <div class="article-meta">${articleMeta}</div>
        <h4>${escapeHtml(article.title || "Extracted article")}</h4>
        <p>${escapeHtml(article.author || "Unknown author")}</p>
        ${sourceLink ? `<a href="${escapeHtml(sourceLink)}" target="_blank" rel="noopener noreferrer">${escapeHtml(sourceLink)}</a>` : ""}
        ${preview}
      </div>
    `
    : "";

  const secondaryModel = breakdown.secondary_model;
  const secondaryMarkup = secondaryModel
    ? `
      <div class="breakdown-card">
        <span>Chunked language model</span>
        <strong>${escapeHtml(secondaryModel.probabilities?.real ?? "--")}% real</strong>
      </div>
    `
    : "";

  resultStage.innerHTML = `
    <article class="verdict-card">
      ${articleMarkup}
      ${renderSourceProfile(sourceProfile, article)}
      <span class="verdict-badge ${verdictClass}">${escapeHtml(data.label || "Unknown")}</span>
      <div class="confidence-number">${escapeHtml(data.confidence ?? "--")}%</div>
      <p class="supporting-text">${escapeHtml(
        data.input_quality?.recommendation || "Confidence score based on the current article signals."
      )}</p>

      <div class="probability-list">
        <div class="probability-item">
          <span>Real probability: ${escapeHtml(data.probabilities?.real ?? "--")}%</span>
          <div class="bar-track"><div class="bar-fill" style="width:${data.probabilities?.real ?? 0}%"></div></div>
        </div>
        <div class="probability-item">
          <span>Fake probability: ${escapeHtml(data.probabilities?.fake ?? "--")}%</span>
          <div class="bar-track"><div class="bar-fill" style="width:${data.probabilities?.fake ?? 0}%"></div></div>
        </div>
      </div>

      <div class="breakdown-grid">
        <div class="breakdown-card">
          <span>Headline model</span>
          <strong>${escapeHtml(breakdown.headline_model?.real ?? "--")}% real</strong>
        </div>
        <div class="breakdown-card">
          <span>Body model</span>
          <strong>${escapeHtml(breakdown.body_model?.real ?? "--")}% real</strong>
        </div>
        <div class="breakdown-card">
          <span>Fusion model</span>
          <strong>${escapeHtml(breakdown.fusion_model?.real ?? "--")}% real</strong>
        </div>
        <div class="breakdown-card">
          <span>Consensus</span>
          <strong>${escapeHtml(data.analysis?.model_consensus || "n/a")}</strong>
        </div>
        ${secondaryMarkup}
      </div>

      <div class="signals-wrap">
        ${renderSignalGroup("Source cues", sourceCues)}
        ${renderSignalGroup("Reporting cues", reportingCues)}
        ${renderSignalGroup("Metadata cues", metadataCues, "neutral")}
        ${renderSignalGroup("Risk cues", riskCues, "warning")}
      </div>

      ${renderVerificationLinks(data.verification_links || [])}
    </article>
  `;
}

function renderError(message) {
  resultStage.innerHTML = `
    <div class="error-card">
      <strong>Prediction failed.</strong>
      <p>${escapeHtml(message)}</p>
    </div>
  `;
}

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setMode(button.dataset.mode);
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitBtn.disabled = true;
  submitBtn.textContent = activeMode === "url" ? "Analyzing..." : "Checking...";

  const endpoint = activeMode === "url" ? "/api/predict-url" : "/api/predict";
  const payload =
    activeMode === "url"
      ? { url: inputs.url.value.trim() }
      : {
          title: inputs.title.value.trim(),
          author: inputs.author.value.trim(),
          text: inputs.text.value.trim(),
        };

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Unable to analyze this content.");
    }

    renderResult(data);
    loadHistory();
  } catch (error) {
    renderError(error.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = activeMode === "url" ? "Analyze URL" : "Analyze Article";
  }
});

setMode(activeMode);
initializeTheme();
loadMetrics();
loadHistory();
