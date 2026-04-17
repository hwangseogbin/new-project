const adminThemeToggle = document.getElementById("theme-toggle");
const adminMetricEls = {
  test: document.getElementById("admin-test-accuracy"),
  train: document.getElementById("admin-train-accuracy"),
  rows: document.getElementById("admin-dataset-rows"),
  strategy: document.getElementById("admin-fusion-strategy"),
  refreshNote: document.getElementById("admin-refresh-note"),
};
const adminHistoryBody = document.getElementById("admin-history-body");

if (window.VerifiJinTheme) {
  window.VerifiJinTheme.initializeThemeToggle(adminThemeToggle);
}

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

function formatPercent(value) {
  return typeof value === "number" ? `${value}%` : "--";
}

function formatCount(value) {
  return typeof value === "number" ? value.toLocaleString() : "--";
}

function updateRefreshNote(label) {
  if (!adminMetricEls.refreshNote) {
    return;
  }
  adminMetricEls.refreshNote.textContent = label;
}

async function loadAdminMetrics() {
  const response = await fetch("/api/metrics");
  if (!response.ok) {
    throw new Error("Unable to load model metrics.");
  }

  const metrics = await response.json();
  if (adminMetricEls.test) {
    adminMetricEls.test.textContent = formatPercent(metrics.test_accuracy);
  }
  if (adminMetricEls.train) {
    adminMetricEls.train.textContent = formatPercent(metrics.train_accuracy);
  }
  if (adminMetricEls.rows) {
    adminMetricEls.rows.textContent = formatCount(metrics.dataset_rows);
  }
  if (adminMetricEls.strategy) {
    adminMetricEls.strategy.textContent = metrics.fusion_strategy || "Hybrid scoring pipeline";
  }
}

function renderAdminHistoryRows(items) {
  if (!adminHistoryBody) {
    return;
  }

  if (!items.length) {
    adminHistoryBody.innerHTML = `
      <tr>
        <td colspan="6">No predictions saved yet.</td>
      </tr>
    `;
    return;
  }

  adminHistoryBody.innerHTML = items
    .map((item) => `
      <tr>
        <td>${escapeHtml(item.created_at || "Unknown time")}</td>
        <td>${escapeHtml(item.title || "Untitled")}</td>
        <td>${escapeHtml(item.source || item.author || "Unknown")}</td>
        <td>${escapeHtml(item.input_mode || "manual")}</td>
        <td>${escapeHtml(item.label || "Unknown")}</td>
        <td>${escapeHtml(item.confidence ?? "--")}%</td>
      </tr>
    `)
    .join("");
}

async function loadAdminHistory() {
  const response = await fetch("/api/history?limit=20");
  if (!response.ok) {
    throw new Error("Unable to load prediction history.");
  }

  const data = await response.json();
  renderAdminHistoryRows(data.items || []);
}

async function refreshAdminDashboard() {
  updateRefreshNote("Refreshing...");
  try {
    await Promise.all([loadAdminMetrics(), loadAdminHistory()]);
    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    updateRefreshNote(`Auto-refresh on - ${timestamp}`);
  } catch (error) {
    updateRefreshNote("Auto-refresh unavailable");
  }
}

refreshAdminDashboard();
window.setInterval(refreshAdminDashboard, 10000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    refreshAdminDashboard();
  }
});
