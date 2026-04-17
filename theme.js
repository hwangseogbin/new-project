const VERIFIJIN_THEME_STORAGE_KEY = "verifijin-theme";

function readStoredTheme() {
  try {
    return window.localStorage.getItem(VERIFIJIN_THEME_STORAGE_KEY);
  } catch (error) {
    return null;
  }
}

function saveTheme(theme) {
  try {
    window.localStorage.setItem(VERIFIJIN_THEME_STORAGE_KEY, theme);
  } catch (error) {
    // Ignore storage failures and keep the active theme only in memory.
  }
}

function applyTheme(theme) {
  const normalizedTheme = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = normalizedTheme;
  return normalizedTheme;
}

function syncThemeToggleState(toggle) {
  if (!toggle) {
    return;
  }

  const activeTheme = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
  const nextThemeLabel = activeTheme === "dark" ? "Light Theme" : "Dark Theme";
  toggle.textContent = nextThemeLabel;
  toggle.setAttribute("aria-pressed", String(activeTheme === "dark"));
  toggle.setAttribute("aria-label", `Switch to ${nextThemeLabel.toLowerCase()}`);
}

function initializeThemeToggle(toggleOrId = "theme-toggle") {
  const toggle = typeof toggleOrId === "string" ? document.getElementById(toggleOrId) : toggleOrId;
  applyTheme(readStoredTheme() || document.documentElement.dataset.theme || "light");
  syncThemeToggleState(toggle);

  if (!toggle || toggle.dataset.themeBound === "true") {
    return;
  }

  toggle.dataset.themeBound = "true";
  toggle.addEventListener("click", () => {
    const nextTheme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(nextTheme);
    saveTheme(nextTheme);
    syncThemeToggleState(toggle);
  });
}

window.VerifiJinTheme = {
  applyTheme,
  initializeThemeToggle,
  readStoredTheme,
  saveTheme,
};
