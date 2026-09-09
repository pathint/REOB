(function () {
  var DEFAULT_THEME = "catppuccin-latte";
  var ALLOWED_THEMES = [
    "documenter-light",
    "documenter-dark",
    "catppuccin-latte",
    "catppuccin-frappe",
    "catppuccin-macchiato",
    "catppuccin-mocha",
  ];

  function isAllowedTheme(theme) {
    return ALLOWED_THEMES.indexOf(theme) !== -1;
  }

  function storageGet(key) {
    try {
      return window.localStorage && window.localStorage.getItem(key);
    } catch (_) {
      return null;
    }
  }

  function storageSet(key, value) {
    try {
      if (window.localStorage) window.localStorage.setItem(key, value);
    } catch (_) {
      return;
    }
  }

  function applyDefaultTheme() {
    var theme = storageGet("documenter-theme");

    if (!theme || !isAllowedTheme(theme)) {
      storageSet("documenter-theme", DEFAULT_THEME);
    }

    if (typeof window.set_theme_from_local_storage === "function") {
      window.set_theme_from_local_storage();
    }
  }

  function syncThemePicker() {
    var picker = document.getElementById("documenter-themepicker");
    if (!picker) return;

    var theme = storageGet("documenter-theme") || DEFAULT_THEME;
    if (!isAllowedTheme(theme)) theme = DEFAULT_THEME;

    picker.value = theme;

    // Documenter 的主题切换依赖 require.js + jQuery。若 CDN 无法加载，
    // 这里用原生事件兜底，保证下拉框仍能切换主题。
    if (typeof window.require !== "undefined") return;

    picker.addEventListener("change", function () {
      if (picker.value === "auto") {
        try {
          window.localStorage && window.localStorage.removeItem("documenter-theme");
        } catch (_) {
          return;
        }
      } else {
        storageSet("documenter-theme", picker.value);
      }

      if (typeof window.set_theme_from_local_storage === "function") {
        window.set_theme_from_local_storage();
      }
    });
  }

  function setupSettingsFallback() {
    if (typeof window.require !== "undefined") return;

    var button = document.getElementById("documenter-settings-button");
    var settings = document.getElementById("documenter-settings");
    if (!button || !settings) return;

    button.addEventListener("click", function (event) {
      event.preventDefault();
      settings.classList.toggle("is-active");
    });

    var close = settings.querySelector("button.delete");
    if (close) {
      close.addEventListener("click", function () {
        settings.classList.remove("is-active");
      });
    }

    document.addEventListener("keyup", function (event) {
      if (event.key === "Escape") settings.classList.remove("is-active");
    });
  }

  applyDefaultTheme();

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      setupSettingsFallback();
      syncThemePicker();
      applyDefaultTheme();
    });
  } else {
    setupSettingsFallback();
    syncThemePicker();
    applyDefaultTheme();
  }
})();
