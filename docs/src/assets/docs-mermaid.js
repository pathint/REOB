(function () {
  var STYLE_ID = "reob-mermaid-style";
  var rendered = false;

  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;

    var style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = [
      ".mermaid {",
      "  display: block;",
      "  margin: 1rem 0 1.5rem;",
      "  overflow-x: auto;",
      "  background: transparent;",
      "  padding: 0.25rem 0;",
      "}",
      ".mermaid svg {",
      "  max-width: 100%;",
      "  height: auto;",
      "  overflow: visible;",
      "  background: transparent;",
      "}",
    ].join("\n");

    document.head.appendChild(style);
  }

  function prepareBlocks() {
    var blocks = document.querySelectorAll("pre > code.language-mermaid");

    blocks.forEach(function (code) {
      var pre = code.parentNode;
      if (!pre || pre.dataset.mermaidProcessed === "1") return;

      var wrapper = document.createElement("div");
      wrapper.className = "mermaid";
      wrapper.textContent = code.textContent.trim();
      pre.dataset.mermaidProcessed = "1";
      pre.replaceWith(wrapper);
    });
  }

  function initializeMermaid() {
    if (!window.mermaid || typeof window.mermaid.initialize !== "function") {
      return false;
    }

    window.mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      flowchart: { htmlLabels: true },
    });
    return true;
  }

  function renderMermaid() {
    if (rendered) return;
    if (!initializeMermaid()) return;

    injectStyle();
    prepareBlocks();

    if (typeof window.mermaid.run === "function") {
      window.mermaid.run({
        querySelector: ".mermaid",
        suppressErrors: true,
      });
    } else if (typeof window.mermaid.init === "function") {
      window.mermaid.init(undefined, document.querySelectorAll(".mermaid"));
    }

    rendered = true;
  }

  function boot() {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", renderMermaid);
    } else {
      renderMermaid();
    }

    window.addEventListener("load", function () {
      renderMermaid();
    });
  }

  boot();
})();
