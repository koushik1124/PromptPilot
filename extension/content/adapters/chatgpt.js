/**
 * ChatGPT adapter for PromptPilot.
 * Adds prompt enhancement controls to the ChatGPT composer.
 */

class ChatGPTAdapter extends window.PromptPilotBaseAdapter {

  matches() {
    return (
      window.location.hostname === "chatgpt.com" ||
      window.location.hostname === "chat.openai.com"
    );
  }

  getInputElement() {
    // Querying for the FIRST match (document.querySelector) is unsafe
    // here: if the page has more than one element matching these broad
    // selectors (a stray contenteditable, a hidden edit-in-place box
    // for a previous message, etc.), the wrong one can be returned
    // silently. Collect all matches, filter to visible ones, and
    // prefer the element sitting lowest on the page — the real chat
    // composer reliably sits at the bottom of the UI.
    const candidates = [
      ...document.querySelectorAll('#prompt-textarea'),
      ...document.querySelectorAll('div[contenteditable="true"][data-placeholder]'),
      ...document.querySelectorAll('div[contenteditable="true"]'),
      ...document.querySelectorAll('textarea'),
    ];

    const visible = candidates.filter((el) => {
      if (!el || !el.isConnected) return false;
      if (el.offsetParent === null && getComputedStyle(el).position !== 'fixed') return false;
      const rect = el.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0;
    });

    if (visible.length === 0) {
      return null;
    }

    const byId = visible.find((el) => el.id === 'prompt-textarea');
    if (byId) {
      return byId;
    }

    visible.sort((a, b) => b.getBoundingClientRect().bottom - a.getBoundingClientRect().bottom);
    return visible[0];
  }

  getPromptText() {
    const input = this.getInputElement();

    if (!input) {
      return "";
    }

    if (
      input.tagName === "TEXTAREA" ||
      input.tagName === "INPUT"
    ) {
      return input.value || "";
    }

    return input.innerText || input.textContent || "";
  }

  setPromptText(text) {
    const input = this.getInputElement();

    if (!input) {
      throw new Error("ChatGPT composer not found.");
    }

    input.focus();

    if (
      input.tagName === "TEXTAREA" ||
      input.tagName === "INPUT"
    ) {
      const prototype =
        input.tagName === "TEXTAREA"
          ? HTMLTextAreaElement.prototype
          : HTMLInputElement.prototype;

      const setter = Object.getOwnPropertyDescriptor(
        prototype,
        "value"
      )?.set;

      if (setter) {
        setter.call(input, text);
      } else {
        input.value = text;
      }
    } else {
      // Replace the content of the editable composer.
      input.replaceChildren(
        document.createTextNode(text)
      );
    }

    // Notify the host application's UI framework.
    input.dispatchEvent(
      new InputEvent("input", {
        bubbles: true,
        inputType: "insertText",
        data: text
      })
    );

    input.dispatchEvent(
      new Event("change", { bubbles: true })
    );
  }

  injectUI() {
    if (document.getElementById("promptpilot-controls")) {
      return;
    }

    this.createControls();

    // ChatGPT dynamically recreates its composer.
    // Keep trying to attach our controls when it changes.
    const observer = new MutationObserver(() => {
      this.attachControls();
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });

    this.observer = observer;

    this.attachControls();
  }

  createControls() {
    const container = document.createElement("div");

    container.id = "promptpilot-controls";

    container.style.cssText = `
      display: flex;
      flex-direction: column;
      gap: 8px;
      margin: 8px 0;
      width: 100%;
      font-family: inherit;
      color: var(--text-primary, inherit);
    `;

    const buttonRow = document.createElement("div");

    buttonRow.style.cssText = `
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    `;

    const enhanceButton = document.createElement("button");

    enhanceButton.id = "promptpilot-enhance-button";
    enhanceButton.type = "button";
    enhanceButton.textContent = "✦ Enhance with PromptPilot";

    enhanceButton.style.cssText = `
      border: 1px solid #635bff;
      background: #635bff;
      color: white;
      padding: 8px 14px;
      border-radius: 10px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    `;

    const status = document.createElement("span");

    status.id = "promptpilot-inline-status";

    status.style.cssText = `
      font-size: 12px;
      color: var(--text-secondary, #888);
    `;

    buttonRow.appendChild(enhanceButton);
    buttonRow.appendChild(status);

    const preview = document.createElement("div");

    preview.id = "promptpilot-preview";

    preview.style.cssText = `
      display: none;
      padding: 12px;
      border: 1px solid #635bff;
      border-radius: 12px;
      background: var(--background-secondary, #1e1e1e);
      color: var(--text-primary, #fff);
      font-size: 13px;
    `;

    container.appendChild(buttonRow);
    container.appendChild(preview);

    enhanceButton.addEventListener("click", () => {
      this.enhancePrompt();
    });

    this.container = container;
    this.statusElement = status;
    this.previewElement = preview;
    this.enhanceButton = enhanceButton;
  }

  attachControls() {
    if (!this.container) {
      return;
    }

    const input = this.getInputElement();

    if (!input || !input.isConnected) {
      return;
    }

    // Do not add the controls multiple times.
    if (this.container.isConnected) {
      return;
    }

    const composer =
      input.closest("form") ||
      input.parentElement?.parentElement;

    if (composer && composer.parentElement) {
      composer.parentElement.insertBefore(
        this.container,
        composer.nextSibling
      );
    }
  }

  async enhancePrompt() {
    const originalPrompt = this.getPromptText().trim();

    if (!originalPrompt) {
      this.setStatus("Enter a prompt first.");
      return;
    }

    this.enhanceButton.disabled = true;
    this.enhanceButton.textContent = "Enhancing...";
    this.setStatus("Connecting to PromptPilot...");
    this.previewElement.style.display = "none";

    try {
      // Retrieve the user's saved mode. Defaults to "auto" — a manual
      // pick is a one-shot override that the background worker resets
      // after use, so this correctly falls back to auto-detect on
      // normal runs.
      let mode = "auto";

      if (chrome.storage?.sync) {
        const settings = await chrome.storage.sync.get([
          "enhancementMode"
        ]);

        mode = settings.enhancementMode || "auto";
      }

      const response = await chrome.runtime.sendMessage({
        type: "ENHANCE_PROMPT",
        prompt: originalPrompt,
        mode: mode
      });

      if (!response || !response.success) {
        throw new Error(
          response?.message || "Enhancement request failed."
        );
      }

      const result = response.data;

      console.log("PromptPilot decision:", {
        decision: result.decision,
        reason: result.reason,
        requestedMode: result.requested_mode,
        detectedMode: result.detected_mode,
      });

      this.showPreview(originalPrompt, result);

      const changeText = result.token_change_display ||
        this.formatTokenChange(result.original_tokens, result.enhanced_tokens);

      const modeNote = result.auto_detected
        ? ` (auto-detected: ${result.detected_mode})`
        : "";

      this.setStatus(`${changeText}${modeNote}.`);

    } catch (error) {
      console.error("PromptPilot:", error);

      this.setStatus(
        error.message || "Could not enhance prompt."
      );

    } finally {
      this.enhanceButton.disabled = false;
      this.enhanceButton.textContent =
        "✦ Enhance with PromptPilot";
    }
  }

  formatTokenChange(originalTokens, enhancedTokens) {
    const delta = enhancedTokens - originalTokens;
    if (originalTokens === 0) {
      return enhancedTokens === 0
        ? "No change (0%)"
        : `${enhancedTokens} more tokens (+100.0%)`;
    }
    const pct = ((delta / originalTokens) * 100).toFixed(1);
    if (delta < 0) {
      // Use the magnitude here — pct is negative in this branch, and
      // pairing "fewer tokens" with a raw negative percentage reads
      // as a confusing double-negative ("15 fewer tokens (-37.5%)").
      return `${Math.abs(delta)} fewer tokens (${Math.abs(pct)}%)`;
    } else if (delta === 0) {
      return "No change (0%)";
    } else {
      return `${delta} more tokens (+${pct}%)`;
    }
  }

  showPreview(originalPrompt, result) {
    const preview = this.previewElement;

    preview.replaceChildren();

    const heading = document.createElement("strong");
    heading.textContent = "Enhanced prompt";

    const enhancedText = document.createElement("div");

    enhancedText.style.cssText = `
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      margin: 10px 0;
      line-height: 1.5;
    `;

    enhancedText.textContent = result.enhanced_prompt;

    const changeText = result.token_change_display ||
      this.formatTokenChange(result.original_tokens, result.enhanced_tokens);

    const metrics = document.createElement("div");

    metrics.style.cssText = `
      font-size: 12px;
      opacity: 0.8;
      margin-bottom: 10px;
    `;

    metrics.textContent =
      `${result.original_tokens} → ${result.enhanced_tokens} ` +
      `prompt tokens | ${changeText}`;

    const applyButton = document.createElement("button");

    applyButton.textContent = "Apply to ChatGPT";
    applyButton.type = "button";

    applyButton.style.cssText = `
      background: #635bff;
      color: white;
      border: none;
      padding: 8px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      margin-right: 8px;
    `;

    applyButton.addEventListener("click", () => {
      // Only replace the draft. Never submit it automatically.
      this.setPromptText(result.enhanced_prompt);

      this.setStatus("Enhanced prompt applied to composer.");

      preview.style.display = "none";
    });

    const discardButton = document.createElement("button");

    discardButton.textContent = "Discard";
    discardButton.type = "button";

    discardButton.style.cssText = `
      background: transparent;
      color: inherit;
      border: 1px solid #777;
      padding: 8px 12px;
      border-radius: 8px;
      cursor: pointer;
    `;

    discardButton.addEventListener("click", () => {
      preview.style.display = "none";
      this.setStatus("Enhancement discarded.");
    });

    preview.appendChild(heading);
    preview.appendChild(enhancedText);
    preview.appendChild(metrics);
    preview.appendChild(applyButton);
    preview.appendChild(discardButton);

    preview.style.display = "block";
  }

  setStatus(message) {
    if (this.statusElement) {
      this.statusElement.textContent = message;
    }
  }
}

window.PromptPilotChatGPTAdapter = ChatGPTAdapter;