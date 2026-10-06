/**
 * PromptPilot Background Service Worker
 * Handles backend requests from extension content scripts.
 */

const BACKEND_URL = "http://127.0.0.1:8000";

chrome.runtime.onInstalled.addListener(() => {
  console.log("PromptPilot extension installed successfully.");
});

chrome.runtime.onMessage.addListener(
  (request, sender, sendResponse) => {

    if (request.type === "PING") {
      sendResponse({ status: "PONG" });
      return false;
    }

    if (request.type === "ENHANCE_PROMPT") {
      handleEnhancement(request)
        .then((result) => {
          sendResponse(result);
        })
        .catch((error) => {
          console.error("PromptPilot enhancement error:", error);

          sendResponse({
            success: false,
            message: error.message || "Enhancement failed."
          });
        });

      // Keep the message channel open for the async response.
      return true;
    }

    return false;
  }
);

async function handleEnhancement(request) {
  const prompt = request.prompt;
  const mode = request.mode || "auto";

  if (typeof prompt !== "string" || !prompt.trim()) {
    return {
      success: false,
      message: "Please enter a prompt first."
    };
  }

  console.log("PromptPilot: Sending enhancement request.");

  const response = await fetch(
    `${BACKEND_URL}/api/enhance`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        prompt: prompt,
        mode: mode
      })
    }
  );

  let data;

  try {
    data = await response.json();
  } catch {
    throw new Error(
      `Backend returned an invalid response (${response.status}).`
    );
  }

  if (!response.ok) {
    throw new Error(
      data.detail ||
      data.message ||
      `Backend error: ${response.status}`
    );
  }

  if (!data.success || !data.enhanced_prompt) {
    throw new Error(
      data.message || "The backend did not return an enhanced prompt."
    );
  }

  console.log("PromptPilot: Enhancement completed.", {
    mode: data.mode,
    detectedMode: data.detected_mode,
    autoDetected: data.auto_detected,
    decision: data.decision,
    tokensSaved: data.tokens_saved
  });

  // One-shot mode override: a manually-picked mode applies to exactly
  // this enhancement. Reset to "auto" immediately after a successful
  // call so a future unrelated prompt doesn't silently inherit a
  // stale manual choice. Only resets on success — a failed request
  // (network error, backend down) leaves the user's pick intact so
  // they don't lose their choice to a transient failure.
  if (chrome.storage && chrome.storage.sync && mode !== "auto") {
    chrome.storage.sync.set({ enhancementMode: "auto" });
  }

  return {
    success: true,
    data: data
  };
}