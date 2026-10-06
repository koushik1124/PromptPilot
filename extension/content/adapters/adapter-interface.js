/**
 * Base interface for web LLM chat UI adapters
 */
class BaseAdapter {
  /**
   * Check if the current page matches this adapter
   * @returns {boolean}
   */
  matches() {
    throw new Error('matches() must be implemented');
  }

  /**
   * Get the active prompt input element
   * @returns {HTMLElement|null}
   */
  getInputElement() {
    throw new Error('getInputElement() must be implemented');
  }

  /**
   * Get current prompt text from the input
   * @returns {string}
   */
  getPromptText() {
    throw new Error('getPromptText() must be implemented');
  }

  /**
   * Set prompt text into the active input
   * @param {string} text
   */
  setPromptText(text) {
    throw new Error('setPromptText() must be implemented');
  }

  /**
   * Inject UI elements into the host page
   */
  injectUI() {
    throw new Error('injectUI() must be implemented');
  }
}

window.PromptPilotBaseAdapter = BaseAdapter;
