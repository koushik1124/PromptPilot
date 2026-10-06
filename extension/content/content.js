// Content script entry point for PromptPilot

(() => {
  console.log('PromptPilot content script loaded');

  const adapters = [
    new window.PromptPilotChatGPTAdapter()
  ];

  const activeAdapter = adapters.find((adapter) => adapter.matches());

  if (activeAdapter) {
    activeAdapter.injectUI();
  }
})();
