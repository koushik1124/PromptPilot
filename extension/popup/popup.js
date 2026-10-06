document.addEventListener('DOMContentLoaded', async () => {
  const statusElement = document.getElementById('backend-status');
  const modeSelect = document.getElementById('enhancement-mode');

  // Load saved mode. After a successful enhancement, the background
  // worker resets this back to "auto" — so this will normally show
  // "auto" unless the user picked a mode and hasn't enhanced anything
  // since (one-shot manual override).
  if (chrome.storage && chrome.storage.sync) {
    const saved = await chrome.storage.sync.get(['enhancementMode']);
    modeSelect.value = saved.enhancementMode || 'auto';
  }

  // Handle mode change
  modeSelect.addEventListener('change', (e) => {
    if (chrome.storage && chrome.storage.sync) {
      chrome.storage.sync.set({ enhancementMode: e.target.value });
    }
  });

  // Check backend health
  try {
    const response = await fetch('http://localhost:8000/health', { method: 'GET' });
    if (response.ok) {
      statusElement.textContent = 'Online';
      statusElement.className = 'status-pill online';
    } else {
      throw new Error('Backend error');
    }
  } catch (err) {
    statusElement.textContent = 'Offline';
    statusElement.className = 'status-pill offline';
  }
});