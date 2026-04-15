const backend = window.location.origin;

const dom = {
  modeSelect: document.getElementById('mode-select'),
  fileSection: document.getElementById('file-section'),
  pasteSection: document.getElementById('paste-section'),
  indexFilesBtn: document.getElementById('index-files-btn'),
  indexPasteBtn: document.getElementById('index-paste-btn'),
  clearHistoryBtn: document.getElementById('clear-history-btn'),
  fileUploader: document.getElementById('file-uploader'),
  pasteContent: document.getElementById('paste-content'),
  pasteName: document.getElementById('paste-name'),
  promptForm: document.getElementById('prompt-form'),
  promptText: document.getElementById('prompt-text'),
  messageList: document.getElementById('message-list'),
  backendStatus: document.getElementById('backend-status'),
  sessionId: document.getElementById('session-id'),
  processingStatus: document.getElementById('processing-status'),
  sendButton: document.querySelector('#prompt-form button[type=submit]'),
};

let isProcessing = false;
let messageCount = 0;

/* ── Session ─────────────────────────────────────────── */
let sessionId = localStorage.getItem('rag_session_id') || '';
function setActiveSession(id) {
  if (!id) return;
  sessionId = id;
  localStorage.setItem('rag_session_id', sessionId);
  dom.sessionId.textContent = `Session: ${sessionId.slice(0, 8)}…`;
}
if (!sessionId) {
  setActiveSession(crypto.randomUUID ? crypto.randomUUID() : `sess_${Date.now()}`);
} else {
  dom.sessionId.textContent = `Session: ${sessionId.slice(0, 8)}…`;
}

/* ── Theme toggle ─────────────────────────────────────── */
(function initTheme() {
  const saved = localStorage.getItem('rag_theme') || 'dark';
  applyTheme(saved);
})();

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('rag_theme', theme);
  const lightPill = document.getElementById('light-pill');
  const darkPill  = document.getElementById('dark-pill');
  if (theme === 'light') {
    lightPill && lightPill.classList.add('active');
    darkPill  && darkPill.classList.remove('active');
  } else {
    darkPill  && darkPill.classList.add('active');
    lightPill && lightPill.classList.remove('active');
  }
}

document.getElementById('theme-toggle').addEventListener('click', () => {
  const current = document.documentElement.getAttribute('data-theme');
  applyTheme(current === 'dark' ? 'light' : 'dark');
});

/* ── Message counter ──────────────────────────────────── */
function updateMsgCount(delta) {
  messageCount += delta;
  const el = document.getElementById('msg-count');
  if (el) el.textContent = `${messageCount} message${messageCount !== 1 ? 's' : ''}`;
}

/* ── Mode ─────────────────────────────────────────────── */
function setMode() {
  const mode = dom.modeSelect.value;
  dom.fileSection.classList.toggle('hidden', mode !== 'files');
  dom.pasteSection.classList.toggle('hidden', mode !== 'pasted');

  const modeInfo    = document.getElementById('mode-info');
  const headerLabel = document.getElementById('header-mode-label');

  const labels = {
    normal: { info: 'Normal Chat: no retrieval, direct LLM response (quick answers, may hallucinate).', badge: 'Normal Chat' },
    files:  { info: 'Ask From Files: Uses indexed documents for grounded answers (best for precision).', badge: 'Ask From Files' },
    pasted: { info: 'Ask From Pasted Text: Uses pasted content as knowledge source for that session.', badge: 'Ask From Pasted Text' },
  };
  if (modeInfo)    modeInfo.textContent    = labels[mode].info;
  if (headerLabel) headerLabel.textContent = labels[mode].badge;
}

/* ── Backend health ───────────────────────────────────── */
async function updateBackendHealth() {
  const dot = document.getElementById('status-dot');
  try {
    const res  = await fetch(`${backend}/health`);
    const data = await parseJsonResponse(res);
    if (data.status === 'ok') {
      dom.backendStatus.textContent = `online · ${data.active_sessions} sessions`;
      dot && dot.classList.add('online') && dot.classList.remove('error');
      dot && (dot.classList.remove('error'), dot.classList.add('online'));
    } else {
      dom.backendStatus.textContent = 'unhealthy';
      dot && (dot.classList.remove('online'), dot.classList.add('error'));
    }
  } catch {
    dom.backendStatus.textContent = 'unavailable';
    dot && (dot.classList.remove('online'), dot.classList.add('error'));
  }
}

/* ── Empty state ──────────────────────────────────────── */
function hideEmptyState() {
  const es = document.getElementById('empty-state');
  if (es) es.remove();
}

function showEmptyState() {
  if (document.getElementById('empty-state')) return;
  const es = document.createElement('div');
  es.className = 'empty-state';
  es.id = 'empty-state';
  es.innerHTML = `
    <div class="empty-state-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
      </svg>
    </div>
    <p>Choose a mode in the sidebar, then ask any question to get started.</p>
  `;
  dom.messageList.appendChild(es);
}

/* ── Add message ──────────────────────────────────────── */
function addMessage(text, sender = 'assistant', options = {}) {
  if (!text) return null;
  hideEmptyState();
  const message = document.createElement('div');
  message.className = `message ${sender}`;
  if (options.placeholder) message.classList.add('placeholder');
  message.textContent = text;
  dom.messageList.appendChild(message);
  dom.messageList.scrollTop = dom.messageList.scrollHeight;
  if (!options.placeholder) updateMsgCount(1);
  return message;
}

/* ── Processing state ─────────────────────────────────── */
function setProcessing(active) {
  isProcessing = active;
  dom.promptText.disabled = active;
  if (dom.sendButton) dom.sendButton.disabled = active;
  dom.processingStatus.textContent = active ? 'Assistant is thinking…' : '';
}

/* ── Source block ─────────────────────────────────────── */
function addSourceBlock(sources) {
  hideEmptyState();
  const block = document.createElement('div');
  block.className = 'message assistant source-block';
  block.innerHTML = `
    <div class="source-header" onclick="toggleSource(this)">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="transform:rotate(0deg);transition:transform 0.2s">
        <polyline points="9 18 15 12 9 6"/>
      </svg>
      Sources (${sources.length}) — click to expand
    </div>
    <div class="source-list hidden">
      ${sources.map((s, i) => {
        const title = s.metadata?.source || s.metadata?.file_name || s.metadata?.source_name || 'unknown';
        const snippet = (s.content || s.page_content || '').slice(0, 200);
        return `
        <div class="source-item">
          <div class="source-title">${i + 1}. ${title}</div>
          <div class="source-snippet">${snippet}…</div>
        </div>
      `;
      }).join('')}
    </div>
  `;
  dom.messageList.appendChild(block);
  dom.messageList.scrollTop = dom.messageList.scrollHeight;
}

window.toggleSource = function(header) {
  const list = header.nextElementSibling;
  const icon = header.querySelector('svg');
  list.classList.toggle('hidden');
  if (icon) icon.style.transform = list.classList.contains('hidden') ? 'rotate(0deg)' : 'rotate(90deg)';
};

/* ── JSON parser ──────────────────────────────────────── */
async function parseJsonResponse(response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch (err) {
    console.error('Not JSON response:', text);
    throw new Error('Server returned invalid JSON.');
  }
}

/* ── Sanitize ─────────────────────────────────────────── */
function sanitize(text) {
  if (!text) return '';
  return text
    .split('\n')
    .filter((line) => !line.trim().startsWith('Role:'))
    .join('\n')
    .trim();
}

/* ── Ask question ─────────────────────────────────────── */
async function askQuestion(question) {
  if (isProcessing) {
    addMessage('Please wait until the assistant finishes thinking.', 'assistant');
    return;
  }

  addMessage(question, 'user');
  setProcessing(true);
  const placeholder = addMessage('Thinking…', 'assistant', { placeholder: true });

  try {
    const endpoint = dom.modeSelect.value === 'normal' ? 'chat' : 'ask';

    if (endpoint === 'ask' && !sessionId) {
      if (placeholder) placeholder.textContent = 'Error: please index files or pasted text first before asking in retrieval mode.';
      setProcessing(false);
      return;
    }

    const response = await fetch(`${backend}/${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: sessionId }),
    });

    const text = await response.text();
    let payload;
    try {
      payload = JSON.parse(text);
    } catch {
      if (placeholder) placeholder.textContent = 'Error: Server returned invalid JSON.';
      setProcessing(false);
      return { error: 'Server returned invalid JSON.' };
    }

    if (!response.ok) {
      const errorText = payload.error || payload.message || 'Request failed';
      if (placeholder) placeholder.textContent = `Error: ${errorText}`;
      setProcessing(false);
      return payload;
    }

    const answer = sanitize(payload.answer || payload.result || 'No answer');
    if (placeholder) {
      placeholder.textContent = answer;
      placeholder.classList.remove('placeholder');
      updateMsgCount(1);
    }

    const sources = payload.sources || payload.source_documents || [];
    if (Array.isArray(sources) && sources.length > 0) addSourceBlock(sources);

    setProcessing(false);
    return payload;
  } catch (error) {
    if (placeholder) placeholder.textContent = `Error: ${error.message}`;
    setProcessing(false);
    return { error: error.message };
  }
}

/* ── Index files ──────────────────────────────────────── */
async function indexFiles() {
  const files = dom.fileUploader.files;
  if (!files || files.length === 0) return addMessage('Error: choose files to index.', 'assistant');

  const form = new FormData();
  form.append('session_id', sessionId);
  for (const file of files) form.append('files', file);

  dom.indexFilesBtn.disabled = true;
  dom.indexFilesBtn.textContent = 'Indexing…';

  try {
    const res     = await fetch(`${backend}/upload`, { method: 'POST', body: form });
    const payload = await parseJsonResponse(res);
    if (!res.ok) throw new Error(payload.error || 'Upload failed');

    if (payload.session_id) setActiveSession(payload.session_id);
    const count = Number.isInteger(payload.files_indexed_count)
      ? payload.files_indexed_count
      : Array.isArray(payload.files_indexed)
      ? payload.files_indexed.length
      : 0;
    addMessage(`✓ Indexed ${count} file(s) successfully.`, 'assistant');
  } catch (err) {
    addMessage(`Index error: ${err.message}`, 'assistant');
  } finally {
    dom.indexFilesBtn.disabled = false;
    dom.indexFilesBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="width:13px;height:13px">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
      </svg>
      Index files`;
  }
}

/* ── Index pasted text ────────────────────────────────── */
async function indexPastedText() {
  const text = dom.pasteContent.value.trim();
  const name = dom.pasteName.value.trim() || 'notes.txt';
  if (!text) return addMessage('Error: paste some text first.', 'assistant');

  const form = new FormData();
  form.append('session_id', sessionId);
  form.append('text_content', text);
  form.append('text_name', name);

  dom.indexPasteBtn.disabled = true;
  dom.indexPasteBtn.textContent = 'Indexing…';

  try {
    const res     = await fetch(`${backend}/upload`, { method: 'POST', body: form });
    const payload = await parseJsonResponse(res);
    if (!res.ok) throw new Error(payload.error || 'Upload failed');

    if (payload.session_id) setActiveSession(payload.session_id);
    addMessage(`✓ Indexed pasted text as "${name}".`, 'assistant');
  } catch (err) {
    addMessage(`Index error: ${err.message}`, 'assistant');
  } finally {
    dom.indexPasteBtn.disabled = false;
    dom.indexPasteBtn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" style="width:13px;height:13px">
        <polyline points="20 12 20 22 4 22 4 12"/><rect x="2" y="7" width="20" height="5"/>
        <path d="M12 22V7"/><path d="M12 7H7.5a2.5 2.5 0 010-5C11 2 12 7 12 7z"/><path d="M12 7h4.5a2.5 2.5 0 000-5C13 2 12 7 12 7z"/>
      </svg>
      Index pasted content`;
  }
}

/* ── Clear history ────────────────────────────────────── */
function clearHistory() {
  dom.messageList.innerHTML = '';
  messageCount = 0;
  updateMsgCount(0);
  showEmptyState();
}

/* ── Init ─────────────────────────────────────────────── */
function init() {
  setMode();
  dom.modeSelect.addEventListener('change', setMode);

  dom.promptForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = dom.promptText.value.trim();
    if (!text) return;
    dom.promptText.value = '';
    await askQuestion(text);
  });

  dom.promptText.addEventListener('keydown', async (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      const text = dom.promptText.value.trim();
      if (!text) return;
      dom.promptText.value = '';
      await askQuestion(text);
    }
  });

  dom.indexFilesBtn.addEventListener('click', indexFiles);
  dom.indexPasteBtn.addEventListener('click', indexPastedText);
  dom.clearHistoryBtn.addEventListener('click', clearHistory);

  updateBackendHealth();
  setInterval(updateBackendHealth, 15000);
}

init();