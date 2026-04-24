const backend = "https://emphasize-untaxed-statutory.ngrok-free.dev";

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
  chatHistoryList: document.getElementById('chat-history-list'),
  downloadChatBtn: document.getElementById('download-chat-btn'),
};

window.__ragMessageRenderingManagedInApp = true;
window.__ragHistoryManagedInApp = true;

let isProcessing = false;
let messageCount = 0;
const CHAT_HISTORY_KEY = 'rag_all_chats';
let chatStore = readChatStore();

const USER_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>`;
const AI_ICON = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6z"/></svg>`;

/* ── Session ─────────────────────────────────────────── */
let sessionId = localStorage.getItem('rag_session_id') || '';
function setActiveSession(id) {
  if (!id) return;
  sessionId = id;
  localStorage.setItem('rag_session_id', sessionId);
  dom.sessionId.textContent = `Session: ${sessionId.slice(0, 8)}...`;
  migrateLegacyHistoryForSession();
  renderHistory();
}
if (!sessionId) {
  setActiveSession(crypto.randomUUID ? crypto.randomUUID() : `sess_${Date.now()}`);
} else {
  dom.sessionId.textContent = `Session: ${sessionId.slice(0, 8)}...`;
}

/* Chat history */
function readChatStore() {
  try {
    const parsed = JSON.parse(localStorage.getItem(CHAT_HISTORY_KEY) || '{}');
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {};

    const { store, changed } = repairChatStore(parsed);
    if (changed) {
      localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(store));
    }
    return store;
  } catch {
    return {};
  }
}

function saveChatStore() {
  const repaired = repairChatStore(chatStore);
  chatStore = repaired.store;
  localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(chatStore));
  renderHistory();
}

function normalizeHistoryEntry(message) {
  if (!message || typeof message !== 'object') return null;

  const text = String(message.text || '').trim();
  if (!text) return null;

  const role = message.role === 'user' ? 'user' : 'assistant';
  const entry = {
    ...message,
    role,
    text,
  };

  if (Array.isArray(entry.sources)) {
    entry.sources = normalizeSources(entry.sources);
  }

  return entry;
}

function historyKey(message) {
  return `${message.role}\u001f${message.text.trim().toLowerCase()}`;
}

function sameHistoryBlock(messages, leftStart, rightStart, length) {
  for (let offset = 0; offset < length; offset += 1) {
    if (historyKey(messages[leftStart + offset]) !== historyKey(messages[rightStart + offset])) {
      return false;
    }
  }
  return true;
}

function compactRepeatedHistory(messages) {
  const compacted = [];
  let index = 0;

  while (index < messages.length) {
    let bestBlockSize = 0;
    let bestRepeats = 1;
    const maxBlockSize = Math.min(30, Math.floor((messages.length - index) / 3));

    for (let blockSize = 1; blockSize <= maxBlockSize; blockSize += 1) {
      let repeats = 1;
      while (
        index + ((repeats + 1) * blockSize) <= messages.length &&
        sameHistoryBlock(messages, index, index + (repeats * blockSize), blockSize)
      ) {
        repeats += 1;
      }

      if (repeats >= 3 && repeats * blockSize > bestRepeats * bestBlockSize) {
        bestBlockSize = blockSize;
        bestRepeats = repeats;
      }
    }

    if (bestBlockSize) {
      const repeatedBlock = messages.slice(index, index + bestBlockSize);
      const isCorruptedUserOnlyBlock = bestRepeats >= 5 && repeatedBlock.every((message) => message.role === 'user');

      if (!isCorruptedUserOnlyBlock) {
        compacted.push(...repeatedBlock);
      }
      index += bestBlockSize * bestRepeats;
    } else {
      compacted.push(messages[index]);
      index += 1;
    }
  }

  return compacted;
}

function repairChatStore(store) {
  const repaired = {};
  let changed = false;

  Object.entries(store).forEach(([id, messages]) => {
    if (!Array.isArray(messages)) {
      changed = true;
      return;
    }

    const normalized = messages.map(normalizeHistoryEntry).filter(Boolean);
    const compacted = compactRepeatedHistory(normalized);
    repaired[id] = compacted;

    if (compacted.length !== messages.length || normalized.length !== messages.length) {
      changed = true;
    }
  });

  return { store: repaired, changed };
}

function migrateLegacyHistoryForSession() {
  if (!sessionId) return;
  const shortId = sessionId.slice(0, 8);
  if (!chatStore[shortId]) return;

  chatStore[sessionId] = [
    ...(chatStore[sessionId] || []),
    ...chatStore[shortId],
  ];
  delete chatStore[shortId];
  saveChatStore();
}

function normalizeSources(sources) {
  if (!Array.isArray(sources)) return [];
  return sources.map((source) => ({
    title: source.metadata?.source || source.metadata?.file_name || source.metadata?.source_name || 'unknown',
    snippet: (source.content || source.page_content || '').slice(0, 300),
    content: source.content || source.page_content || '',
    metadata: source.metadata || {},
  }));
}

function saveChatMessage(role, text, options = {}) {
  const cleanedText = (text || '').trim();
  if (!cleanedText || !sessionId) return;

  const entry = {
    role,
    text: cleanedText,
    mode: options.mode || dom.modeSelect.value,
    created_at: new Date().toISOString(),
  };

  const sources = normalizeSources(options.sources);
  if (sources.length) entry.sources = sources;

  chatStore[sessionId] = [...(chatStore[sessionId] || []), entry];
  saveChatStore();
}

function renderHistory() {
  if (!dom.chatHistoryList) return;

  const sessionIds = Object.keys(chatStore)
    .filter((id) => Array.isArray(chatStore[id]) && chatStore[id].length > 0)
    .sort((a, b) => {
      const aMessages = chatStore[a];
      const bMessages = chatStore[b];
      const aLast = aMessages[aMessages.length - 1]?.created_at || '';
      const bLast = bMessages[bMessages.length - 1]?.created_at || '';
      return bLast.localeCompare(aLast);
    });

  dom.chatHistoryList.innerHTML = '';

  if (!sessionIds.length) {
    const empty = document.createElement('div');
    empty.className = 'history-empty';
    empty.textContent = 'No saved chats yet.';
    dom.chatHistoryList.appendChild(empty);
    return;
  }

  sessionIds.forEach((id) => {
    const messages = chatStore[id];
    const btn = document.createElement('button');
    btn.className = 'btn secondary';
    btn.type = 'button';
    btn.style.width = '100%';
    btn.style.marginBottom = '5px';
    btn.title = id;
    btn.textContent = `${id.slice(0, 8)}... (${messages.length})`;
    btn.addEventListener('click', () => loadChat(id));
    dom.chatHistoryList.appendChild(btn);
  });
}

function loadChat(id) {
  const messages = chatStore[id] || [];
  setActiveSession(id);
  dom.messageList.innerHTML = '';
  setMessageCount(0);
  document.body.classList.toggle('chat-active', messages.length > 0);

  if (!messages.length) {
    showEmptyState();
    return;
  }

  messages.forEach((message) => {
    addMessage(message.text, message.role || 'assistant', { skipHistory: true });
    if (message.role === 'assistant' && Array.isArray(message.sources) && message.sources.length) {
      addSourceBlock(message.sources);
    }
  });
}

function formatChatTranscript(id, messages) {
  const lines = [
    `Session: ${id}`,
    `Exported: ${new Date().toLocaleString()}`,
    '',
  ];

  messages.forEach((message) => {
    lines.push(`${(message.role || 'assistant').toUpperCase()}:`);
    lines.push(message.text || '');

    if (Array.isArray(message.sources) && message.sources.length) {
      lines.push('');
      lines.push('SOURCES:');
      message.sources.forEach((source, index) => {
        lines.push(`${index + 1}. ${source.title || 'unknown'}`);
        if (source.snippet) lines.push(`   ${source.snippet}`);
      });
    }

    lines.push('');
  });

  return lines.join('\n');
}

function downloadCurrentChat() {
  const repaired = repairChatStore(chatStore);
  if (repaired.changed) {
    chatStore = repaired.store;
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(chatStore));
    renderHistory();
  }

  const messages = chatStore[sessionId] || [];
  if (!messages.length) {
    alert('No chat to download.');
    return;
  }

  const blob = new Blob([formatChatTranscript(sessionId, messages)], { type: 'text/plain' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  const safeId = (sessionId || 'session').replace(/[^a-z0-9_-]/gi, '_').slice(0, 40);
  link.href = url;
  link.download = `chat_${safeId}.txt`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
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
function setMessageCount(value) {
  messageCount = Math.max(0, value);
  const el = document.getElementById('msg-count');
  if (el) el.textContent = `${messageCount} message${messageCount !== 1 ? 's' : ''}`;
}

function updateMsgCount(delta) {
  setMessageCount(messageCount + delta);
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
  if (!options.placeholder) {
    document.body.classList.add('chat-active');
  }

  hideEmptyState();

  const isUser = sender === 'user';
  const wrapper = document.createElement('div');
  wrapper.className = `message-wrapper ${sender}`;

  const labelRow = document.createElement('div');
  labelRow.className = 'msg-label-row';

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar';
  avatar.innerHTML = isUser ? USER_ICON : AI_ICON;

  const nameEl = document.createElement('span');
  nameEl.className = 'msg-sender-name';
  nameEl.textContent = isUser ? 'You' : 'RAG Assistant';

  labelRow.appendChild(avatar);
  labelRow.appendChild(nameEl);

  const message = document.createElement('div');
  message.className = `message ${sender}`;
  if (options.placeholder) message.classList.add('placeholder');
  message.textContent = text;

  wrapper.appendChild(labelRow);
  wrapper.appendChild(message);
  dom.messageList.appendChild(wrapper);
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
function escapeHtml(value) {
  return String(value || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

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
        const title = s.title || s.metadata?.source || s.metadata?.file_name || s.metadata?.source_name || 'unknown';
        const snippet = (s.snippet || s.content || s.page_content || '').slice(0, 200);
        return `
        <div class="source-item">
          <div class="source-title">${i + 1}. ${escapeHtml(title)}</div>
          <div class="source-snippet">${escapeHtml(snippet)}...</div>
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
  saveChatMessage('user', question);
  setProcessing(true);
  const placeholder = addMessage('Thinking…', 'assistant', { placeholder: true });

  try {
    const endpoint = dom.modeSelect.value === 'normal' ? 'chat' : 'ask';

    if (endpoint === 'ask' && !sessionId) {
      if (placeholder) placeholder.textContent = 'Error: please index files or pasted text first before asking in retrieval mode.';
      saveChatMessage('assistant', 'Error: please index files or pasted text first before asking in retrieval mode.');
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
      saveChatMessage('assistant', 'Error: Server returned invalid JSON.');
      setProcessing(false);
      return { error: 'Server returned invalid JSON.' };
    }

    if (!response.ok) {
      const errorText = payload.error || payload.message || 'Request failed';
      if (placeholder) placeholder.textContent = `Error: ${errorText}`;
      saveChatMessage('assistant', `Error: ${errorText}`);
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
    saveChatMessage('assistant', answer, { sources });

    setProcessing(false);
    return payload;
  } catch (error) {
    if (placeholder) placeholder.textContent = `Error: ${error.message}`;
    saveChatMessage('assistant', `Error: ${error.message}`);
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
  setMessageCount(0);
  document.body.classList.remove('chat-active');
  delete chatStore[sessionId];
  saveChatStore();
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
  if (dom.downloadChatBtn) dom.downloadChatBtn.addEventListener('click', downloadCurrentChat);

  migrateLegacyHistoryForSession();
  renderHistory();
  updateBackendHealth();
  setInterval(updateBackendHealth, 15000);
}

init();
