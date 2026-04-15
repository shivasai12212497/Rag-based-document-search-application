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

let sessionId = localStorage.getItem('rag_session_id') || '';
function setActiveSession(id) {
  if (!id) return;
  sessionId = id;
  localStorage.setItem('rag_session_id', sessionId);
  dom.sessionId.textContent = `Session: ${sessionId.slice(0, 8)}...`;
}

if (!sessionId) {
  setActiveSession(crypto.randomUUID ? crypto.randomUUID() : `sess_${Date.now()}`);
} else {
  dom.sessionId.textContent = `Session: ${sessionId.slice(0, 8)}...`;
}

function setMode() {
  const mode = dom.modeSelect.value;
  dom.fileSection.classList.toggle('hidden', mode !== 'files');
  dom.pasteSection.classList.toggle('hidden', mode !== 'pasted');

  const modeInfo = document.getElementById('mode-info');
  if (mode === 'normal') {
    modeInfo.textContent = 'Normal Chat: no retrieval, direct LLM response (quick answers, may hallucinate).';
  } else if (mode === 'files') {
    modeInfo.textContent = 'Ask From Files: Uses indexed documents for grounded answers (best for precision).';
  } else {
    modeInfo.textContent = 'Ask From Pasted Text: Uses pasted content as knowledge source for that session.';
  }
}

async function updateBackendHealth() {
  try {
    const res = await fetch(`${backend}/health`);
    const data = await parseJsonResponse(res);
    dom.backendStatus.textContent = data.status === 'ok' ? `online • ${data.active_sessions} sessions` : 'unhealthy';
  } catch (error) {
    dom.backendStatus.textContent = 'unavailable';
  }
}

function addMessage(text, sender = 'assistant', options = {}) {
  if (!text) return null;
  const message = document.createElement('div');
  message.className = `message ${sender}`;
  if (options.placeholder) {
    message.classList.add('placeholder');
  }
  message.textContent = text;
  dom.messageList.appendChild(message);
  dom.messageList.scrollTop = dom.messageList.scrollHeight;
  return message;
}

function setProcessing(active) {
  isProcessing = active;
  dom.promptText.disabled = active;
  if (dom.sendButton) dom.sendButton.disabled = active;
  dom.processingStatus.textContent = active ? 'Assistant is thinking... please wait.' : '';
}

function addSourceBlock(sources) {
  const block = document.createElement('div');
  block.className = 'message assistant source-block';
  block.innerHTML = `
    <div class="source-header" onclick="this.nextElementSibling.classList.toggle('hidden')">
      Sources (${sources.length}) - Click to expand
    </div>
    <div class="source-list hidden">
      ${sources.map((s, i) => `
        <div class="source-item">
          <div class="source-title">${i + 1}. ${s.metadata.source || 'unknown'}</div>
          <div class="source-snippet">${s.content.slice(0, 200)}...</div>
        </div>
      `).join('')}
    </div>
  `;
  dom.messageList.appendChild(block);
  dom.messageList.scrollTop = dom.messageList.scrollHeight;
}

async function parseJsonResponse(response) {
  const text = await response.text();
  try {
    return JSON.parse(text);
  } catch (err) {
    console.error('Not JSON response:', text);
    throw new Error('Server returned invalid JSON.');
  }
}

function sanitize(text) {
  if (!text) return '';
  return text
    .split('\n')
    .filter((line) => !line.trim().startsWith('Role:'))
    .join('\n')
    .trim();
}

async function askQuestion(question) {
  if (isProcessing) {
    addMessage('Please wait until the assistant finishes thinking.', 'assistant');
    return;
  }

  addMessage(question, 'user');
  setProcessing(true);
  const placeholder = addMessage('Thinking...', 'assistant', { placeholder: true });

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
      body: JSON.stringify({
        question,
        session_id: sessionId,
      }),
    });

    const text = await response.text();
    let payload;
    try {
      payload = JSON.parse(text);
    } catch (err) {
      console.error('Not JSON response:', text);
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
    if (placeholder) placeholder.textContent = answer;
    if (payload.sources && payload.sources.length > 0) {
      addSourceBlock(payload.sources);
    }

    setProcessing(false);
    return payload;
  } catch (error) {
    if (placeholder) placeholder.textContent = `Error: ${error.message}`;
    setProcessing(false);
    return { error: error.message };
  }
}

async function indexFiles() {
  const files = dom.fileUploader.files;
  if (!files || files.length === 0) return addMessage('Error: choose files to index', 'assistant');

  const form = new FormData();
  form.append('session_id', sessionId);
  for (const file of files) form.append('files', file);

  dom.indexFilesBtn.disabled = true;
  dom.indexFilesBtn.textContent = 'Indexing...';

  try {
    const res = await fetch(`${backend}/upload`, { method: 'POST', body: form });
    const payload = await parseJsonResponse(res);
    if (!res.ok) throw new Error(payload.error || 'Upload failed');

    if (payload.session_id) {
      setActiveSession(payload.session_id);
    }
    const indexedCount = Number.isInteger(payload.files_indexed_count)
      ? payload.files_indexed_count
      : Array.isArray(payload.files_indexed)
      ? payload.files_indexed.length
      : 0;
    addMessage(`Indexed ${indexedCount} file(s).`, 'assistant');
  } catch (err) {
    addMessage(`Index error: ${err.message}`, 'assistant');
  } finally {
    dom.indexFilesBtn.disabled = false;
    dom.indexFilesBtn.textContent = 'Index files';
  }
}

async function indexPastedText() {
  const text = dom.pasteContent.value.trim();
  const name = dom.pasteName.value.trim() || 'notes.txt';
  if (!text) return addMessage('Error: paste some text first.', 'assistant');

  const form = new FormData();
  form.append('session_id', sessionId);
  form.append('text_content', text);
  form.append('text_name', name);

  dom.indexPasteBtn.disabled = true;
  dom.indexPasteBtn.textContent = 'Indexing...';

  try {
    const res = await fetch(`${backend}/upload`, { method: 'POST', body: form });
    const payload = await parseJsonResponse(res);
    if (!res.ok) throw new Error(payload.error || 'Upload failed');

    if (payload.session_id) {
      setActiveSession(payload.session_id);
    }
    addMessage(`Indexed pasted text as ${name}.`, 'assistant');
  } catch (err) {
    addMessage(`Index error: ${err.message}`, 'assistant');
  } finally {
    dom.indexPasteBtn.disabled = false;
    dom.indexPasteBtn.textContent = 'Index pasted content';
  }
}

function clearHistory() {
  dom.messageList.innerHTML = '';
}

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
