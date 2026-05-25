const layout = document.querySelector(".layout");
const messages = document.getElementById("messages");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const indexBtn = document.getElementById("indexBtn");
const indexStatus = document.getElementById("indexStatus");
const sessionsList = document.getElementById("sessionsList");
const topbarTitle = document.getElementById("topbarTitle");
const modeBadge = document.getElementById("modeBadge");
const xaiToggle = document.getElementById("xaiToggle");
const xaiClose = document.getElementById("xaiClose");
const xaiPanel = document.getElementById("xaiPanel");
const xaiCards = document.getElementById("xaiCards");
const xaiCount = document.getElementById("xaiCount");
const xaiFooter = document.getElementById("xaiFooter");
const welcome = document.getElementById("welcome");

let isStreaming = false;
let currentSession = null;

// Toggling XAI attribution panel open and closed
xaiToggle.addEventListener("click", () => {
  layout.classList.toggle("xai-open");
  xaiToggle.classList.toggle("active");
});

xaiClose.addEventListener("click", () => {
  layout.classList.remove("xai-open");
  xaiToggle.classList.remove("active");
});

// Rendering XAI attribution cards in panel
function renderAttribution(chunks, mode) {
  xaiCards.innerHTML = "";
  xaiCount.textContent = `${chunks.length} chunks`;

  modeBadge.textContent = mode === "rag" ? "● RAG" : "● General";
  modeBadge.className = `mode-badge ${mode}`;

  if (!chunks.length) {
    xaiCards.innerHTML = '<div class="xai-empty">No chunks retrieved.</div>';
    return;
  }

  chunks.forEach((chunk, i) => {
    const rank = i + 1;
    const card = document.createElement("div");
    card.className = `xai-card rank-${rank}`;

    card.innerHTML = `
      <div class="xai-card-top">
        <span class="xai-fn" title="${chunk.function_name}">${
      chunk.function_name
    }</span>
        <span class="xai-pct">${chunk.attribution_pct.toFixed(1)}%</span>
      </div>
      <div class="xai-file">${chunk.file_name}</div>
      <div class="xai-bar">
        <div class="xai-bar-fill" style="width:0%" data-target="${Math.min(
          chunk.attribution_pct,
          100
        ).toFixed(1)}"></div>
      </div>
      <div class="xai-scores">
        <div class="xai-score">
          <span class="xai-score-label">CrossEnc</span>
          <span class="xai-score-value">${chunk.cross_encoder_score.toFixed(
            3
          )}</span>
        </div>
        <div class="xai-score">
          <span class="xai-score-label">RRF</span>
          <span class="xai-score-value">${chunk.rrf_score.toFixed(4)}</span>
        </div>
      </div>
    `;
    xaiCards.appendChild(card);
  });

  // Animating progress bars after render
  requestAnimationFrame(() => {
    document.querySelectorAll(".xai-bar-fill").forEach((bar) => {
      setTimeout(() => {
        bar.style.width = `${bar.dataset.target}%`;
      }, 60);
    });
  });

  // Rendering XAI footer summary
  const top = chunks[0];
  xaiFooter.innerHTML = `
    <div class="xai-footer-row">
      <span class="xai-footer-label">Mode</span>
      <span class="xai-footer-value ${
        mode === "rag" ? "green" : ""
      }">${mode.toUpperCase()}</span>
    </div>
    <div class="xai-footer-row">
      <span class="xai-footer-label">Top chunk</span>
      <span class="xai-footer-value purple">${
        top.function_name
      } · ${top.attribution_pct.toFixed(1)}%</span>
    </div>
    <div class="xai-footer-row">
      <span class="xai-footer-label">Pipeline</span>
      <span class="xai-footer-value">BM25+Dense→RRF→CE</span>
    </div>
  `;

  // Auto-opening XAI panel on first attribution
  if (!layout.classList.contains("xai-open")) {
    layout.classList.add("xai-open");
    xaiToggle.classList.add("active");
  }
}

// Loading all sessions and rendering in sidebar
async function loadSessions() {
  try {
    const res = await fetch("/sessions");
    const sessions = await res.json();
    renderSessions(sessions);
  } catch (e) {
    console.error("Failed loading sessions:", e);
  }
}

// Rendering session list grouped by date
// Rendering session list grouped by date
function renderSessions(sessions) {
  sessionsList.innerHTML = "";
  if (!sessions.length) return;

  const label = document.createElement("div");
  label.className = "session-group";
  label.textContent = "Recents";
  sessionsList.appendChild(label);

  sessions.forEach((s) => {
    const item = document.createElement("div");
    item.className = `session-item${s.id === currentSession ? " active" : ""}`;
    item.dataset.id = s.id;
    item.innerHTML = `
      <span class="session-title" title="${s.title}">${s.title}</span>
      <span class="session-del" data-del="${s.id}" title="Delete">✕</span>
    `;
    item.addEventListener("click", (e) => {
      if (e.target.dataset.del) {
        deleteSession(e.target.dataset.del);
        return;
      }
      loadSession(s.id, s.title);
    });
    sessionsList.appendChild(item);
  });
}

// Loading existing session messages into chat
async function loadSession(sessionId, title) {
  try {
    const res = await fetch(`/sessions/${sessionId}`);
    const msgs = await res.json();

    currentSession = sessionId;
    topbarTitle.textContent = title;
    messages.innerHTML = "";

    msgs.forEach((msg) => {
      if (msg.role === "user") {
        appendUserMessage(msg.content);
      } else if (msg.role === "assistant") {
        const bubble = createAssistantBubble(msg.attribution);
        bubble.innerHTML = renderMarkdown(msg.content);
        hljs.highlightAll();
      }
    });

    scrollToBottom();
    loadSessions();
  } catch (e) {
    console.error("Failed loading session:", e);
  }
}

// Deleting session from database and refreshing sidebar
async function deleteSession(sessionId) {
  try {
    await fetch(`/sessions/${sessionId}`, { method: "DELETE" });
    if (currentSession === sessionId) {
      currentSession = null;
      topbarTitle.textContent = "DevAssist";
      messages.innerHTML = "";
      messages.appendChild(welcome);
    }
    loadSessions();
  } catch (e) {
    console.error("Failed deleting session:", e);
  }
}

// Starting a new chat session
function startNewChat() {
  currentSession = null;
  topbarTitle.textContent = "DevAssist";
  messages.innerHTML = "";
  messages.appendChild(welcome);
  xaiCards.innerHTML =
    '<div class="xai-empty">Attribution scores appear after your first query.</div>';
  xaiFooter.innerHTML = "";
  xaiCount.textContent = "";
  modeBadge.className = "mode-badge";
  modeBadge.textContent = "";
  loadSessions();
}

// Appending user message bubble to chat
function appendUserMessage(text) {
  if (welcome.parentNode) welcome.remove();
  const msg = document.createElement("div");
  msg.className = "msg user";
  msg.innerHTML = `
    <div class="msg-label">You</div>
    <div class="bubble user">${escapeHtml(text)}</div>
  `;
  messages.appendChild(msg);
  scrollToBottom();
}

// Creating assistant message bubble with streaming cursor
function createAssistantBubble(attribution) {
  const chunks = attribution ? attribution.length : 0;
  const msg = document.createElement("div");
  msg.className = "msg assistant";
  msg.innerHTML = `
    <div class="msg-label">
      DevAssist
      ${
        chunks
          ? `<span class="msg-chunk-badge">RAG · ${chunks} chunks</span>`
          : ""
      }
    </div>
    <div class="bubble assistant" id="streamBubble"><span class="cursor"></span></div>
  `;
  messages.appendChild(msg);
  scrollToBottom();
  return document.getElementById("streamBubble");
}

// Appending memory saved toast to chat
function appendMemoryToast(content) {
  if (welcome.parentNode) welcome.remove();
  const toast = document.createElement("div");
  toast.className = "memory-toast";
  toast.innerHTML = `<span>◈</span> Memory saved — ${escapeHtml(content)}`;
  messages.appendChild(toast);
  scrollToBottom();
}

// Sending query and streaming response via SSE
async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query || isStreaming) return;

  isStreaming = true;
  sendBtn.disabled = true;
  queryInput.value = "";
  queryInput.style.height = "auto";

  appendUserMessage(query);

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id: currentSession, top_k: 5 }),
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let bubble = null;
    let fullText = "";

    // Reading SSE stream tokens from backend
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const lines = decoder.decode(value).split("\n");
      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;

        try {
          const event = JSON.parse(raw);

          if (event.type === "session_created") {
            currentSession = event.data;
            loadSessions();
          } else if (event.type === "attribution") {
            renderAttribution(event.data, event.mode);
            bubble = createAssistantBubble(event.data);
            if (event.session_id) {
              currentSession = event.session_id;
              loadSessions();
            }
          } else if (event.type === "token") {
            if (!bubble) bubble = createAssistantBubble(null);
            fullText += event.data;
            bubble.innerHTML =
              renderMarkdown(fullText) + '<span class="cursor"></span>';
            scrollToBottom();
          } else if (event.type === "done") {
            if (bubble) {
              bubble.innerHTML = renderMarkdown(fullText);
              hljs.highlightAll();
            }
            topbarTitle.textContent =
              query.slice(0, 42) + (query.length > 42 ? "..." : "");
            loadSessions();
            scrollToBottom();
          } else if (event.type === "memory_saved") {
            appendMemoryToast(event.data.replace("Memory saved: ", ""));
          }
        } catch {
          continue;
        }
      }
    }
  } catch (err) {
    const b = document.getElementById("streamBubble");
    if (b) b.innerHTML = `<span style="color:#ef4444">Connection error.</span>`;
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    queryInput.focus();
  }
}

// Rendering markdown with syntax-highlighted code blocks
function renderMarkdown(text) {
  let html = escapeHtml(text);
  html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
    const l = lang || "plaintext";
    const raw = unescapeHtml(code.trim());
    try {
      const highlighted = hljs.highlight(raw, { language: l }).value;
      return `<pre><code class="hljs language-${l}">${highlighted}</code></pre>`;
    } catch {
      return `<pre><code class="hljs">${code.trim()}</code></pre>`;
    }
  });
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html
    .split("\n\n")
    .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
    .join("");
  return html;
}

function escapeHtml(t) {
  return t
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function unescapeHtml(t) {
  return t
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"');
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

// Binding event listeners
sendBtn.addEventListener("click", sendQuery);
document.getElementById("newChatBtn").addEventListener("click", startNewChat);
indexBtn.addEventListener("click", async () => {
  indexBtn.disabled = true;
  indexStatus.textContent = "Indexing...";
  try {
    const res = await fetch("/index", { method: "POST" });
    const data = await res.json();
    indexStatus.textContent = `✓ ${data.indexed_files} files · ${data.total_chunks} chunks`;
  } catch {
    indexStatus.textContent = "✗ Failed";
  } finally {
    indexBtn.disabled = false;
  }
});

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuery();
  }
});

queryInput.addEventListener("input", () => {
  queryInput.style.height = "auto";
  queryInput.style.height = `${Math.min(queryInput.scrollHeight, 160)}px`;
});

// Initializing app on load
loadSessions();

const modelSelector = document.getElementById("modelSelector");
let selectedModel = "devassist";

modelSelector.addEventListener("click", (e) => {
  modelSelector.classList.toggle("open");
  e.stopPropagation();
});

modelSelector.addEventListener("click", (e) => {
  const option = e.target.closest(".model-option");
  if (!option) return;
  selectedModel = option.dataset.model;
  modelSelector.querySelector(".model-name").textContent = selectedModel;
  document
    .querySelectorAll(".model-option")
    .forEach((o) => o.classList.remove("active"));
  option.classList.add("active");
  modelSelector.classList.remove("open");
});

document.addEventListener("click", () => {
  modelSelector.classList.remove("open");
});
