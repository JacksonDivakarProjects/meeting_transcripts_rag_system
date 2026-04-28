import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';
import './App.css';

const API_URL = import.meta.env.VITE_API_URL;

console.log('API URL:', API_URL);
export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
  }, [input]);

  const handleSend = async () => {
    const question = input.trim();
    if (!question || loading) return;

    const userMsg = { role: 'user', content: question };
    const history = [...messages, userMsg];
    setMessages(history);
    setInput('');
    setLoading(true);

    try {
      const historyPayload = messages.map(m => ({ role: m.role, content: m.content }));

      const { data } = await axios.post(`${API_URL}/query`, {
        question,
        chat_history: historyPayload,
        hybrid: true,
        bm25_weight: 0.3,
      });

      setMessages([...history, {
        role: 'assistant',
        content: data.answer,
        sources: data.sources || [],
      }]);
    } catch (err) {
      const detail = err.response?.data?.detail || err.message || 'Connection error — is the API running?';
      setMessages([...history, { role: 'assistant', content: `❌ ${detail}`, sources: [] }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app-container">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <h2>✨ Meeting Analyst</h2>
        <p>Ask about meeting transcripts. Answers include source citations.</p>
        <hr />
        <p className="tip">
          💡 Tip: prefix with <code>/meeting</code> to bypass intent filtering and go straight to search.
        </p>
        <button onClick={() => setMessages([])} className="clear-btn">🗑️ Clear Chat</button>
        <div className="footer-info">v1.1 · Enterprise RAG</div>
      </aside>

      {/* ── Main chat area ── */}
      <main className="chat-area">
        <div className="header">
          <h1>Meeting Analyst</h1>
          <p>AI-powered Q&amp;A over your meeting transcripts.</p>
        </div>

        <div className="messages-container">
          {messages.length === 0 && (
            <div className="empty-state">
              <span>Start a conversation about your meetings…</span>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`message-row ${msg.role}`}>
              <div className="avatar">{msg.role === 'user' ? '👤' : '🤖'}</div>
              <div className={`message-bubble ${msg.role}`}>
                {msg.role === 'assistant' ? (
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                ) : (
                  <span>{msg.content}</span>
                )}

                {msg.role === 'assistant' && msg.sources?.length > 0 && (
                  <details className="sources-expander">
                    <summary>📚 View {msg.sources.length} source{msg.sources.length !== 1 ? 's' : ''}</summary>
                    <div className="sources-list">
                      {msg.sources.map((src, i) => (
                        <div key={i} className="source-item">
                          <strong>👤 {src.speaker || 'Unknown'}</strong>
                          {' at '}
                          <code>{src.timestamp_str || '00:00'}</code>
                          {' — '}
                          <em>{src.source_file || 'Unknown file'}</em>
                          {src.topic && <span className="source-topic"> · {src.topic}</span>}
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message-row assistant">
              <div className="avatar">🤖</div>
              <div className="message-bubble assistant loading">
                <div className="spinner" />
                Retrieving context…
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>

        {/* ── Input bar ── */}
        <div className="input-container">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the meetings… (Shift+Enter for new line)"
            rows={1}
            disabled={loading}
          />
          <button onClick={handleSend} disabled={loading || !input.trim()}>
            {loading ? <div className="spinner btn-spinner" /> : 'Send'}
          </button>
        </div>
      </main>
    </div>
  );
}
