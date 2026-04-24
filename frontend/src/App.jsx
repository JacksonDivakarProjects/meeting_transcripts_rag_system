import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import axios from 'axios';

function cleanAnswer(text) {
  if (!text) return "";
  
  text = text.replace(/<[^>]+>.*?<\/[^>]+>/g, "");
  text = text.replace(/<[^>]+>/g, "");
  text = text.replace(/\n\s*\n/g, "\n\n").trim();
  return text;
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = { role: "user", content: input };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      // Prepare history for backend (exclude the very last message we just added)
      const historyPayload = newMessages.slice(0, -1).map(m => ({
        role: m.role,
        content: m.content
      }));

      const response = await axios.post('/query', {
        question: input,
        chat_history: historyPayload,
        hybrid: true,
        bm25_weight: 0.3
      });

      const answer = cleanAnswer(response.data.answer);
      const sources = response.data.sources || [];

      setMessages([...newMessages, { role: "assistant", content: answer, sources }]);
    } catch (error) {
      const errMsg = error.response?.data?.detail || "Connection Error. Is FastAPI running?";
      setMessages([...newMessages, { role: "assistant", content: `❌ ${errMsg}`, sources: [] }]);
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
      <div className="sidebar">
        <h2>✨ Meeting Analyst</h2>
        <p>Ask about meeting transcripts. Answers include source citations.</p>
        <hr />
        <button onClick={() => setMessages([])} className="clear-btn">🗑️ Clear Chat</button>
        <div className="footer-info">v1.0 – Enterprise RAG</div>
      </div>

      <main className="chat-area">
        <div className="header">
          <h1>Meeting Analyst</h1>
          <p>Ask questions about your transcripts with AI-powered context.</p>
        </div>

        <div className="messages-container">
          {messages.length === 0 && (
            <div className="empty-state">Start a conversation about your meetings...</div>
          )}

          {messages.map((msg, idx) => (
            <div key={idx} className={`message-row ${msg.role}`}>
              <div className="avatar">
                {msg.role === 'user' ? '👤' : '🤖'}
              </div>
              <div className={`message-bubble ${msg.role}`}>
                {msg.role === 'assistant' ? (
                  <ReactMarkdown>{msg.content}</ReactMarkdown>
                ) : (
                  msg.content
                )}

                {msg.role === 'assistant' && msg.sources?.length > 0 && (
                  <details className="sources-expander">
                    <summary>📚 View {msg.sources.length} Citations</summary>
                    <div className="sources-list">
                      {msg.sources.map((src, i) => (
                        <div key={i} className="source-item">
                          <strong>👤 {src.speaker || 'Unknown'}</strong> at <code>{src.timestamp_str || '00:00'}</code> – <em>{src.source_file || 'Unknown File'}</em>
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
                <div className="spinner"></div> Retrieving context...
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <div className="input-container">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about the meetings... (Shift+Enter for new line)"
            rows={1}
            disabled={loading}
          />
          <button onClick={handleSend} disabled={loading || !input.trim()}>
            Send
          </button>
        </div>
      </main>
    </div>
  );
}