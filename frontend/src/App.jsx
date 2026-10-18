import { useState } from 'react'
import './App.css'

const API = 'YOUR_BACKEND_URL'

function App() {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [history, setHistory] = useState([])

  async function analyze() {
    if (!text.trim()) {
      setError('Enter some text before starting the analysis.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const response = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      })

      const data = await response.json()
      if (!response.ok) throw new Error(data.detail || 'Analysis failed.')

      setResult(data)
      setHistory((previous) => [
        {
          text: text.trim(),
          sentiment: data.sentiment,
          confidence: data.confidence,
          time: new Date().toLocaleTimeString(),
        },
        ...previous,
      ].slice(0, 5))
    } catch {
      setError('Could not connect to the API. Check that your backend is running on port 8000.')
    } finally {
      setLoading(false)
    }
  }

  const positive = result?.probabilities?.positive ?? 0
  const negative = result?.probabilities?.negative ?? 0

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <aside className="sidebar glass">
        <a className="brand" href="#">
          <span className="brand-mark">M</span>
          <span>MEGAN<span className="brand-ai"> AI</span></span>
        </a>

        <div className="nav-label">WORKSPACE</div>
        <a className="nav-item active" href="#overview"><span>◈</span> Overview</a>
        <a className="nav-item" href="#analyzer"><span>⌁</span> Sentiment lab</a>
        <a className="nav-item" href="#history"><span>◷</span> Recent analyses</a>

        <div className="sidebar-bottom">
          <div className="status-card">
            <span className="status-dot" />
            <div><b>Neural engine</b><small>Local API connection</small></div>
          </div>
          <small className="version">MEGAN AI · RESEARCH PREVIEW</small>
        </div>
      </aside>

      <section className="main-content" id="overview">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> Overview</div>
          <div className="system-status"><span className="status-dot" /> ML ENGINE</div>
        </header>

        <section className="welcome">
          <div>
            <div className="eyebrow"><span className="sparkle">✳</span> SENTIMENT INTELLIGENCE</div>
            <h1>Understand the <span>feeling</span><br />behind every word.</h1>
            <p>Transform raw text into actionable sentiment signals with your machine-learning model.</p>
          </div>
          <div className="orbit-art" aria-label="Abstract AI visualization">
            <div className="orbit orbit-a" />
            <div className="orbit orbit-b" />
            <div className="orbit orbit-c" />
            <div className="orbit-core">M</div>
            <span className="orbit-node node-a" />
            <span className="orbit-node node-b" />
          </div>
        </section>

        <section className="stats-grid">
          <article className="stat-card glass">
            <div className="stat-top"><span>MODEL STATUS</span><span className="stat-icon">⌘</span></div>
            <div className="stat-value"><span className="status-dot" /> Ready to analyze</div>
            <div className="stat-foot">TF-IDF + Logistic Regression</div>
          </article>
          <article className="stat-card glass">
            <div className="stat-top"><span>ANALYSES THIS SESSION</span><span className="stat-icon">◷</span></div>
            <div className="stat-value">{history.length.toString().padStart(2, '0')}</div>
            <div className="stat-foot">New analyses in this browser session</div>
          </article>
          <article className="stat-card glass">
            <div className="stat-top"><span>CLASSIFICATION</span><span className="stat-icon">◈</span></div>
            <div className="stat-value">Binary sentiment</div>
            <div className="stat-foot">Positive · Negative</div>
          </article>
        </section>

        <section className="workspace-grid" id="analyzer">
          <article className="analyzer-card glass">
            <div className="section-heading">
              <div><div className="eyebrow">LIVE INFERENCE</div><h2>Sentiment lab</h2></div>
              <span className="live-badge"><span className="status-dot" /> LIVE</span>
            </div>

            <label className="field-label" htmlFor="review">INPUT TEXT</label>
            <textarea
              id="review"
              value={text}
              onChange={(event) => setText(event.target.value)}
              maxLength={5000}
              placeholder="Paste a product review, customer feedback, or any text you want MEGAN AI to analyze..."
            />
            <div className="input-meta"><span>Natural language input</span><span>{text.length}/5000</span></div>

            <div className="examples">
              <span>TRY AN EXAMPLE</span>
              <button onClick={() => setText('This product is excellent and works perfectly.')}>Positive review ↗</button>
              <button onClick={() => setText('The service was terrible and I am very disappointed.')}>Negative review ↗</button>
            </div>

            <button className="analyze-button" onClick={analyze} disabled={loading}>
              {loading ? 'Analyzing text...' : '✳  Analyze sentiment'} <span>→</span>
            </button>

            {error && <div className="error-message">{error}</div>}

            {result && (
              <div className="result-panel">
                <div className="result-title">MODEL PREDICTION</div>
                <div className="result-main">
                  <div className={`sentiment-symbol ${result.sentiment.toLowerCase()}`}>
                    {result.sentiment === 'Positive' ? '↗' : '↘'}
                  </div>
                  <div><div className="result-sentiment">{result.sentiment}</div><small>Predicted sentiment</small></div>
                  <div className="confidence"><b>{result.confidence}%</b><small>Class probability</small></div>
                </div>
                <div className="probability-row"><span>Positive</span><div className="progress"><div className="positive-fill" style={{ width: `${positive}%` }} /></div><b>{positive}%</b></div>
                <div className="probability-row"><span>Negative</span><div className="progress"><div className="negative-fill" style={{ width: `${negative}%` }} /></div><b>{negative}%</b></div>
                <p className="disclaimer">Probabilities are model estimates, not guarantees of correctness.</p>
              </div>
            )}
          </article>

          <aside className="insights-column">
            <article className="insight-card glass">
              <div className="eyebrow">SYSTEM INTELLIGENCE</div>
              <h3>Inside the model</h3>
              <p>Your text is transformed into weighted word and phrase features before the classifier estimates sentiment.</p>
              <div className="pipeline">
                <div><span>01</span><b>Text input</b><small>Raw language</small></div>
                <span className="pipeline-arrow">↓</span>
                <div><span>02</span><b>TF-IDF vectors</b><small>Word & phrase features</small></div>
                <span className="pipeline-arrow">↓</span>
                <div><span>03</span><b>Logistic regression</b><small>Probability estimates</small></div>
              </div>
            </article>

            <article className="insight-card glass">
              <div className="eyebrow">DATA FOUNDATION</div>
              <h3>Trained on real reviews</h3>
              <p>The project uses the UCI Sentiment Labelled Sentences dataset.</p>
              <div className="source-list"><span>Amazon reviews</span><span>IMDb reviews</span><span>Yelp reviews</span></div>
              <small className="muted">Predictions are limited by the training data and model performance.</small>
            </article>
          </aside>
        </section>

        <section className="history-card glass" id="history">
          <div className="section-heading">
            <div><div className="eyebrow">SESSION ACTIVITY</div><h2>Recent analyses</h2></div>
            <span className="history-count">{history.length} records</span>
          </div>
          {history.length === 0 ? (
            <div className="empty-state"><span>◷</span><p>Your analyses will appear here after your first prediction.</p></div>
          ) : (
            <div className="history-list">
              {history.map((item, index) => (
                <div className="history-row" key={`${item.time}-${index}`}>
                  <span className={`history-indicator ${item.sentiment.toLowerCase()}`} />
                  <div className="history-text"><b>{item.text}</b><small>{item.time}</small></div>
                  <span className={`history-tag ${item.sentiment.toLowerCase()}`}>{item.sentiment}</span>
                  <b className="history-confidence">{item.confidence}%</b>
                </div>
              ))}
            </div>
          )}
        </section>

        <footer><span>MEGAN AI</span> · Machine-learning sentiment intelligence <span className="footer-right">BUILT FOR EXPLORATION</span></footer>
      </section>
    </main>
  )
}

export default App
