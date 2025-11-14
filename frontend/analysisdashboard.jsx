// src/components/AnalysisDashboard.jsx

import React, { useState, useMemo, useEffect } from 'react';
import axios from 'axios';
import Plot from 'react-plotly.js';

// The backend API URL
const API_URL = 'http://127.0.0.1:8000';

// Reusable component for displaying single metrics
const MetricCard = ({ title, value }) => (
  <div className="card metric-card">
    <h3>{title}</h3>
    <p>{value}</p>
  </div>
);

// Reusable component for the horizontal bar charts (Leaders and Parties)
const EntityBarChart = ({ title, data, entityType }) => {
  if (!data || data.length === 0) {
    return <div className="chart-placeholder">No {entityType} mentions found.</div>;
  }
  return (
    <Plot
      data={[{
        x: data.map(d => d.Mentions),
        y: data.map(d => d.entity_name),
        type: 'bar',
        orientation: 'h',
        hoverinfo: 'y+text',
        text: data.map(d => `Mentions: ${d.Mentions}, Avg Sentiment: ${d.Avg_Sentiment.toFixed(2)}`),
        marker: {
          color: data.map(d => d.Avg_Sentiment),
          colorscale: 'RdYlGn',
          reversescale: true,
          cmin: -1,
          cmax: 1,
          colorbar: { title: 'Sentiment' }
        }
      }]}
      layout={{
        title,
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: 'white' },
        yaxis: { autorange: 'reversed', automargin: true },
        xaxis: { automargin: true, title: 'Number of Mentions' },
        margin: { l: 150, r: 40, t: 50, b: 40 }
      }}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler={true}
      config={{ responsive: true }}
    />
  );
};

// Main Dashboard Component
export default function AnalysisDashboard() {
  // State management
  const [topics, setTopics] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('Please select a topic to begin.');
  const [results, setResults] = useState(null);
  const [timeSeriesFreq, setTimeSeriesFreq] = useState('D');
  const [selectedModel, setSelectedModel] = useState('gemini-pro');

  // 1. Fetch the list of available topics when the component first loads
  useEffect(() => {
    setStatus('Fetching available topics...');
    axios.get(`${API_URL}/topics/`)
      .then(response => {
        setTopics(response.data);
        if (response.data.length > 0 && response.data[0].id !== 0) {
          setSelectedTopic(response.data[0].id); // Default to the first topic
        } else {
          setStatus('Warning: No topics found in the database. Please add a topic first.');
        }
      })
      .catch(error => {
        console.error("Failed to fetch topics:", error);
        setStatus(`Error: Could not fetch topics. Is the backend running?`);
      });
  }, []); // Empty dependency array means this runs only once on mount

  // 2. Fetch all dashboard data whenever the selected topic or time frequency changes
  useEffect(() => {
    const fetchDashboardData = async () => {
      // Don't fetch if no valid topic is selected
      if (!selectedTopic || selectedTopic === 0) {
        setResults(null);
        return;
      }
      
      setLoading(true);
      setResults(null); // Clear previous results to show loading state
      setStatus(`Loading dashboard for topic ID: ${selectedTopic}...`);
      
      try {
        // Fetch both main dashboard data and time series data in parallel for speed
        const [dashboardRes, tsRes] = await Promise.all([
          axios.get(`${API_URL}/dashboard-data/${selectedTopic}`),
          axios.get(`${API_URL}/analyze-timeseries/${selectedTopic}?freq=${timeSeriesFreq}`)
        ]);

        const analysisData = {
          aiAnalysis: dashboardRes.data,
          timeSeries: tsRes.data
        };
        
        setResults(analysisData);
        setStatus('Dashboard loaded successfully. You can analyze new comments if needed.');
      } catch (err) {
        const errorMsg = err.response?.data?.detail || err.message;
        setStatus(`Error: ${errorMsg}`);
        setResults(null); // Ensure dashboard is cleared on error
      } finally {
        setLoading(false);
      }
    };

    fetchDashboardData();
  }, [selectedTopic, timeSeriesFreq]); // This effect re-runs if `selectedTopic` or `timeSeriesFreq` changes

  // 3. Function to trigger the analysis of NEW (pending) comments
  const runNewAnalysis = async () => {
    if (!selectedTopic) return alert("Please select a topic first.");
    
    setLoading(true);
    setStatus('Analyzing new comments... This may take several minutes.');
    
    try {
      await axios.post(`${API_URL}/analyze-comments/${selectedTopic}?model_choice=${selectedModel}`);
      
      // After analysis, we trigger a refresh of the dashboard data
      setStatus('Analysis complete! Refreshing dashboard...');
      const currentTopic = selectedTopic;
      // This is a simple trick to force the `useEffect` hook to re-run by changing its dependency
      setSelectedTopic(''); 
      setTimeout(() => setSelectedTopic(currentTopic), 100);

    } catch (err) {
      const errorMsg = err.response?.data?.detail || err.message;
      setStatus(`Error during analysis: ${errorMsg}`);
      setLoading(false); // Make sure loading stops on error
    }
  };

  // 4. Memoized function to process the raw API data into a format usable by the charts
  const processedData = useMemo(() => {
    if (!results?.aiAnalysis) return null;
    
    const { per_comment_data = [], per_entity_data = [] } = results.aiAnalysis;
    
    const calculateAvg = (scores) => {
      const validScores = scores.filter(s => typeof s === 'number');
      return validScores.length ? validScores.reduce((a, b) => a + b, 0) / validScores.length : 0;
    };

    const processEntities = (type) => {
      const entityMap = per_entity_data.filter(e => e.entity_type === type).reduce((acc, curr) => {
        if (!acc[curr.entity_name]) {
          acc[curr.entity_name] = { Mentions: 0, SentimentScores: [] };
        }
        acc[curr.entity_name].Mentions++;
        if (typeof curr.entity_sentiment_score === 'number') {
          acc[curr.entity_name].SentimentScores.push(curr.entity_sentiment_score);
        }
        return acc;
      }, {});
      return Object.entries(entityMap)
        .map(([name, data]) => ({
          entity_name: name,
          Mentions: data.Mentions,
          Avg_Sentiment: calculateAvg(data.SentimentScores)
        }))
        .sort((a, b) => b.Mentions - a.Mentions);
    };

    const sentimentCounts = per_comment_data.reduce((acc, curr) => {
      const score = curr.overall_sentiment_score;
      let label = 'Neutral';
      if (typeof score === 'number') {
        if (score > 0.15) label = 'Positive';
        else if (score < -0.15) label = 'Negative';
      }
      acc[label] = (acc[label] || 0) + 1;
      return acc;
    }, {});
    
    const orderedSentimentCounts = { 'Negative': sentimentCounts.Negative || 0, 'Positive': sentimentCounts.Positive || 0, 'Neutral': sentimentCounts.Neutral || 0 };

    const sentimentByLanguage = per_comment_data.reduce((acc, curr) => {
      const lang = curr.language || 'Unknown';
      if (!acc[lang]) acc[lang] = { scores: [], count: 0 };
      if (typeof curr.overall_sentiment_score === 'number') {
        acc[lang].scores.push(curr.overall_sentiment_score);
      }
      acc[lang].count++;
      return acc;
    }, {});

    return {
      totalComments: per_comment_data.length,
      avgSentiment: calculateAvg(per_comment_data.map(c => c.overall_sentiment_score)),
      sentimentCounts: orderedSentimentCounts,
      avgSentimentByLang: Object.entries(sentimentByLanguage).map(([lang, data]) => ({
        language: lang,
        avg_sentiment: calculateAvg(data.scores),
        count: data.count
      })),
      topLeaders: processEntities('LEADER'),
      topParties: processEntities('PARTY'),
    };
  }, [results]); // This calculation re-runs only when the `results` state changes

  return (
    <div className="container">
      <header className="header">
        <h1>🚀 Hybrid Political Analysis Engine</h1>
        <div className="controls">
          <div className="model-selector">
            <label htmlFor="topic-select">Analysis Topic:</label>
            <select id="topic-select" value={selectedTopic} onChange={(e) => setSelectedTopic(e.target.value)} className="chart-dropdown" disabled={loading}>
              {topics.map(topic => (<option key={topic.id} value={topic.id}>{topic.title}</option>))}
            </select>
          </div>
          <div className="model-selector">
            <label htmlFor="model-select">AI Model:</label>
            <select id="model-select" value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} className="chart-dropdown" disabled={loading}>
              <option value="gemini-pro">Google Gemini 2.5 Pro</option>
              <option value="openai-gpt5-mini">OpenAI GPT-5 Mini</option>
            </select>
          </div>
          <button onClick={runNewAnalysis} disabled={loading || !selectedTopic || selectedTopic === 0}>
            {loading ? 'Processing...' : 'Analyze New Comments'}
          </button>
        </div>
      </header>
      
      <div className="status-bar">{status}</div>
      
      {loading && !results && <div className="loading-spinner">Loading Data...</div>}

      {results && processedData && (
        <main className="dashboard">
          <section className="grid-3-col">
            <MetricCard title="Total Comments Analyzed" value={processedData.totalComments.toLocaleString()} />
            <MetricCard title="Avg. Overall Sentiment" value={processedData.avgSentiment.toFixed(2)} />
            <div className="card pie-card">
              <Plot
                data={[{
                  values: Object.values(processedData.sentimentCounts),
                  labels: Object.keys(processedData.sentimentCounts),
                  type: 'pie',
                  hole: .4,
                  marker: { colors: Object.keys(processedData.sentimentCounts).map(label => ({ 'Positive': '#2ca02c', 'Negative': '#d62728', 'Neutral': '#ff7f0e' }[label])) }
                }]}
                layout={{ title: 'Overall Sentiment', paper_bgcolor: 'transparent', font: { color: 'white' }, showlegend: true, margin: { t: 40, b: 40, l: 40, r: 40 } }}
                style={{ width: '100%', height: '100%' }}
                useResizeHandler={true} config={{ responsive: true }}
              />
            </div>
          </section>
          
          <section className="grid-full-width">
            <div className="card">
              <div className="chart-header">
                <h2>⏳ Comment Volume (Analyzed)</h2>
                <select onChange={(e) => setTimeSeriesFreq(e.target.value)} value={timeSeriesFreq} className="chart-dropdown">
                  <option value="D">Daily</option>
                  <option value="W">Weekly</option>
                  <option value="M">Monthly</option>
                </select>
              </div>
              <Plot
                data={[{ x: results.timeSeries.map(d => d.date), y: results.timeSeries.map(d => d.count), type: 'scatter', mode: 'lines+markers', marker: { color: '#17A2B8' } }]}
                layout={{ paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: 'white' }, yaxis: { title: 'Comment Count', gridcolor: '#444' }, xaxis: { title: 'Date', gridcolor: '#444' } }}
                style={{ width: '100%', height: '400px' }}
                useResizeHandler={true} config={{ responsive: true }}
              />
            </div>
          </section>
          
          <section className="grid-2-col">
            <div className="card">
              <h2>Top 10 Mentioned Leaders</h2>
              <EntityBarChart title="" data={processedData.topLeaders.slice(0, 10)} entityType="leader" />
            </div>
            <div className="card">
              <h2>Top 10 Mentioned Parties</h2>
              <EntityBarChart title="" data={processedData.topParties.slice(0, 10)} entityType="party" />
            </div>
          </section>

          <section className="grid-full-width">
            <div className="card">
              <h2>Avg. Sentiment by Language Used</h2>
              <Plot
                data={[{
                  x: processedData.avgSentimentByLang.map(d => d.language),
                  y: processedData.avgSentimentByLang.map(d => d.avg_sentiment),
                  type: 'bar',
                  text: processedData.avgSentimentByLang.map(d => `Count: ${d.count.toLocaleString()}`),
                  hoverinfo: 'x+y+text',
                  marker: {
                    color: processedData.avgSentimentByLang.map(d => d.avg_sentiment),
                    colorscale: 'RdYlGn',
                    reversescale: true,
                    cmin: -1,
                    cmax: 1
                  }
                }]}
                layout={{ paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: 'white' }, yaxis: { title: 'Average Sentiment', range: [-1, 1], gridcolor: '#444' }, xaxis: { gridcolor: '#444', title: 'Language' } }}
                style={{ width: '100%', height: '400px' }}
                useResizeHandler={true} config={{ responsive: true }}
              />
            </div>
          </section>
        </main>
      )}
    </div>
  );
}