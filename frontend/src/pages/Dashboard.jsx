
import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Sun, Moon, Volume2, Globe, Leaf, CloudRain, Activity, Droplets } from 'lucide-react';
import { motion } from 'framer-motion';
import { translations } from '../translations';
import '../styles/main.css';

const Dashboard = ({ theme, toggleTheme }) => {
  const [lang, setLang] = useState('en');
  const t = translations[lang] || translations['en'];

  const [formData, setFormData] = useState({
    crop: 'Rice',
    soil: 'Clay',
    temp: '',
    rain: '',
    ndvi: '',
    nitrogen: 60,
    ph: 6.5
  });

  // Dashboard Component Logic
  const [prediction, setPrediction] = useState(null);
  const [advisoryCodes, setAdvisoryCodes] = useState([]); // Store backend codes
  const [loading, setLoading] = useState(false);

  // Helper to get translated advisory text
  const getAdvisoryText = () => {
    if (!advisoryCodes || advisoryCodes.length === 0) return t.adv_normal;
    return advisoryCodes.map(code => t[code] || code).join(" ");
  };

  // Dynamic Chart Data State
  const [chartData, setChartData] = useState([
    { name: '2020', yield: 2400 },
    { name: '2021', yield: 2800 },
    { name: '2022', yield: 2600 },
    { name: '2023', yield: 3000 },
    { name: '2024', yield: 3200 },
  ]);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch('http://localhost:8000/predict/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });
      if (!response.ok) throw new Error('Prediction failed');

      const result = await response.json();
      const predVal = parseFloat(result.prediction);
      setPrediction(predVal);
      setAdvisoryCodes(result.advisory_codes || []); // Store codes

      // Update Chart Logic
      const baseData = [
        { name: '2020', yield: predVal * 0.8 },
        { name: '2021', yield: predVal * 0.7 },
        { name: '2022', yield: predVal * 0.9 },
        { name: '2023', yield: predVal * 0.85 },
        { name: '2024', yield: predVal * 0.95 },
      ];
      setChartData([...baseData, { name: '2025', yield: predVal }]);

      // Reset Form after success
      setFormData({
        crop: 'Rice',
        soil: 'Clay',
        temp: '',
        rain: '',
        ndvi: '',
        nitrogen: 60,
        ph: 6.5
      });

    } catch (err) {
      console.error(err);
      // alert('Failed to connect to AI server.');
    } finally {
      setLoading(false);
    }
  };

  const speakResult = () => {
    if (!prediction) return;
    // Speak both result and advisory
    const advisoryText = getAdvisoryText();
    const text = `${t.speechStart} ${t.resultTitle} is ${parseFloat(prediction).toFixed(2)} ${t.unit}. ${advisoryText}`;
    const utterance = new SpeechSynthesisUtterance(text);
    const langMap = { 'hi': 'hi-IN', 'es': 'es-ES', 'te': 'te-IN', 'ta': 'ta-IN', 'en': 'en-US' };
    utterance.lang = langMap[lang] || 'en-US';
    window.speechSynthesis.speak(utterance);
  };

  return (
    <div className={`dashboard-wrapper ${theme}`}>
      {/* Top Bar */}
      <nav className="navbar">
        <div className="navbar-brand">
          <Leaf className="w-8 h-8 text-primary" />
          <span>{t.appTitle}</span>
        </div>

        <div className="nav-actions" style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          {/* Language Selector */}
          <div className="lang-select" style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
            <Globe size={18} style={{ marginRight: '0.5rem', color: 'var(--text-muted)' }} />
            <select
              value={lang}
              onChange={(e) => setLang(e.target.value)}
              style={{ padding: '0.5rem', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}
            >
              <option value="en">English</option>
              <option value="hi">हिंदी (Hindi)</option>
              <option value="te">తెలుగు (Telugu)</option>
              <option value="ta">தமிழ் (Tamil)</option>
            </select>
          </div>

          {/* Theme Toggle */}
          <button onClick={toggleTheme} className="btn-icon">
            {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
          </button>
        </div>
      </nav>

      <div className="main-content">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid-layout"
        >
          {/* Input Panel */}
          <div className="card input-section">
            <h2 className="card-title">{t.predict}</h2>
            <form onSubmit={handlePredict}>
              {/* Crop & Soil Selection */}
              <div className="form-group-row" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div className="form-group">
                  <label>{t.crop || "Crop Type"}</label>
                  <select
                    name="crop"
                    value={formData.crop}
                    onChange={handleChange}
                    style={{ width: '100%', padding: '0.75rem', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}
                  >
                    <option value="Rice">{t.crops?.rice || "Rice"}</option>
                    <option value="Maize">{t.crops?.maize || "Maize"}</option>
                    <option value="Wheat">{t.crops?.wheat || "Wheat"}</option>
                    <option value="Cotton">{t.crops?.cotton || "Cotton"}</option>
                  </select>
                </div>
                <div className="form-group">
                  <label>{t.soil || "Soil Type"}</label>
                  <select
                    name="soil"
                    value={formData.soil}
                    onChange={handleChange}
                    style={{ width: '100%', padding: '0.75rem', borderRadius: 'var(--radius)', border: '1px solid var(--border)' }}
                  >
                    <option value="Clay">{t.soils?.clay || "Clay"}</option>
                    <option value="Sandy">{t.soils?.sandy || "Sandy"}</option>
                    <option value="Loam">{t.soils?.loam || "Loam"}</option>
                    <option value="Chalky">{t.soils?.chalky || "Chalky"}</option>
                  </select>
                </div>
              </div>

              {[
                { name: 'temp', label: t.temp, icon: <Sun size={18} /> },
                { name: 'rain', label: t.rain, icon: <CloudRain size={18} /> },
                { name: 'ndvi', label: t.ndvi, icon: <Activity size={18} /> },
                { name: 'nitrogen', label: t.nitrogen, icon: <Droplets size={18} /> },
                { name: 'ph', label: t.ph, icon: <Leaf size={18} /> },
              ].map((field) => (
                <div key={field.name} className="form-group">
                  <label>{field.label}</label>
                  <div className="input-icon-wrapper">
                    <span className="icon">{field.icon}</span>
                    <input
                      type="number"
                      step="any"
                      name={field.name}
                      value={formData[field.name]}
                      onChange={handleChange}
                      required={['temp', 'rain', 'ndvi'].includes(field.name)}
                    />
                  </div>
                </div>
              ))}

              <button type="submit" className="btn btn-primary full-width" disabled={loading}>
                {loading ? t.analyzing : t.analyze}
              </button>
            </form>
          </div>

          {/* Result & Chart Panel */}
          <div className="visualization-section">
            {(prediction !== null && prediction !== undefined) && (
              <motion.div
                initial={{ scale: 0.9 }}
                animate={{ scale: 1 }}
                className="card result-card"
              >
                <div className="result-header">
                  <h3>{t.resultTitle}</h3>
                  <button onClick={speakResult} className="btn-icon" title="Read Aloud">
                    <Volume2 size={24} />
                  </button>
                </div>
                <div className="result-value">
                  {parseFloat(prediction).toFixed(2)} <span>{t.unit}</span>
                </div>

                <div className="advisory-box">
                  <strong>{t.advisoryTitle}:</strong>
                  <p>{getAdvisoryText()}</p>
                </div>
              </motion.div>
            )}

            <div className="card chart-card">
              <h3>{t.chartTitle}</h3>
              <div style={{ height: '300px', width: '100%' }}>
                <ResponsiveContainer>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="name" stroke="var(--text-muted)" />
                    <YAxis stroke="var(--text-muted)" />
                    <Tooltip contentStyle={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }} />
                    <Line type="monotone" dataKey="yield" name={t.yieldLabel || "Yield"} stroke="var(--primary)" strokeWidth={3} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;
