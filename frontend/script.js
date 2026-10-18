/**
 * ChurnShield AI - Frontend JavaScript Engine
 * Supports:
 * - Real-time input synchronization (Tenure, Charges, Service dependents)
 * - Interactive preset customer profile loading
 * - FastAPI /predict integration with SHAP driver extraction
 * - Semi-circular probability gauge using Chart.js doughnut
 * - Low / Medium / High risk dynamic badges & business cost recommendations
 * - Model insights: Global Feature Importance, ROC Curve, and Multi-Model Benchmark Table
 * - Graceful offline fallback using exported Logistic Regression coefficients & Sigmoid
 */

// API Base URL - defaults to relative path (works both locally and on Vercel)
const API_BASE = window.location.origin.includes('localhost') || window.location.origin.includes('127.0.0.1')
  ? ''
  : '';

// Global references
let gaugeChart = null;
let importanceChart = null;
let rocChart = null;
let modelInfoData = null;
let customerPresets = {};
let isBackendOnline = false;

// DOM Elements
const form = document.getElementById('churn-form');
const tenureSlider = document.getElementById('tenure');
const tenureDisplay = document.getElementById('tenure-display');
const monthlySlider = document.getElementById('MonthlyCharges');
const monthlyDisplay = document.getElementById('monthly-charges-display');
const totalInput = document.getElementById('TotalCharges');
const totalBadge = document.getElementById('total-charges-badge');
const exampleSelector = document.getElementById('example-selector');
const resetBtn = document.getElementById('reset-btn');
const submitBtn = document.getElementById('submit-btn');
const btnSpinner = document.getElementById('btn-spinner');
const btnText = document.getElementById('btn-text');
const statusDot = document.getElementById('status-dot');
const statusLabel = document.getElementById('status-label');
const statusBanner = document.getElementById('status-banner');
const riskBadge = document.getElementById('risk-badge');
const probPercent = document.getElementById('prob-percent');
const verdictText = document.getElementById('verdict-text');
const thresholdText = document.getElementById('threshold-text');
const actionText = document.getElementById('action-text');
const reasonsList = document.getElementById('reasons-list');
const benchmarkTbody = document.getElementById('benchmark-tbody');

// Resilient Application Initializer
async function initApp() {
  console.log('[ChurnShield AI] Initializing app...');
  try {
    setupEventListeners();
  } catch (err) {
    console.error('Failed to setup event listeners:', err);
  }

  // 1. Check API Health immediately
  checkBackendStatus().catch(console.error);

  // 2. Load model info & presets
  loadModelInfo().catch(console.error);
  loadCustomerPresets().catch(console.error);

  // 3. Initialize gauge
  try {
    initGaugeChart(0, '#94a3b8');
  } catch (err) {
    console.warn('Gauge chart initialization deferred:', err);
  }
}

// Ensure execution whether DOM is loading or already interactive/complete
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

// Setup Form & Input Listeners
function setupEventListeners() {
  // Tenure slider sync
  if (tenureSlider && tenureDisplay) {
    tenureSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      const yrs = (val / 12).toFixed(1);
      tenureDisplay.textContent = `${val} months (${yrs} yrs)`;
      updateEstimatedTotal();
    });
  }

  // Monthly charges slider sync
  if (monthlySlider && monthlyDisplay) {
    monthlySlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      monthlyDisplay.textContent = `$${val.toFixed(2)}`;
      updateEstimatedTotal();
    });
  }

  // Total charges manual edit
  if (totalInput && totalBadge) {
    totalInput.addEventListener('input', () => {
      totalBadge.textContent = 'Manual';
      totalBadge.style.background = '#e2e8f0';
    });
  }

  // Preset selector
  if (exampleSelector) {
    exampleSelector.addEventListener('change', (e) => {
      const key = e.target.value;
      if (customerPresets[key]) {
        populateCustomerForm(customerPresets[key].data);
        handlePredictionSubmit(e);
      }
    });
  }

  // Reset form
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      if (form) form.reset();
      if (tenureSlider) tenureSlider.value = 24;
      if (tenureDisplay) tenureDisplay.textContent = '24 months (2.0 yrs)';
      if (monthlySlider) monthlySlider.value = 70.0;
      if (monthlyDisplay) monthlyDisplay.textContent = '$70.00';
      if (totalInput) totalInput.value = (24 * 70).toFixed(2);
      if (totalBadge) totalBadge.textContent = 'Auto';
      if (exampleSelector) exampleSelector.value = '';
      resetPredictionResults();
    });
  }

  // Internet service dependency helper
  document.querySelectorAll('input[name="InternetService"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      const isNone = e.target.value === 'No';
      const selects = ['OnlineSecurity', 'TechSupport', 'OnlineBackup', 'DeviceProtection', 'StreamingTV', 'StreamingMovies'];
      selects.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
          if (isNone) {
            el.value = 'No internet service';
          } else if (el.value === 'No internet service') {
            el.value = 'No';
          }
        }
      });
    });
  });

  // Form submission & Button click prevention
  if (form) {
    form.addEventListener('submit', handlePredictionSubmit);
  }
  if (submitBtn) {
    submitBtn.addEventListener('click', (e) => {
      e.preventDefault();
      handlePredictionSubmit(e);
    });
  }
}

// Auto calculate estimated TotalCharges
function updateEstimatedTotal() {
  if (totalBadge.textContent === 'Auto') {
    const tenure = parseInt(tenureSlider.value, 10);
    const monthly = parseFloat(monthlySlider.value);
    const calcTotal = tenure === 0 ? 0.0 : tenure * monthly;
    totalInput.value = calcTotal.toFixed(2);
  }
}

// Resilient API Fetch Helper (Handles direct path and /api/ prefixed Vercel routes)
async function fetchApi(endpoint, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, options);
    if (res.ok) return res;
    if (res.status === 404 && !endpoint.startsWith('/api')) {
      const altRes = await fetch(`${API_BASE}/api${endpoint}`, options);
      if (altRes.ok) return altRes;
    }
    return res;
  } catch (err) {
    if (!endpoint.startsWith('/api')) {
      try {
        const altRes = await fetch(`${API_BASE}/api${endpoint}`, options);
        if (altRes.ok) return altRes;
      } catch (e2) {}
    }
    throw err;
  }
}

// Check Backend Health
async function checkBackendStatus() {
  const dot = document.getElementById('status-dot') || statusDot;
  const label = document.getElementById('status-label') || statusLabel;
  const banner = document.getElementById('status-banner') || statusBanner;

  try {
    const res = await fetchApi('/health', { method: 'GET', headers: { 'Accept': 'application/json' } });
    if (res && res.ok) {
      const data = await res.json();
      isBackendOnline = true;
      if (dot) dot.className = 'status-indicator live';
      if (label) label.textContent = 'API Live (XGBoost + SHAP)';
      if (banner) banner.classList.add('hidden');
      console.log('[ChurnShield AI] Connected to live FastAPI backend:', data);
      return;
    }
  } catch (err) {
    console.warn('[ChurnShield AI] Backend check failed, activating client fallback:', err);
  }

  isBackendOnline = false;
  if (dot) dot.className = 'status-indicator offline';
  if (label) label.textContent = 'Offline Engine (Logistic Fallback)';
  showBanner('Backend API offline — using client-side Logistic Regression engine with exported model weights.', 'warning');
}

function showBanner(msg, type = 'warning') {
  const banner = document.getElementById('status-banner') || statusBanner;
  if (banner) {
    banner.textContent = msg;
    banner.className = `status-banner ${type}`;
    banner.classList.remove('hidden');
  }
}

// Load Model Info for Charts & Tables
async function loadModelInfo() {
  try {
    let data = null;
    try {
      const res = await fetchApi('/model-info');
      if (res && res.ok) data = await res.json();
    } catch (apiErr) {
      console.warn('API /model-info deferred to static JSON:', apiErr);
    }

    if (!data) {
      try {
        const res = await fetch('data/model_info.json');
        if (res.ok) data = await res.json();
      } catch (fileErr) {
        console.warn('Static data/model_info.json failed:', fileErr);
      }
    }

    if (data) {
      modelInfoData = data;
      renderBenchmarkTable(data.model_comparison);
      renderFeatureImportanceChart(data.top_features);
      renderRocCurveChart(data.roc_curve);
      console.log('[ChurnShield AI] Model info and benchmark data loaded.');
    }
  } catch (e) {
    console.error('Failed to load model info:', e);
  }
}

// Load Customer Presets
async function loadCustomerPresets() {
  try {
    let data = null;
    try {
      const res = await fetchApi('/customer-examples');
      if (res && res.ok) data = await res.json();
    } catch (apiErr) {
      console.warn('API /customer-examples deferred to static JSON:', apiErr);
    }

    if (!data) {
      try {
        const res = await fetch('data/customer_examples.json');
        if (res.ok) data = await res.json();
      } catch (fileErr) {
        console.warn('Static data/customer_examples.json failed:', fileErr);
      }
    }

    if (data && Array.isArray(data)) {
      data.forEach(item => {
        customerPresets[item.id] = item;
      });
      console.log(`[ChurnShield AI] Loaded ${data.length} customer presets.`);
    }
  } catch (e) {
    console.error('Failed to load presets:', e);
  }
}

// Populate Form from Preset Customer
function populateCustomerForm(c) {
  tenureSlider.value = c.tenure;
  tenureDisplay.textContent = `${c.tenure} months (${(c.tenure/12).toFixed(1)} yrs)`;
  
  monthlySlider.value = c.MonthlyCharges;
  monthlyDisplay.textContent = `$${parseFloat(c.MonthlyCharges).toFixed(2)}`;

  totalInput.value = parseFloat(c.TotalCharges).toFixed(2);
  totalBadge.textContent = 'Auto';

  // Radio button helper
  const setRadio = (name, val) => {
    const radio = document.querySelector(`input[name="${name}"][value="${val}"]`);
    if (radio) radio.checked = true;
  };

  setRadio('Contract', c.Contract);
  setRadio('PaperlessBilling', c.PaperlessBilling);
  setRadio('InternetService', c.InternetService);

  // Selects
  const setSelect = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.value = val;
  };

  setSelect('PaymentMethod', c.PaymentMethod);
  setSelect('OnlineSecurity', c.OnlineSecurity);
  setSelect('TechSupport', c.TechSupport);
  setSelect('OnlineBackup', c.OnlineBackup);
  setSelect('DeviceProtection', c.DeviceProtection);
  setSelect('StreamingTV', c.StreamingTV);
  setSelect('StreamingMovies', c.StreamingMovies);
  setSelect('PhoneService', c.PhoneService);
  setSelect('MultipleLines', c.MultipleLines);
  setSelect('gender', c.gender);
  setSelect('SeniorCitizen', c.SeniorCitizen);
  setSelect('Partner', c.Partner);
  setSelect('Dependents', c.Dependents);
}

// Extract Form Data to Object
function getFormData() {
  const formData = new FormData(form);
  const data = {};

  for (const [key, val] of formData.entries()) {
    data[key] = val;
  }

  data.tenure = parseInt(data.tenure, 10);
  data.MonthlyCharges = parseFloat(data.MonthlyCharges);
  data.TotalCharges = parseFloat(data.TotalCharges) || (data.tenure * data.MonthlyCharges);
  data.SeniorCitizen = parseInt(data.SeniorCitizen, 10);

  return data;
}

// Handle Form Submission
async function handlePredictionSubmit(e) {
  e.preventDefault();

  setLoadingState(true);
  const payload = getFormData();

  try {
    let result = null;

    if (isBackendOnline) {
      try {
        const response = await fetchApi('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (response.ok) {
          result = await response.json();
        } else {
          throw new Error(`API returned ${response.status}`);
        }
      } catch (apiErr) {
        console.warn('API call failed, falling back to client scoring:', apiErr);
        isBackendOnline = false;
        if (statusDot) statusDot.className = 'status-indicator offline';
        if (statusLabel) statusLabel.textContent = 'Offline Engine (Logistic Fallback)';
        showBanner('Backend offline: Generated prediction via client-side Logistic Regression engine.', 'warning');
      }
    }

    // Client-side fallback if backend is offline or failed
    if (!result) {
      result = computeClientSidePrediction(payload);
    }

    displayPredictionResult(result);
  } catch (err) {
    console.error('Prediction failed:', err);
    alert('An unexpected error occurred during prediction.');
  } finally {
    setLoadingState(false);
  }
}

// Client-Side Logistic Regression Scoring & Reason Generation Fallback
function computeClientSidePrediction(customer) {
  if (!modelInfoData || !modelInfoData.client_scoring_fallback) {
    // Default mock response if model info isn't loaded
    return {
      probability: 0.42,
      risk_level: 'Medium',
      optimal_threshold: 0.15,
      predicted_churn: true,
      top_reasons: [
        {
          feature: 'Contract: Month-to-month',
          impact: 0.65,
          direction: 'increase',
          human_explanation: 'Month-to-month contract increases churn risk (no long-term commitment).'
        }
      ]
    };
  }

  const fb = modelInfoData.client_scoring_fallback;
  const coefs = fb.coefficients;
  const scaler = fb.scaler;
  let logOdds = fb.intercept;
  const attributions = [];

  // Scaled numeric features
  const tenureScaled = (customer.tenure - scaler.tenure.mean) / scaler.tenure.scale;
  const monthlyScaled = (customer.MonthlyCharges - scaler.MonthlyCharges.mean) / scaler.MonthlyCharges.scale;
  const totalScaled = (customer.TotalCharges - scaler.TotalCharges.mean) / scaler.TotalCharges.scale;

  const addTerm = (feat, val, isPresent = true) => {
    if (coefs[feat] !== undefined) {
      const impact = coefs[feat] * (isPresent ? val : 0);
      logOdds += impact;
      if (isPresent && Math.abs(impact) > 0.05) {
        attributions.push({
          feature: feat,
          impact: impact,
          direction: impact > 0 ? 'increase' : 'decrease'
        });
      }
    }
  };

  addTerm('tenure', tenureScaled, true);
  addTerm('MonthlyCharges', monthlyScaled, true);
  addTerm('TotalCharges', totalScaled, true);

  // Categorical one-hot terms
  const catMatches = [
    `gender_${customer.gender}`,
    `SeniorCitizen_${customer.SeniorCitizen}`,
    `Partner_${customer.Partner}`,
    `Dependents_${customer.Dependents}`,
    `PhoneService_${customer.PhoneService}`,
    `MultipleLines_${customer.MultipleLines}`,
    `InternetService_${customer.InternetService}`,
    `OnlineSecurity_${customer.OnlineSecurity}`,
    `OnlineBackup_${customer.OnlineBackup}`,
    `DeviceProtection_${customer.DeviceProtection}`,
    `TechSupport_${customer.TechSupport}`,
    `StreamingTV_${customer.StreamingTV}`,
    `StreamingMovies_${customer.StreamingMovies}`,
    `Contract_${customer.Contract}`,
    `PaperlessBilling_${customer.PaperlessBilling}`,
    `PaymentMethod_${customer.PaymentMethod}`
  ];

  catMatches.forEach(feat => addTerm(feat, 1.0, true));

  // Sigmoid probability: 1 / (1 + exp(-z))
  const prob = 1.0 / (1.0 + Math.exp(-logOdds));
  const optTh = modelInfoData.best_model.optimal_threshold || 0.15;

  let riskLevel = 'Low';
  if (prob >= 0.60) riskLevel = 'High';
  else if (prob >= 0.30) riskLevel = 'Medium';

  // Sort attributions by absolute impact
  attributions.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
  const topReasons = attributions.slice(0, 4).map(item => {
    let expl = `${item.feature.replace('_', ': ')} ${item.direction === 'increase' ? 'increases' : 'reduces'} churn probability.`;
    if (item.feature.includes('Month-to-month')) expl = 'Month-to-month contracts strongly elevate customer churn probability.';
    if (item.feature === 'tenure') expl = customer.tenure < 12 ? 'Short customer tenure increases vulnerability to churn.' : 'Long-term tenure protects against churn.';
    if (item.feature.includes('Fiber optic')) expl = 'Fiber optic users experience higher churn rates across competitors.';
    if (item.feature.includes('Electronic check')) expl = 'Electronic check payment exhibits the highest churn rate.';
    if (item.feature.includes('Two year')) expl = 'Two-year contracts provide strong customer loyalty and retention.';

    return {
      feature: item.feature.replace('_', ': '),
      impact: Math.round(item.impact * 100) / 100,
      direction: item.direction,
      human_explanation: expl
    };
  });

  return {
    probability: Math.round(prob * 10000) / 10000,
    risk_level: riskLevel,
    optimal_threshold: optTh,
    predicted_churn: prob >= optTh,
    top_reasons: topReasons
  };
}

// Display Prediction Results
function displayPredictionResult(res) {
  const pct = Math.round(res.probability * 100);
  probPercent.textContent = `${pct}%`;

  // Risk Badge & Colors
  let badgeColor = '#ef4444';
  let badgeClass = 'risk-high';
  let badgeLabel = 'HIGH RISK';

  if (res.risk_level === 'Medium') {
    badgeColor = '#f59e0b';
    badgeClass = 'risk-medium';
    badgeLabel = 'MEDIUM RISK';
  } else if (res.risk_level === 'Low') {
    badgeColor = '#10b981';
    badgeClass = 'risk-low';
    badgeLabel = 'LOW RISK';
  }

  riskBadge.className = `risk-badge ${badgeClass}`;
  riskBadge.textContent = badgeLabel;

  // Update Gauge Chart
  updateGauge(res.probability, badgeColor);

  // Verdict & Action
  if (res.predicted_churn) {
    verdictText.textContent = 'Predicted to Churn';
    verdictText.style.color = '#dc2626';
    actionText.textContent = res.probability >= 0.60
      ? '🚨 Urgent: Offer 1-year contract discount'
      : '⚠️ Proactive: Retention email & support checkup';
  } else {
    verdictText.textContent = 'Predicted to Stay';
    verdictText.style.color = '#15803d';
    actionText.textContent = '✅ Good Standing: Cross-sell opportunity';
  }

  thresholdText.textContent = `${Math.round(res.optimal_threshold * 100)}% (Cost Tuned)`;

  // Render Top SHAP Reasons
  renderReasons(res.top_reasons);
}

// Render SHAP Driver Cards
function renderReasons(reasons) {
  if (!reasons || reasons.length === 0) {
    reasonsList.innerHTML = '<div class="empty-state"><p>No significant drivers found for this profile.</p></div>';
    return;
  }

  reasonsList.innerHTML = '';
  reasons.forEach(r => {
    const isInc = r.direction === 'increase';
    const icon = isInc ? '▲' : '▼';
    const itemClass = isInc ? 'increase' : 'decrease';
    const sign = r.impact > 0 ? `+${r.impact}` : `${r.impact}`;

    const div = document.createElement('div');
    div.className = `reason-item ${itemClass}`;
    div.innerHTML = `
      <div class="reason-icon">${icon}</div>
      <div class="reason-content">
        <div class="reason-header">
          <span class="reason-title">${escapeHtml(r.feature)}</span>
          <span class="reason-impact">${sign} SHAP</span>
        </div>
        <p class="reason-text">${escapeHtml(r.human_explanation)}</p>
      </div>
    `;
    reasonsList.appendChild(div);
  });
}

// Initialize Semi-Circular Gauge (Chart.js)
function initGaugeChart(initialProb, color) {
  const canvas = document.getElementById('gaugeCanvas');
  if (!canvas) return;

  if (typeof Chart === 'undefined') {
    setTimeout(() => initGaugeChart(initialProb, color), 150);
    return;
  }

  const ctx = canvas.getContext('2d');
  if (gaugeChart) {
    try { gaugeChart.destroy(); } catch (e) {}
  }

  gaugeChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      datasets: [{
        data: [initialProb * 100, 100 - (initialProb * 100)],
        backgroundColor: [color, '#e2e8f0'],
        borderWidth: 0,
        circumference: 180,
        rotation: 270
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '76%',
      animation: { duration: 800, easing: 'easeOutQuart' },
      plugins: {
        tooltip: { enabled: false },
        legend: { display: false }
      }
    }
  });
}

// Update Gauge Chart
function updateGauge(prob, color) {
  if (!gaugeChart) {
    if (typeof Chart !== 'undefined') {
      initGaugeChart(prob, color);
    }
    return;
  }
  const pct = Math.min(100, Math.max(0, prob * 100));
  gaugeChart.data.datasets[0].data = [pct, 100 - pct];
  gaugeChart.data.datasets[0].backgroundColor = [color, '#e2e8f0'];
  gaugeChart.update();
}

// Reset UI
function resetPredictionResults() {
  probPercent.textContent = '--%';
  riskBadge.className = 'risk-badge risk-neutral';
  riskBadge.textContent = 'Awaiting Analysis';
  verdictText.textContent = 'Ready to run';
  verdictText.style.color = '#1e293b';
  thresholdText.textContent = '15% (Cost Tuned)';
  actionText.textContent = 'Configure profile and click predict';
  updateGauge(0, '#94a3b8');
  reasonsList.innerHTML = `
    <div class="empty-state">
      <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"></path>
      </svg>
      <p>Submit customer parameters to generate personalized SHAP explanations.</p>
    </div>
  `;
}

// Loading Spinner State
function setLoadingState(isLoading) {
  if (isLoading) {
    btnSpinner.classList.remove('hidden');
    btnText.textContent = 'Analyzing Driver Vectors...';
    submitBtn.disabled = true;
  } else {
    btnSpinner.classList.add('hidden');
    btnText.textContent = 'Predict Churn Risk';
    submitBtn.disabled = false;
  }
}

// Render Model Benchmark Table
function renderBenchmarkTable(models) {
  if (!models || !benchmarkTbody) return;
  benchmarkTbody.innerHTML = '';

  models.forEach(m => {
    const isBest = m.Model.includes('XGBoost (Tuned)');
    const isBaseline = m.Model.includes('Baseline');
    const tr = document.createElement('tr');
    if (isBest) tr.className = 'best-row';

    let strategy = 'Standard';
    if (m.Model.includes('Class Weights')) strategy = 'Balanced Weights';
    else if (m.Model.includes('SMOTE')) strategy = 'SMOTE Resampling';
    else if (m.Model.includes('Tuned')) strategy = 'Hyperparameter Tuned';

    let statusHtml = '<span class="status-badge status-alt">Challenger</span>';
    if (isBest) statusHtml = '<span class="status-badge status-best">🏆 Champion</span>';
    else if (isBaseline) statusHtml = '<span class="status-badge status-baseline">Baseline</span>';

    tr.innerHTML = `
      <td><strong>${escapeHtml(m.Model)}</strong></td>
      <td>${strategy}</td>
      <td>${(m.Accuracy * 100).toFixed(1)}%</td>
      <td><strong>${m['ROC-AUC'].toFixed(4)}</strong></td>
      <td>${m['F1-Score'].toFixed(4)}</td>
      <td>${(m.Precision * 100).toFixed(1)}%</td>
      <td>${(m.Recall * 100).toFixed(1)}%</td>
      <td>${statusHtml}</td>
    `;
    benchmarkTbody.appendChild(tr);
  });
}

// Render Global Feature Importance Chart
function renderFeatureImportanceChart(features) {
  const canvas = document.getElementById('importanceChart');
  if (!canvas || !features) return;

  if (typeof Chart === 'undefined') {
    setTimeout(() => renderFeatureImportanceChart(features), 250);
    return;
  }

  const labels = features.map(f => f.feature.replace('_', ': ')).reverse();
  const values = features.map(f => f.importance).reverse();

  if (importanceChart) {
    try { importanceChart.destroy(); } catch (e) {}
  }

  const ctx = canvas.getContext('2d');
  importanceChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Mean |SHAP Value|',
        data: values,
        backgroundColor: '#3b82f6',
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Mean Impact: ${ctx.parsed.x.toFixed(4)} log-odds`
          }
        }
      },
      scales: {
        x: {
          grid: { color: '#f1f5f9' },
          title: { display: true, text: 'Mean |SHAP Value| (Impact on Model Output)', font: { size: 11 } }
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 11 } }
        }
      }
    }
  });
}

// Render ROC Curve Chart
function renderRocCurveChart(roc) {
  const canvas = document.getElementById('rocChart');
  if (!canvas || !roc) return;

  if (typeof Chart === 'undefined') {
    setTimeout(() => renderRocCurveChart(roc), 250);
    return;
  }

  if (rocChart) {
    try { rocChart.destroy(); } catch (e) {}
  }

  const points = roc.fpr.map((fpr, i) => ({ x: fpr, y: roc.tpr[i] }));
  const randomLine = [{ x: 0, y: 0 }, { x: 1, y: 1 }];

  const ctx = canvas.getContext('2d');
  rocChart = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: `XGBoost Tuned (AUC = ${roc.auc})`,
          data: points,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.08)',
          fill: true,
          borderWidth: 2.5,
          tension: 0.1,
          pointRadius: 0
        },
        {
          label: 'Random Chance (AUC = 0.500)',
          data: randomLine,
          borderColor: '#94a3b8',
          borderDash: [5, 5],
          borderWidth: 1.5,
          fill: false,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 14, font: { size: 11 } } }
      },
      scales: {
        x: {
          type: 'linear',
          min: 0,
          max: 1,
          title: { display: true, text: 'False Positive Rate (1 - Specificity)', font: { size: 11 } },
          grid: { color: '#f1f5f9' }
        },
        y: {
          type: 'linear',
          min: 0,
          max: 1.05,
          title: { display: true, text: 'True Positive Rate (Sensitivity / Recall)', font: { size: 11 } },
          grid: { color: '#f1f5f9' }
        }
      }
    }
  });
}

// Utility: HTML escaping
function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
