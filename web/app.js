/* Fraud Detection Frontend - Application Logic */

const API_BASE = window.location.origin;
const STATUS_DOT = document.getElementById("statusDot");
const STATUS_TEXT = document.getElementById("statusText");
const THRESHOLD_INPUT = document.getElementById("threshold");
const THRESHOLD_VAL = document.getElementById("thresholdVal");
const TXN_FORM = document.getElementById("txnForm");
const PREDICT_BTN = document.getElementById("predictBtn");
const RESULT_CARD = document.getElementById("resultCard");
const V_FEATURES = document.getElementById("vFeatures");

// Realistic samples from the credit card fraud dataset.
const SAMPLE_LEGIT = {
  Time: 69808, Amount: 540,
  V1: 0.1362, V2: -1.6478, V3: 0.8500, V4: 1.5762, V5: -1.5691,
  V6: -0.1068, V7: 0.0266, V8: -0.0917, V9: 0.5176, V10: -0.3330,
  V11: -0.3312, V12: 0.6520, V13: 1.1614, V14: -0.2107, V15: 1.5555,
  V16: 0.7935, V17: -0.7988, V18: 0.6076, V19: -0.9650, V20: 1.0313,
  V21: 0.5846, V22: 0.5824, V23: -0.5485, V24: 0.4567, V25: 0.2023,
  V26: -0.3143, V27: -0.0182, V28: 0.1381,
};
const SAMPLE_FRAUD = {
  Time: 129095, Amount: 720.8,
  V1: -1.8369, V2: -1.6467, V3: -3.3812, V4: 0.4734, V5: 0.0742,
  V6: -0.4468, V7: 3.7919, V8: -1.3510, V9: 0.0952, V10: -0.0845,
  V11: -1.5423, V12: -0.5405, V13: -0.2721, V14: 0.7125, V15: 0.2350,
  V16: -0.9077, V17: -0.3911, V18: -0.0347, V19: 1.0843, V20: -1.5861,
  V21: 0.0107, V22: 1.7867, V23: -0.1512, V24: -0.5821, V25: -0.9561,
  V26: -0.3344, V27: 0.7156, V28: 0.3704,
};

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
async function init() {
  await checkHealth();
  buildVFeatures();
  THRESHOLD_INPUT.addEventListener("input", (e) => {
    THRESHOLD_VAL.textContent = parseFloat(e.target.value).toFixed(2);
  });
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    STATUS_DOT.className = data.model_loaded ? "dot online" : "dot offline";
    STATUS_TEXT.textContent = data.model_loaded
      ? "API online — model loaded"
      : "API online — model NOT loaded";
  } catch (err) {
    STATUS_DOT.className = "dot offline";
    STATUS_TEXT.textContent = "API unreachable — is the server running?";
  }
}

// Build V1..V28 input fields.
function buildVFeatures() {
  V_FEATURES.innerHTML = "";
  for (let i = 1; i <= 28; i++) {
    const field = document.createElement("div");
    field.className = "field";
    field.innerHTML = `
      <label>V${i}</label>
      <input type="number" name="V${i}" step="any" value="0" />
    `;
    V_FEATURES.appendChild(field);
  }
}

// ---------------------------------------------------------------------------
// Sample presets / reset
// ---------------------------------------------------------------------------
window.loadSample = function (kind) {
  const sample = kind === "fraud" ? SAMPLE_FRAUD : SAMPLE_LEGIT;
  document.getElementById("Time").value = sample.Time;
  document.getElementById("Amount").value = sample.Amount;
  for (let i = 1; i <= 28; i++) {
    const el = document.querySelector(`input[name="V${i}"]`);
    if (el) el.value = sample[`V${i}`];
  }
};

window.resetForm = function () {
  TXN_FORM.reset();
  RESULT_CARD.style.display = "none";
};

// ---------------------------------------------------------------------------
// Single transaction prediction
// ---------------------------------------------------------------------------
window.predictSingle = async function () {
  const formData = new FormData(TXN_FORM);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = value === "" ? 0 : parseFloat(value);
  }
  const threshold = parseFloat(THRESHOLD_INPUT.value);

  SET_LOADING(PREDICT_BTN, true);
  RESULT_CARD.style.display = "none";

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, threshold }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Prediction failed (${res.status})`);
    }
    const result = await res.json();
    RENDER_RESULT(result);
  } catch (err) {
    alert("Prediction error: " + err.message);
    console.error(err);
  } finally {
    SET_LOADING(PREDICT_BTN, false);
  }
};

function SET_LOADING(btn, loading) {
  if (loading) {
    btn.disabled = true;
    btn.textContent = "Predicting…";
  } else {
    btn.disabled = false;
    btn.textContent = "Predict Fraud";
  }
}

function RENDER_RESULT(r) {
  const isFraud = r.is_fraud;
  const pct = (r.fraud_probability * 100).toFixed(1);

  const gauge = document.getElementById("gauge");
  const needleAngle = 180 + r.fraud_probability * 180;
  gauge.style.setProperty("--angle", `${needleAngle}deg`);

  document.getElementById("gaugeValue").textContent = pct + "%";
  document.getElementById("gaugeLabel").textContent = isFraud ? "FRAUD RISK" : "LEGITIMATE";

  const badge = document.getElementById("verdictBadge");
  badge.textContent = isFraud ? "FRAUD DETECTED" : "LEGITIMATE";
  badge.className = "badge " + (isFraud ? "fraud" : "legit");

  document.getElementById("probability").textContent = pct + "% probability of fraud";
  document.getElementById("riskText").textContent = `Risk level: ${r.risk_level}`;
  document.getElementById("metaText").textContent =
    `Threshold: ${r.threshold} · Features used: ${r.features_used}`;

  const explanation = document.getElementById("explanation");
  if (isFraud) {
    explanation.innerHTML = `<strong>⚠️ This transaction is flagged as potentially fraudulent.</strong> Review it before authorizing.`;
    explanation.style.background = "#fef2f2";
    explanation.style.color = "#991b1b";
  } else {
    explanation.innerHTML = `<strong>✅ This transaction appears legitimate.</strong> No fraud indicators detected above the threshold.`;
    explanation.style.background = "#f0fdf4";
    explanation.style.color = "#166534";
  }

  RESULT_CARD.style.display = "block";
  RESULT_CARD.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------------------------------------------------------------------------
// Batch prediction
// ---------------------------------------------------------------------------
window.predictBatch = async function () {
  const fileInput = document.getElementById("batchFile");
  if (!fileInput.files || fileInput.files.length === 0) {
    alert("Please select a CSV file first.");
    return;
  }
  const file = fileInput.files[0];

  const formData = new FormData();
  formData.append("file", file);

  const summary = document.getElementById("batchSummary");
  const resultsWrap = document.getElementById("batchResults");
  const downloadLink = document.getElementById("downloadLink");
  summary.style.display = "none";
  resultsWrap.style.display = "none";
  downloadLink.style.display = "none";

  try {
    const res = await fetch(`${API_BASE}/predict/batch`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Batch prediction failed (${res.status})`);
    }
    const data = await res.json();

    summary.innerHTML = `
      <strong>${data.total_transactions}</strong> transactions scored ·
      <strong style="color:var(--danger)">${data.fraud_count} fraud</strong> ·
      <strong style="color:var(--success)">${data.legitimate_count} legitimate</strong> ·
      fraud rate <strong>${(data.fraud_rate * 100).toFixed(2)}%</strong> ·
      threshold <strong>${data.threshold}</strong>
    `;
    summary.style.display = "block";

    const table = document.createElement("table");
    table.innerHTML = `
      <thead><tr>
        <th>#</th><th>Fraud probability</th><th>Prediction</th><th>Risk level</th>
      </tr></thead>
      <tbody></tbody>
    `;
    const tbody = table.querySelector("tbody");
    let idx = 0;
    data.results.slice(0, 200).forEach((r) => {
      idx++;
      const isFraud = r.is_fraud;
      const tr = document.createElement("tr");
      if (isFraud) tr.className = "fraud-row";
      tr.innerHTML = `
        <td>${idx}</td>
        <td>${(r.fraud_probability * 100).toFixed(2)}%</td>
        <td><span class="pill ${isFraud ? "fraud" : "legit"}">${isFraud ? "FRAUD" : "LEGIT"}</span></td>
        <td>${r.risk_level}</td>
      `;
      tbody.appendChild(tr);
    });
    resultsWrap.innerHTML = "";
    resultsWrap.appendChild(table);
    if (data.results.length > 200) {
      const note = document.createElement("p");
      note.style.fontSize = "0.85rem"; note.style.color = "var(--muted)";
      note.textContent = `Showing first 200 of ${data.results.length} transactions.`;
      resultsWrap.appendChild(note);
    }
    resultsWrap.style.display = "block";

    if (data.download_url) {
      downloadLink.href = `${API_BASE}${data.download_url}`;
      downloadLink.style.display = "inline-block";
    }
  } catch (err) {
    alert("Batch prediction error: " + err.message);
    console.error(err);
  }
};

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
init();


