// ================= CONFIGURATION =================
// Replace with your raw GitHub CSV URL
// Format: https://raw.githubusercontent.com/ YOUR_USERNAME / YOUR_REPO / main/backend/Diabetes_prediction.csv
const CSV_URL = 'https://raw.githubusercontent.com/CodeWander-666/healthmasterAI/main/backend/Diabetes_prediction.csv';

// ================= GLOBAL VARIABLES =================
let dataset = [];
let featureStats = {};
let isDataLoaded = false;

const FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
];
const TARGET_NAME = 'Diagnosis';

// ================= DOM ELEMENTS =================
const form = document.getElementById('prediction-form');
const submitBtn = document.getElementById('submit-btn');
const resultDiv = document.getElementById('result');
const statusLed = document.getElementById('status-led');
const statusText = document.getElementById('status-text');
const checkConnBtn = document.getElementById('check-connection');
const diagnosticOutput = document.getElementById('diagnostic-output');

// ================= VALIDATION RULES =================
const validationRules = {
    pregnancies: { min: 0, max: 20, message: 'Must be between 0 and 20' },
    glucose: { min: 0, max: 300, message: 'Typical range 0–300 mg/dL' },
    bloodpressure: { min: 0, max: 200, message: '0–200 mm Hg' },
    skinthickness: { min: 0, max: 100, message: '0–100 mm' },
    insulin: { min: 0, max: 900, message: '0–900 μU/mL' },
    bmi: { min: 10, max: 70, message: 'BMI 10–70 kg/m²' },
    dpf: { min: 0, max: 3, message: '0–3 (unitless)' },
    age: { min: 0, max: 120, message: '0–120 years' }
};

// ================= INIT =================
document.addEventListener('DOMContentLoaded', () => {
    attachValidationListeners();
    updateDiagnosticInfo();
    loadCSV();
    checkConnBtn.addEventListener('click', () => {
        // Manual reload of CSV
        loadCSV();
    });
});

// ================= CSV LOADING =================
async function loadCSV() {
    statusLed.className = 'status-led checking';
    statusText.textContent = 'Loading dataset from GitHub...';

    try {
        const response = await fetch(CSV_URL);
        if (!response.ok) throw new Error(`HTTP ${response.status} – ${response.statusText}`);
        const csvText = await response.text();
        parseCSV(csvText);
        computeFeatureStats();
        isDataLoaded = true;
        statusLed.className = 'status-led online';
        statusText.textContent = `✅ Data loaded (${dataset.length} records)`;
        updateDiagnosticInfo(`Dataset loaded successfully from ${CSV_URL}`);
    } catch (error) {
        statusLed.className = 'status-led offline';
        statusText.textContent = '❌ Failed to load CSV';
        updateDiagnosticInfo(`CSV load error: ${error.message}`);
        console.error(error);
    }
}

function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',').map(h => h.trim());

    // Verify required columns
    const required = [...FEATURE_NAMES, TARGET_NAME];
    const missing = required.filter(h => !headers.includes(h));
    if (missing.length) throw new Error(`Missing columns: ${missing.join(', ')}`);

    dataset = [];
    for (let i = 1; i < lines.length; i++) {
        if (lines[i].trim() === '') continue;
        const values = lines[i].split(',').map(v => parseFloat(v.trim()));
        if (values.length !== headers.length) continue;

        const record = {};
        headers.forEach((h, idx) => { record[h] = values[idx]; });
        dataset.push(record);
    }
}

function computeFeatureStats() {
    featureStats = {};
    FEATURE_NAMES.forEach(f => {
        const values = dataset.map(row => row[f]).filter(v => !isNaN(v));
        featureStats[f] = {
            min: Math.min(...values),
            max: Math.max(...values)
        };
    });
}

// ================= SCALING & DISTANCE =================
function scaleValue(feature, value) {
    const { min, max } = featureStats[feature];
    if (max === min) return 0.5;
    return (value - min) / (max - min);
}

function distance(record1, record2) {
    let sum = 0;
    for (let f of FEATURE_NAMES) {
        const v1 = scaleValue(f, record1[f]);
        const v2 = scaleValue(f, record2[f]);
        sum += (v1 - v2) ** 2;
    }
    return Math.sqrt(sum);
}

// ================= KNN PREDICTION =================
function predictKNN(input, k = 5) {
    if (dataset.length === 0) return null;

    const distances = dataset.map(record => ({
        record,
        dist: distance(record, input)
    }));

    distances.sort((a, b) => a.dist - b.dist);
    const neighbors = distances.slice(0, k);

    const votes = neighbors.reduce((acc, curr) => {
        const diag = curr.record[TARGET_NAME];
        acc[diag] = (acc[diag] || 0) + 1;
        return acc;
    }, {});

    const prediction = (votes[1] || 0) > (votes[0] || 0) ? 1 : 0;
    const probability = (votes[1] || 0) / k;

    return { prediction, probability };
}

// ================= VALIDATION HELPERS =================
function attachValidationListeners() {
    Object.keys(validationRules).forEach(field => {
        const input = document.getElementById(field);
        if (input) {
            input.addEventListener('input', () => validateField(field));
            input.addEventListener('blur', () => validateField(field));
        }
    });
}

function validateField(field) {
    const input = document.getElementById(field);
    const errorSpan = document.getElementById(`error-${field}`);
    const rules = validationRules[field];
    const value = parseFloat(input.value);

    let isValid = true;
    let errorMsg = '';

    if (isNaN(value) || input.value.trim() === '') {
        isValid = false;
        errorMsg = 'This field is required';
    } else if (value < rules.min || value > rules.max) {
        isValid = false;
        errorMsg = rules.message;
    }

    if (!isValid) {
        input.classList.add('invalid');
        errorSpan.textContent = errorMsg;
    } else {
        input.classList.remove('invalid');
        errorSpan.textContent = '';
    }

    updateSubmitButton();
    return isValid;
}

function isFormValid() {
    return Object.keys(validationRules).every(field => {
        const input = document.getElementById(field);
        if (!input) return false;
        const value = parseFloat(input.value);
        const rules = validationRules[field];
        return !isNaN(value) && input.value.trim() !== '' && value >= rules.min && value <= rules.max;
    });
}

function updateSubmitButton() {
    submitBtn.disabled = !isFormValid() || !isDataLoaded;
}

// ================= FORM SUBMISSION =================
form.addEventListener('submit', (e) => {
    e.preventDefault();

    if (!isFormValid()) {
        alert('Please correct the errors in the form.');
        return;
    }
    if (!isDataLoaded) {
        alert('Dataset still loading. Please wait or click "Test Now".');
        return;
    }

    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing...';

    const inputRecord = {};
    FEATURE_NAMES.forEach(f => {
        const fieldId = f === 'DiabetesPedigreeFunction' ? 'dpf' : f.toLowerCase();
        inputRecord[f] = parseFloat(document.getElementById(fieldId).value);
    });

    try {
        const result = predictKNN(inputRecord, 5);
        if (!result) throw new Error('Prediction failed');
        displayResult(result);
        updateDiagnosticInfo('Prediction successful');
    } catch (error) {
        showError(error.message);
        updateDiagnosticInfo(`Prediction error: ${error.message}`);
    } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Analyze Risk';
    }
});

function displayResult({ prediction, probability }) {
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '';

    const badge = document.createElement('div');
    badge.className = `risk-badge ${prediction === 1 ? 'high-risk' : 'low-risk'}`;
    badge.textContent = prediction === 1
        ? `⚠️ High Risk (Probability: ${(probability * 100).toFixed(1)}%)`
        : `✅ Low Risk (Probability: ${(probability * 100).toFixed(1)}%)`;
    resultDiv.appendChild(badge);

    const note = document.createElement('p');
    note.style.textAlign = 'center';
    note.style.marginTop = '1rem';
    note.style.color = '#4a5568';
    note.textContent = 'Feature importance not available with kNN.';
    resultDiv.appendChild(note);
}

function showError(message) {
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `<div class="error-box">❌ Error: ${message}</div>`;
}

// ================= DIAGNOSTIC PANEL =================
function updateDiagnosticInfo(extra = '') {
    const info = {
        timestamp: new Date().toISOString(),
        dataLoaded: isDataLoaded,
        datasetSize: dataset.length,
        featureStats: featureStats,
        csvUrl: CSV_URL,
        userAgent: navigator.userAgent,
        online: navigator.onLine,
        extra: extra
    };
    if (diagnosticOutput) diagnosticOutput.textContent = JSON.stringify(info, null, 2);
}
