// Configuration – replace with your actual backend URL
const API_URL = 'https://your-backend.onrender.com/predict';

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';
    resultDiv.innerHTML = '';

    // Collect form data
    const formData = {
        pregnancies: parseFloat(document.getElementById('pregnancies').value),
        glucose: parseFloat(document.getElementById('glucose').value),
        bloodpressure: parseFloat(document.getElementById('bloodpressure').value),
        skinthickness: parseFloat(document.getElementById('skinthickness').value),
        insulin: parseFloat(document.getElementById('insulin').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        dpf: parseFloat(document.getElementById('dpf').value),
        age: parseFloat(document.getElementById('age').value)
    };

    // Basic validation
    for (let key in formData) {
        if (isNaN(formData[key])) {
            alert('Please fill all fields with valid numbers.');
            return;
        }
    }

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        displayResult(data, formData);
    } catch (error) {
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
    }
});

function displayResult(data, inputFeatures) {
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = '';

    // Prediction box
    const predBox = document.createElement('div');
    predBox.id = 'prediction';
    const riskClass = data.prediction === 1 ? 'high-risk' : 'low-risk';
    predBox.className = riskClass;
    predBox.textContent = data.prediction === 1
        ? `⚠️ High Risk (Probability: ${(data.probability * 100).toFixed(1)}%)`
        : `✅ Low Risk (Probability: ${(data.probability * 100).toFixed(1)}%)`;
    resultDiv.appendChild(predBox);

    // Feature importance chart
    const canvas = document.createElement('canvas');
    canvas.id = 'importance-chart';
    canvas.width = 400;
    canvas.height = 300;
    resultDiv.appendChild(canvas);

    const ctx = canvas.getContext('2d');
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.feature_names,
            datasets: [{
                label: 'Feature Importance (coefficient magnitude)',
                data: data.importance,
                backgroundColor: 'rgba(52, 152, 219, 0.6)',
                borderColor: 'rgba(52, 152, 219, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Coefficient Value' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}
