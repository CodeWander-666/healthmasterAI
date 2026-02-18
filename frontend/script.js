async function analyzeData() {
    const btn = document.getElementById('btn');
    const resultBox = document.getElementById('result-box');
    const riskElem = document.getElementById('risk-score');
    const triageElem = document.getElementById('triage');

    const data = {
        glucose: parseFloat(document.getElementById('glucose').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        age: parseInt(document.getElementById('age').value),
        bp: parseFloat(document.getElementById('bp').value)
    };

    if (!data.glucose || !data.bmi) return alert("Please fill all fields.");

    btn.innerText = "PROCESSING...";
    btn.disabled = true;

    try {
        // REPLACE WITH YOUR RENDER URL
        const response = await fetch('https://healthmaster-api.onrender.com/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();

        resultBox.classList.remove('hidden');
        riskElem.innerText = result.risk_score + "%";
        triageElem.innerText = "TRIAGE: " + result.triage;

        // Visual Triage Feedback
        if (result.triage === "URGENT") {
            triageElem.className = "mt-4 text-lg font-medium py-2 px-4 rounded-lg inline-block bg-red-500/20 text-red-400 border border-red-500/30";
        } else {
            triageElem.className = "mt-4 text-lg font-medium py-2 px-4 rounded-lg inline-block bg-emerald-500/20 text-emerald-400 border border-emerald-500/30";
        }
    } catch (error) {
        alert("API Error. Make sure your Render backend is live.");
    } finally {
        btn.innerText = "RUN CLINICAL ANALYSIS";
        btn.disabled = false;
    }
}
