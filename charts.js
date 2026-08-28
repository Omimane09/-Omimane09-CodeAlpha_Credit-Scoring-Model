/* ============================================
   CreditScoringAI - Charts (Chart.js)
   ============================================ */

document.addEventListener('DOMContentLoaded', function() {
    initRiskPieChart();
    initModelCompareChart();
    initFeatureImportanceChart();
});

/* ============ Risk Distribution Pie Chart ============ */
function initRiskPieChart() {
    const canvas = document.getElementById('riskPieChart');
    if (!canvas) return;

    const low = parseInt(canvas.dataset.low) || 0;
    const medium = parseInt(canvas.dataset.medium) || 0;
    const high = parseInt(canvas.dataset.high) || 0;

    if (low + medium + high === 0) {
        canvas.parentElement.innerHTML += '<p class="text-muted text-center small">No data yet</p>';
        return;
    }

    new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: ['Low Risk', 'Medium Risk', 'High Risk'],
            datasets: [{
                data: [low, medium, high],
                backgroundColor: ['#28a745', '#ffc107', '#dc3545'],
                borderColor: ['#1e7e34', '#d39e00', '#bd2130'],
                borderWidth: 3,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
                            const pct = total > 0 ? ((ctx.parsed / total) * 100).toFixed(1) : 0;
                            return `${ctx.label}: ${ctx.parsed} (${pct}%)`;
                        }
                    }
                }
            }
        }
    });
}

/* ============ Model Comparison Bar Chart ============ */
function initModelCompareChart() {
    const canvas = document.getElementById('modelCompareChart');
    if (!canvas) return;

    let results = {};
    try {
        results = JSON.parse(canvas.dataset.results || '{}');
    } catch(e) {
        canvas.parentElement.innerHTML += '<p class="text-muted text-center small">No model data</p>';
        return;
    }

    const labels = Object.keys(results);
    const accuracy = labels.map(l => results[l]?.accuracy || 0);
    const f1 = labels.map(l => results[l]?.f1 || 0);

    if (!labels.length) {
        canvas.parentElement.innerHTML += '<p class="text-muted text-center small">No model data</p>';
        return;
    }

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accuracy,
                    backgroundColor: 'rgba(26, 115, 232, 0.7)',
                    borderColor: '#1a73e8',
                    borderWidth: 2,
                },
                {
                    label: 'F1 Score (%)',
                    data: f1,
                    backgroundColor: 'rgba(40, 167, 69, 0.7)',
                    borderColor: '#28a745',
                    borderWidth: 2,
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: 'bottom' },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                }
            }
        }
    });
}

/* ============ Feature Importance Bar Chart ============ */
function initFeatureImportanceChart() {
    const canvas = document.getElementById('featureImportanceChart');
    if (!canvas) return;

    let labels = [], values = [];
    try {
        labels = JSON.parse(canvas.dataset.labels || '[]');
        values = JSON.parse(canvas.dataset.values || '[]');
    } catch (e) {
        canvas.parentElement.innerHTML += '<p class="text-muted text-center small">Train model to see feature importance</p>';
        return;
    }

    if (!labels.length || !values.length) {
        canvas.parentElement.innerHTML += '<p class="text-muted text-center small">Train model to see feature importance</p>';
        return;
    }

    // Reverse for horizontal bar (top at top)
    labels = labels.reverse();
    values = values.reverse();

    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance (%)',
                data: values,
                backgroundColor: 'rgba(111, 66, 193, 0.7)',
                borderColor: '#6f42c1',
                borderWidth: 2,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            plugins: {
                legend: { display: false },
            },
            scales: {
                x: {
                    beginAtZero: true,
                }
            }
        }
    });
}
