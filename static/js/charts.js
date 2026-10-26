/**
 * NewsGuard Chart.js Visualizations
 */

function isDarkMode() {
  return document.documentElement.getAttribute('data-theme') === 'dark';
}

function getChartColors() {
  const dark = isDarkMode();
  return {
    text: dark ? '#cbd5e1' : '#334155',
    grid: dark ? '#1e355b' : '#e2e8f0',
    primary: '#2563eb',
    accent: '#38bdf8',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    info: '#0ea5e9'
  };
}

// 1. Session Outcomes Chart (Dashboard)
let outcomesChartInstance = null;

function initOutcomesChart(stats) {
  const ctx = document.getElementById('outcomesChart');
  if (!ctx) return;

  const colors = getChartColors();
  const credible = stats.credible || 0;
  const fake = stats.fake || 0;
  const clickbait = stats.clickbait || 0;
  const mismatch = stats.mismatch || 0;

  // If no sessions yet, show demo distribution placeholder
  const hasData = (credible + fake + clickbait + mismatch) > 0;
  const chartData = hasData ? [credible, fake, clickbait, mismatch] : [1, 0, 0, 0];

  if (outcomesChartInstance) {
    outcomesChartInstance.destroy();
  }

  outcomesChartInstance = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Credible Journalism', 'Suspicious / Fake', 'Clickbait Warnings', 'Headline-Body Mismatch'],
      datasets: [{
        data: chartData,
        backgroundColor: [
          colors.success,
          colors.danger,
          colors.warning,
          colors.info
        ],
        borderWidth: 2,
        borderColor: isDarkMode() ? '#111f38' : '#ffffff'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: function(context) {
              if (!hasData) return 'No articles analyzed yet';
              return ` ${context.label}: ${context.raw} articles`;
            }
          }
        }
      },
      cutout: '68%'
    }
  });
}

// 2. Model Performance Comparison Charts
let fnChartInstance = null;
let cbChartInstance = null;

function initPerformanceCharts(evalData) {
  const colors = getChartColors();

  // Fake News Models Chart
  const fnCtx = document.getElementById('fakeNewsModelChart');
  if (fnCtx && evalData && evalData.fake_news_task && evalData.fake_news_task.models_comparison) {
    const models = evalData.fake_news_task.models_comparison;
    const modelNames = Object.keys(models);

    const accuracies = modelNames.map(m => Math.round(models[m].accuracy * 1000) / 10);
    const precisions = modelNames.map(m => Math.round(models[m].precision_macro * 1000) / 10);
    const recalls = modelNames.map(m => Math.round(models[m].recall_macro * 1000) / 10);
    const f1s = modelNames.map(m => Math.round(models[m].f1_macro * 1000) / 10);

    if (fnChartInstance) fnChartInstance.destroy();

    fnChartInstance = new Chart(fnCtx, {
      type: 'bar',
      data: {
        labels: modelNames,
        datasets: [
          { label: 'Accuracy (%)', data: accuracies, backgroundColor: '#2563eb' },
          { label: 'Macro Precision (%)', data: precisions, backgroundColor: '#38bdf8' },
          { label: 'Macro Recall (%)', data: recalls, backgroundColor: '#818cf8' },
          { label: 'Macro F1 (%)', data: f1s, backgroundColor: '#10b981' }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 80,
            max: 100,
            ticks: { color: colors.text },
            grid: { color: colors.grid }
          },
          x: {
            ticks: { color: colors.text, font: { size: 10 } },
            grid: { display: false }
          }
        },
        plugins: {
          legend: {
            labels: { color: colors.text, font: { size: 11 } }
          }
        }
      }
    });
  }

  // Clickbait Models Chart
  const cbCtx = document.getElementById('clickbaitModelChart');
  if (cbCtx && evalData && evalData.clickbait_task && evalData.clickbait_task.models_comparison) {
    const models = evalData.clickbait_task.models_comparison;
    const modelNames = Object.keys(models);

    const accuracies = modelNames.map(m => Math.round(models[m].accuracy * 1000) / 10);
    const precisions = modelNames.map(m => Math.round(models[m].precision_macro * 1000) / 10);
    const recalls = modelNames.map(m => Math.round(models[m].recall_macro * 1000) / 10);
    const f1s = modelNames.map(m => Math.round(models[m].f1_macro * 1000) / 10);

    if (cbChartInstance) cbChartInstance.destroy();

    cbChartInstance = new Chart(cbCtx, {
      type: 'bar',
      data: {
        labels: modelNames,
        datasets: [
          { label: 'Accuracy (%)', data: accuracies, backgroundColor: '#2563eb' },
          { label: 'Macro Precision (%)', data: precisions, backgroundColor: '#38bdf8' },
          { label: 'Macro Recall (%)', data: recalls, backgroundColor: '#818cf8' },
          { label: 'Macro F1 (%)', data: f1s, backgroundColor: '#10b981' }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            min: 85,
            max: 100,
            ticks: { color: colors.text },
            grid: { color: colors.grid }
          },
          x: {
            ticks: { color: colors.text, font: { size: 10 } },
            grid: { display: false }
          }
        },
        plugins: {
          legend: {
            labels: { color: colors.text, font: { size: 11 } }
          }
        }
      }
    });
  }
}
