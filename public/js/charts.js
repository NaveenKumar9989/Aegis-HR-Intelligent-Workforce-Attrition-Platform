/**
 * Aegis HR - Chart Rendering Component
 * Uses Chart.js if available, with robust fallbacks.
 */

const AppCharts = {
  deptChartInstance: null,
  overtimeChartInstance: null,
  featureChartInstance: null,

  renderDepartmentChart(canvasId, departments) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = departments.map(d => d.department);
    const rates = departments.map(d => d.attrition_rate);
    const totals = departments.map(d => d.total);

    if (this.deptChartInstance) {
      this.deptChartInstance.destroy();
    }

    if (window.Chart) {
      this.deptChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Attrition Rate (%)',
            data: rates,
            backgroundColor: 'rgba(99, 102, 241, 0.7)',
            borderColor: '#6366f1',
            borderWidth: 1,
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                afterLabel: function(context) {
                  return `Total Staff: ${totals[context.dataIndex]}`;
                }
              }
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#94a3b8' }
            },
            x: {
              grid: { display: false },
              ticks: { color: '#94a3b8' }
            }
          }
        }
      });
    }
  },

  renderOvertimeChart(canvasId, overtimeData) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;

    const labels = overtimeData.map(d => `OverTime: ${d.overtime}`);
    const rates = overtimeData.map(d => d.attrition_rate);

    if (this.overtimeChartInstance) {
      this.overtimeChartInstance.destroy();
    }

    if (window.Chart) {
      this.overtimeChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
          labels: labels,
          datasets: [{
            data: rates,
            backgroundColor: ['#ef4444', '#10b981'],
            borderWidth: 2,
            borderColor: '#0e1526'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: 'bottom',
              labels: { color: '#94a3b8' }
            }
          }
        }
      });
    }
  },

  renderFeatureImportanceChart(canvasId, featureImportances) {
    const ctx = document.getElementById(canvasId);
    if (!ctx || !featureImportances) return;

    const topItems = featureImportances.slice(0, 10).reverse();
    const labels = topItems.map(f => f.feature);
    const values = topItems.map(f => f.importance);

    if (this.featureChartInstance) {
      this.featureChartInstance.destroy();
    }

    if (window.Chart) {
      this.featureChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Relative Importance',
            data: values,
            backgroundColor: 'rgba(6, 182, 212, 0.7)',
            borderColor: '#06b6d4',
            borderWidth: 1,
            borderRadius: 4
          }]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false }
          },
          scales: {
            x: {
              grid: { color: 'rgba(255, 255, 255, 0.05)' },
              ticks: { color: '#94a3b8' }
            },
            y: {
              grid: { display: false },
              ticks: { color: '#94a3b8', font: { size: 11 } }
            }
          }
        }
      });
    }
  }
};
