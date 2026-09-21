/**
 * Aegis HR - Application Controller & State Manager
 */

document.addEventListener('DOMContentLoaded', async () => {
  App.init();
});

const PRESETS = {
  highRisk: {
    Age: 29,
    BusinessTravel: 'Travel_Frequently',
    DailyRate: 400,
    Department: 'Sales',
    DistanceFromHome: 24,
    Education: 2,
    EducationField: 'Marketing',
    EnvironmentSatisfaction: 1,
    Gender: 'Male',
    HourlyRate: 50,
    JobInvolvement: 2,
    JobLevel: 1,
    JobRole: 'Sales Representative',
    JobSatisfaction: 1,
    MaritalStatus: 'Single',
    MonthlyIncome: 2500,
    MonthlyRate: 15000,
    NumCompaniesWorked: 5,
    OverTime: 'Yes',
    PercentSalaryHike: 11,
    PerformanceRating: 3,
    RelationshipSatisfaction: 1,
    StockOptionLevel: 0,
    TotalWorkingYears: 4,
    TrainingTimesLastYear: 1,
    WorkLifeBalance: 1,
    YearsAtCompany: 2,
    YearsInCurrentRole: 1,
    YearsSinceLastPromotion: 2,
    YearsWithCurrManager: 1
  },
  stable: {
    Age: 44,
    BusinessTravel: 'Non-Travel',
    DailyRate: 1150,
    Department: 'Research & Development',
    DistanceFromHome: 4,
    Education: 4,
    EducationField: 'Life Sciences',
    EnvironmentSatisfaction: 4,
    Gender: 'Female',
    HourlyRate: 85,
    JobInvolvement: 4,
    JobLevel: 3,
    JobRole: 'Manufacturing Director',
    JobSatisfaction: 4,
    MaritalStatus: 'Married',
    MonthlyIncome: 10500,
    MonthlyRate: 20000,
    NumCompaniesWorked: 1,
    OverTime: 'No',
    PercentSalaryHike: 18,
    PerformanceRating: 4,
    RelationshipSatisfaction: 4,
    StockOptionLevel: 2,
    TotalWorkingYears: 18,
    TrainingTimesLastYear: 4,
    WorkLifeBalance: 3,
    YearsAtCompany: 12,
    YearsInCurrentRole: 8,
    YearsSinceLastPromotion: 1,
    YearsWithCurrManager: 7
  },
  moderate: {
    Age: 35,
    BusinessTravel: 'Travel_Rarely',
    DailyRate: 850,
    Department: 'Research & Development',
    DistanceFromHome: 12,
    Education: 3,
    EducationField: 'Medical',
    EnvironmentSatisfaction: 3,
    Gender: 'Male',
    HourlyRate: 65,
    JobInvolvement: 3,
    JobLevel: 2,
    JobRole: 'Research Scientist',
    JobSatisfaction: 2,
    MaritalStatus: 'Divorced',
    MonthlyIncome: 5200,
    MonthlyRate: 14000,
    NumCompaniesWorked: 3,
    OverTime: 'Yes',
    PercentSalaryHike: 13,
    PerformanceRating: 3,
    RelationshipSatisfaction: 3,
    StockOptionLevel: 1,
    TotalWorkingYears: 9,
    TrainingTimesLastYear: 2,
    WorkLifeBalance: 2,
    YearsAtCompany: 5,
    YearsInCurrentRole: 3,
    YearsSinceLastPromotion: 3,
    YearsWithCurrManager: 3
  }
};

const App = {
  activeTab: 'overview',
  batchResults: [],
  currentBasePredictData: null,

  async init() {
    this.bindEvents();
    this.initTheme();
    await this.loadInitialData();
    this.populateForm(PRESETS.highRisk);
  },

  bindEvents() {
    // Navigation tabs
    document.querySelectorAll('.nav-item').forEach(item => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        const tab = item.getAttribute('data-tab');
        this.switchTab(tab);
      });
    });

    // Theme toggle
    const themeBtn = document.getElementById('themeToggleBtn');
    if (themeBtn) {
      themeBtn.addEventListener('click', () => this.toggleTheme());
    }

    // Presets
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const presetKey = btn.getAttribute('data-preset');
        if (PRESETS[presetKey]) {
          this.populateForm(PRESETS[presetKey]);
          this.runSinglePrediction();
        }
      });
    });

    // Single Predict Form
    const predictForm = document.getElementById('predictForm');
    if (predictForm) {
      predictForm.addEventListener('submit', (e) => {
        e.preventDefault();
        this.runSinglePrediction();
      });
    }

    // What-If Simulator Inputs
    const simOvertime = document.getElementById('simOvertime');
    const simSalary = document.getElementById('simSalary');
    const simWlb = document.getElementById('simWlb');
    const simPromotion = document.getElementById('simPromotion');

    [simOvertime, simSalary, simWlb, simPromotion].forEach(elem => {
      if (elem) {
        elem.addEventListener('input', () => this.runSimulation());
      }
    });

    // Batch File Dropzone
    const dropzone = document.getElementById('batchDropzone');
    const fileInput = document.getElementById('batchFileInput');
    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());
      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
      dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
          this.handleBatchUpload(e.dataTransfer.files[0]);
        }
      });
      fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
          this.handleBatchUpload(e.target.files[0]);
        }
      });
    }

    // Batch Filter & Export
    const filterSelect = document.getElementById('batchRiskFilter');
    if (filterSelect) {
      filterSelect.addEventListener('change', (e) => this.filterBatchTable(e.target.value));
    }

    const exportBtn = document.getElementById('batchExportBtn');
    if (exportBtn) {
      exportBtn.addEventListener('click', () => this.exportBatchCSV());
    }

    // Retrain button
    const retrainBtn = document.getElementById('retrainModelsBtn');
    if (retrainBtn) {
      retrainBtn.addEventListener('click', () => this.triggerModelRetraining());
    }
  },

  switchTab(tabId) {
    this.activeTab = tabId;
    document.querySelectorAll('.nav-item').forEach(item => {
      item.classList.toggle('active', item.getAttribute('data-tab') === tabId);
    });
    document.querySelectorAll('.tab-pane').forEach(pane => {
      pane.classList.toggle('active', pane.id === `${tabId}Tab`);
    });
  },

  initTheme() {
    const saved = localStorage.getItem('aegis-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    this.updateThemeButton(saved);
  },

  toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('aegis-theme', next);
    this.updateThemeButton(next);
  },

  updateThemeButton(theme) {
    const btn = document.getElementById('themeToggleBtn');
    if (btn) {
      btn.innerHTML = theme === 'dark' ? '<span>☀️ Light Mode</span>' : '<span>🌙 Dark Mode</span>';
    }
  },

  async loadInitialData() {
    try {
      const statsRes = await API.getDatasetStats();
      if (statsRes.success) {
        this.renderStats(statsRes.data);
      }

      const metricsRes = await API.getModelMetrics();
      if (metricsRes.success) {
        this.renderModelLab(metricsRes.data);
      }

      // Initial prediction with default form values
      await this.runSinglePrediction();
    } catch (err) {
      console.error("Error loading initial data:", err);
    }
  },

  renderStats(stats) {
    document.getElementById('kpiTotalStaff').textContent = stats.total_employees.toLocaleString();
    document.getElementById('kpiAttritionRate').textContent = `${stats.attrition_rate}%`;
    document.getElementById('kpiAttritionCount').textContent = stats.attrition_count.toLocaleString();
    document.getElementById('kpiAvgIncome').textContent = `$${Math.round(stats.avg_income_retained).toLocaleString()}`;

    // Render department and overtime charts
    AppCharts.renderDepartmentChart('deptAttritionChart', stats.departments);
    AppCharts.renderOvertimeChart('overtimeChart', stats.overtime);
  },

  renderModelLab(metricsData) {
    const container = document.getElementById('modelCardsContainer');
    if (!container) return;

    container.innerHTML = '';
    const bestModelName = metricsData.best_model_name;

    for (const [name, metrics] of Object.entries(metricsData.models)) {
      const isBest = name === bestModelName;
      const card = document.createElement('div');
      card.className = `card model-card ${isBest ? 'best' : ''}`;
      card.innerHTML = `
        ${isBest ? '<div class="best-tag">★ Production Champion</div>' : ''}
        <h3 style="font-size:18px; margin-bottom:12px;">${name}</h3>
        <div class="metric-row"><span>Accuracy</span><strong>${(metrics.accuracy * 100).toFixed(2)}%</strong></div>
        <div class="metric-row"><span>ROC-AUC Score</span><strong>${metrics.roc_auc.toFixed(4)}</strong></div>
        <div class="metric-row"><span>F1-Score</span><strong>${metrics.f1_score.toFixed(4)}</strong></div>
        <div class="metric-row"><span>5-Fold CV AUC</span><strong>${metrics.cv_roc_auc_mean.toFixed(4)}</strong></div>
        <div style="margin-top:14px; font-size:12px; color:var(--text-muted);">Confusion Matrix (Test Set):</div>
        <div class="confusion-matrix-grid">
          <div class="cm-cell tn"><strong>${metrics.confusion_matrix.true_negative}</strong><br><small>True Stay</small></div>
          <div class="cm-cell fp"><strong>${metrics.confusion_matrix.false_positive}</strong><br><small>False Churn</small></div>
          <div class="cm-cell fn"><strong>${metrics.confusion_matrix.false_negative}</strong><br><small>Missed Churn</small></div>
          <div class="cm-cell tp"><strong>${metrics.confusion_matrix.true_positive}</strong><br><small>Caught Churn</small></div>
        </div>
      `;
      container.appendChild(card);
    }

    // Render feature importance chart for best model
    const bestMetrics = metricsData.models[bestModelName];
    if (bestMetrics && bestMetrics.feature_importances) {
      AppCharts.renderFeatureImportanceChart('featureImportanceChart', bestMetrics.feature_importances);
    }
  },

  populateForm(data) {
    for (const [key, val] of Object.entries(data)) {
      const field = document.getElementById(`f_${key}`);
      if (field) {
        field.value = val;
      }
    }
  },

  getFormData() {
    const formData = {};
    document.querySelectorAll('#predictForm input, #predictForm select').forEach(field => {
      const name = field.id.replace('f_', '');
      let val = field.value;
      if (!isNaN(val) && val.trim() !== '') {
        val = Number(val);
      }
      formData[name] = val;
    });
    return formData;
  },

  async runSinglePrediction() {
    const data = this.getFormData();
    this.currentBasePredictData = data;
    try {
      const res = await API.predictSingle(data);
      if (res.success) {
        this.renderPredictionResult(res.data);
        this.syncSimulatorWithBase(data, res.data);
      }
    } catch (err) {
      console.error("Prediction error:", err);
    }
  },

  renderPredictionResult(result) {
    const scoreElem = document.getElementById('gaugeScore');
    const labelElem = document.getElementById('riskBadge');
    const progressCircle = document.getElementById('gaugeProgress');

    const score = result.risk_percentage;
    scoreElem.textContent = `${score.toFixed(1)}%`;
    labelElem.textContent = result.risk_label;
    labelElem.style.backgroundColor = result.risk_color + '22';
    labelElem.style.color = result.risk_color;
    labelElem.style.border = `1px solid ${result.risk_color}`;

    // SVG animated gauge circumference = 2 * PI * 90 = ~565.48
    const circumference = 565.48;
    const offset = circumference - (score / 100) * circumference;
    progressCircle.style.strokeDashoffset = offset;
    progressCircle.style.stroke = result.risk_color;

    // Render drivers
    const driversList = document.getElementById('topDriversList');
    driversList.innerHTML = '';
    result.top_drivers.forEach(driver => {
      const div = document.createElement('div');
      div.className = 'driver-item';
      const isPos = driver.impact === 'positive';
      div.innerHTML = `
        <span>${driver.description}</span>
        <span class="${isPos ? 'driver-tag-pos' : 'driver-tag-neg'}">${isPos ? 'Protective' : 'Risk Factor'}</span>
      `;
      driversList.appendChild(div);
    });

    // Render recommendations
    const recsList = document.getElementById('recommendationsList');
    recsList.innerHTML = '';
    result.recommendations.forEach(rec => {
      const div = document.createElement('div');
      div.className = 'rec-card';
      div.innerHTML = `
        <strong>${rec.factor} <span style="font-size:11px; float:right; opacity:0.8;">Urgency: ${rec.urgency}</span></strong>
        <span>${rec.action}</span>
      `;
      recsList.appendChild(div);
    });
  },

  syncSimulatorWithBase(baseData, baseResult) {
    document.getElementById('simBaseRisk').textContent = `${baseResult.risk_percentage}%`;
    document.getElementById('simBaseRisk').style.color = baseResult.risk_color;

    // Set simulator slider defaults to match base
    const simSalary = document.getElementById('simSalary');
    const simSalaryVal = document.getElementById('simSalaryVal');
    const simOvertime = document.getElementById('simOvertime');
    const simWlb = document.getElementById('simWlb');
    const simWlbVal = document.getElementById('simWlbVal');
    const simPromotion = document.getElementById('simPromotion');
    const simPromotionVal = document.getElementById('simPromotionVal');

    if (simSalary) {
      simSalary.value = baseData.MonthlyIncome || 5000;
      simSalaryVal.textContent = `$${Number(simSalary.value).toLocaleString()}`;
    }
    if (simOvertime) {
      simOvertime.value = baseData.OverTime || 'No';
    }
    if (simWlb) {
      simWlb.value = baseData.WorkLifeBalance || 3;
      simWlbVal.textContent = `${simWlb.value}/4`;
    }
    if (simPromotion) {
      simPromotion.value = baseData.YearsSinceLastPromotion || 0;
      simPromotionVal.textContent = `${simPromotion.value} yrs`;
    }

    this.runSimulation();
  },

  async runSimulation() {
    if (!this.currentBasePredictData) return;

    const simSalary = document.getElementById('simSalary');
    const simSalaryVal = document.getElementById('simSalaryVal');
    const simOvertime = document.getElementById('simOvertime');
    const simWlb = document.getElementById('simWlb');
    const simWlbVal = document.getElementById('simWlbVal');
    const simPromotion = document.getElementById('simPromotion');
    const simPromotionVal = document.getElementById('simPromotionVal');

    if (simSalaryVal) simSalaryVal.textContent = `$${Number(simSalary.value).toLocaleString()}`;
    if (simWlbVal) simWlbVal.textContent = `${simWlb.value}/4`;
    if (simPromotionVal) simPromotionVal.textContent = `${simPromotion.value} yrs`;

    const modifications = {
      MonthlyIncome: Number(simSalary.value),
      OverTime: simOvertime.value,
      WorkLifeBalance: Number(simWlb.value),
      YearsSinceLastPromotion: Number(simPromotion.value)
    };

    try {
      const res = await API.simulateWhatIf(this.currentBasePredictData, modifications);
      if (res.success) {
        const sim = res.data.simulated;
        const delta = res.data.risk_reduction_pct;

        const simElem = document.getElementById('simResultRisk');
        const deltaElem = document.getElementById('simRiskDelta');

        simElem.textContent = `${sim.risk_percentage}%`;
        simElem.style.color = sim.risk_color;

        if (delta > 0) {
          deltaElem.textContent = `-${delta.toFixed(1)}% (Risk Reduced)`;
          deltaElem.className = 'comp-delta improved';
        } else if (delta < 0) {
          deltaElem.textContent = `+${Math.abs(delta).toFixed(1)}% (Risk Increased)`;
          deltaElem.className = 'comp-delta';
          deltaElem.style.background = 'var(--crimson-bg)';
          deltaElem.style.color = 'var(--crimson-danger)';
        } else {
          deltaElem.textContent = `0.0% (No Change)`;
          deltaElem.className = 'comp-delta';
          deltaElem.style.background = 'rgba(255,255,255,0.05)';
          deltaElem.style.color = 'var(--text-muted)';
        }
      }
    } catch (err) {
      console.error("Simulation error:", err);
    }
  },

  async handleBatchUpload(file) {
    const statusBox = document.getElementById('batchStatusBox');
    statusBox.style.display = 'block';
    statusBox.textContent = `Processing ${file.name}... Calculating risk scores for all employees...`;

    try {
      const res = await API.predictBatch(file);
      if (res.success) {
        statusBox.textContent = `Scan complete: Evaluated ${res.summary.total_records} employees. High/Critical Risk: ${res.summary.high_or_critical_risk_count}.`;
        this.batchResults = res.predictions;
        this.renderBatchTable(res.predictions);
      } else {
        statusBox.textContent = `Error: ${res.error}`;
      }
    } catch (err) {
      statusBox.textContent = `Failed to process batch CSV: ${err.message}`;
    }
  },

  renderBatchTable(records) {
    const tableBody = document.querySelector('#batchTable tbody');
    if (!tableBody) return;
    tableBody.innerHTML = '';

    records.slice(0, 100).forEach(r => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td>#${r.index + 1}</td>
        <td><strong>${r.risk_percentage}%</strong></td>
        <td><span class="risk-badge" style="background:${r.risk_color}22; color:${r.risk_color}; border:1px solid ${r.risk_color}; padding:3px 10px; font-size:11px;">${r.risk_label}</span></td>
        <td>${r.attrition_prediction === 1 ? '<span style="color:var(--crimson-danger); font-weight:700;">Attrited (Churn)</span>' : '<span style="color:var(--emerald-success); font-weight:500;">Retained</span>'}</td>
      `;
      tableBody.appendChild(row);
    });

    const exportBtn = document.getElementById('batchExportBtn');
    if (exportBtn) exportBtn.style.display = 'inline-flex';
  },

  filterBatchTable(riskLevel) {
    if (!this.batchResults || this.batchResults.length === 0) return;
    if (riskLevel === 'ALL') {
      this.renderBatchTable(this.batchResults);
    } else {
      const filtered = this.batchResults.filter(r => r.risk_level === riskLevel);
      this.renderBatchTable(filtered);
    }
  },

  exportBatchCSV() {
    if (!this.batchResults || this.batchResults.length === 0) return;
    const headers = ['Employee_Index', 'Risk_Percentage', 'Risk_Level', 'Predicted_Attrition'];
    const rows = this.batchResults.map(r => [
      r.index + 1,
      r.risk_percentage,
      r.risk_label,
      r.attrition_prediction === 1 ? 'Yes' : 'No'
    ]);

    const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', 'employee_attrition_risk_predictions.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  },

  async triggerModelRetraining() {
    const btn = document.getElementById('retrainModelsBtn');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span>⚡ Training In Progress...</span>';

    try {
      const res = await API.triggerTraining();
      if (res.success) {
        alert('Models retrained and evaluated successfully!');
        this.renderModelLab(res.data);
      } else {
        alert(`Training failed: ${res.error}`);
      }
    } catch (err) {
      alert(`Error during training: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.innerHTML = originalText;
    }
  }
};
