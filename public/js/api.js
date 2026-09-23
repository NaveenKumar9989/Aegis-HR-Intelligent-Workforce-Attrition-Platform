/**
 * Aegis HR - API Client
 * Manages communication with backend REST API endpoints.
 */

const API = {
  baseUrl: '',

  async checkHealth() {
    const res = await fetch(`${this.baseUrl}/api/health`);
    return await res.json();
  },

  async getDatasetStats() {
    const res = await fetch(`${this.baseUrl}/api/dataset/stats`);
    return await res.json();
  },

  async getModelMetrics() {
    const res = await fetch(`${this.baseUrl}/api/models/metrics`);
    return await res.json();
  },

  async triggerTraining() {
    const res = await fetch(`${this.baseUrl}/api/models/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    return await res.json();
  },

  async predictSingle(employeeData) {
    const res = await fetch(`${this.baseUrl}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(employeeData)
    });
    return await res.json();
  },

  async predictBatch(fileOrList) {
    let options = {};
    if (fileOrList instanceof File) {
      const formData = new FormData();
      formData.append('file', fileOrList);
      options = {
        method: 'POST',
        body: formData
      };
    } else {
      options = {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(fileOrList)
      };
    }
    const res = await fetch(`${this.baseUrl}/api/predict/batch`, options);
    return await res.json();
  },

  async simulateWhatIf(baseData, modifications) {
    const res = await fetch(`${this.baseUrl}/api/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ base: baseData, modifications: modifications })
    });
    return await res.json();
  }
};
