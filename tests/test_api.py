"""
Integration Tests for Flask REST API Endpoints.
"""

import unittest
import json
from backend.app import app


class TestFlaskAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.sample_employee = {
            "Age": 32,
            "BusinessTravel": "Travel_Rarely",
            "DailyRate": 800,
            "Department": "Research & Development",
            "DistanceFromHome": 5,
            "Education": 3,
            "EducationField": "Medical",
            "EnvironmentSatisfaction": 3,
            "Gender": "Male",
            "HourlyRate": 70,
            "JobInvolvement": 3,
            "JobLevel": 2,
            "JobRole": "Research Scientist",
            "JobSatisfaction": 3,
            "MaritalStatus": "Married",
            "MonthlyIncome": 6000,
            "MonthlyRate": 14000,
            "NumCompaniesWorked": 2,
            "OverTime": "No",
            "PercentSalaryHike": 15,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": 1,
            "TotalWorkingYears": 8,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 6,
            "YearsInCurrentRole": 4,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 4
        }

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["pipeline_ready"])

    def test_dataset_stats_endpoint(self):
        res = self.client.get("/api/dataset/stats")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("total_employees", data["data"])

    def test_model_metrics_endpoint(self):
        res = self.client.get("/api/models/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("best_model_name", data["data"])

    def test_predict_endpoint_success(self):
        res = self.client.post(
            "/api/predict",
            data=json.dumps(self.sample_employee),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("risk_percentage", data["data"])
        self.assertIn("recommendations", data["data"])

    def test_predict_endpoint_empty_payload(self):
        res = self.client.post(
            "/api/predict",
            data="",
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 400)

    def test_simulate_endpoint(self):
        payload = {
            "base": self.sample_employee,
            "modifications": {"MonthlyIncome": 10000, "OverTime": "No"}
        }
        res = self.client.post(
            "/api/simulate",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("risk_reduction_pct", data["data"])

    def test_batch_predict_json(self):
        payload = [self.sample_employee, self.sample_employee]
        res = self.client.post(
            "/api/predict/batch",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(len(data["predictions"]), 2)
        self.assertEqual(data["summary"]["total_records"], 2)


if __name__ == "__main__":
    unittest.main()
