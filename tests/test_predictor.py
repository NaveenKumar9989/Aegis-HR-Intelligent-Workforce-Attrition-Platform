"""
Unit Tests for AttritionPredictor.
"""

import unittest
from backend.pipeline.predictor import AttritionPredictor


class TestAttritionPredictor(unittest.TestCase):

    def setUp(self):
        self.predictor = AttritionPredictor()
        self.sample_employee = {
            "Age": 30,
            "BusinessTravel": "Travel_Frequently",
            "DailyRate": 600,
            "Department": "Sales",
            "DistanceFromHome": 15,
            "Education": 3,
            "EducationField": "Marketing",
            "EnvironmentSatisfaction": 1,
            "Gender": "Male",
            "HourlyRate": 60,
            "JobInvolvement": 2,
            "JobLevel": 1,
            "JobRole": "Sales Representative",
            "JobSatisfaction": 1,
            "MaritalStatus": "Single",
            "MonthlyIncome": 2800,
            "MonthlyRate": 16000,
            "NumCompaniesWorked": 4,
            "OverTime": "Yes",
            "PercentSalaryHike": 12,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 2,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 5,
            "TrainingTimesLastYear": 2,
            "WorkLifeBalance": 1,
            "YearsAtCompany": 2,
            "YearsInCurrentRole": 2,
            "YearsSinceLastPromotion": 2,
            "YearsWithCurrManager": 1
        }

    def test_predictor_ready(self):
        self.assertTrue(self.predictor.is_ready())

    def test_predict_single_structure(self):
        result = self.predictor.predict_single(self.sample_employee)
        self.assertIn("attrition_prediction", result)
        self.assertIn("probability", result)
        self.assertIn("risk_percentage", result)
        self.assertIn("risk_level", result)
        self.assertIn("top_drivers", result)
        self.assertIn("recommendations", result)

        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)
        self.assertGreater(len(result["recommendations"]), 0)

    def test_predict_batch(self):
        batch = [self.sample_employee, self.sample_employee]
        results = self.predictor.predict_batch(batch)
        self.assertEqual(len(results), 2)
        self.assertIn("probability", results[0])
        self.assertIn("risk_label", results[0])

    def test_simulate_what_if(self):
        mods = {
            "MonthlyIncome": 9000,
            "OverTime": "No",
            "WorkLifeBalance": 4
        }
        sim_res = self.predictor.simulate_what_if(self.sample_employee, mods)
        self.assertIn("original", sim_res)
        self.assertIn("simulated", sim_res)
        self.assertIn("risk_reduction_pct", sim_res)
        # Improving salary and eliminating overtime should lower risk
        self.assertLess(sim_res["simulated"]["probability"], sim_res["original"]["probability"])
        self.assertTrue(sim_res["improved"])


if __name__ == "__main__":
    unittest.main()
