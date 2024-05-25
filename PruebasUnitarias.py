import unittest
import json
from app import app


class TestApp(unittest.TestCase):

    def setUp(self):
        app.testing = True
        self.app = app.test_client()

    def test_get_data(self):
        response = self.app.get('/Consulta?EmployeeNumber=1')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('employee_number', data)
        self.assertIn('score', data)

    def test_hr_prediccion(self):
        headers = {'Content-Type': 'application/json'}
        data = json.dumps({"Age": 42,
               "BusinessTravel": "Travel_Frequently",
               "DailyRate": 29,
               "Department": "Research & Development",
               "DistanceFromHome": 15,
               "Education": 3,
               "EducationField": "Life Sciences",
               "EmployeeNumber": "9999",
               "EnvironmentSatisfaction": 16,
               "Gender": "Male",
               "HourlyRate": 61,
               "JobInvolvement": 2,
               "JobRole": "Research Scientist",
               "JobSatisfaction": 1,
               "MaritalStatus": "Married",
               "MonthlyIncome": 51,
               "MonthlyRate": 24907,
               "NumCompaniesWorked": 1,
               "OverTime": "Yes",
               "PercentSalaryHike": 23,
               "PerformanceRating": 4,
               "RelationshipSatisfaction": 4,
               "StockOptionLevel": 1,
               "TrainingTimesLastYear": 3,
               "WorkLifeBalance": 3,
               "YearsAtCompany": 5,
               "YearsInCurrentRole": 2,
               "YearsSinceLastPromotion": 5,
               "YearsWithCurrManager": 2})
        response = self.app.post('/Agregar', headers=headers, data=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'Agregado Correctamente!')



if __name__ == '__main__':
    unittest.main()