import unittest
import pandas as pd
from pricing_simulation_pipeline import DemandModeler

class TestDemandModeler(unittest.TestCase):
    """
    Unit test case for the DemandModeler class.

    This class tests the fit_models() method to ensure it produces a DataFrame
    containing the required regression model outputs: 'Intercept' and 'Coefficient_Price'.

    Attributes:
        data (pd.DataFrame): Dummy dataset simulating sales under different promotion types and prices.
        product_summary (pd.DataFrame): Mapping of product keys to cluster IDs.
    """
    def setUp(self):
        """
        Set up dummy sales data and product summary data before each test.

        Creates:
        - A small DataFrame with sales transactions across different promotions.
        - A DataFrame mapping products to cluster IDs.
        """
        self.data = pd.DataFrame({
            'PromotionType': ['Promotion A', 'Promotion A', 'Promotion B', 'Promotion B'],
            'UnitPrice': [10, 15, 10, 15],
            'SalesQuantity': [100, 80, 90, 70],
            'ProductKey': [1, 1, 2, 2]
        })
        self.product_summary = pd.DataFrame({
            'ProductKey': [1, 2],
            'Cluster': [0, 1]
        })

    def test_fit_models(self):
        """
        Test the fit_models() method to ensure it returns a valid model DataFrame.

        Verifies that:
        - The output DataFrame contains the 'Intercept' column.
        - The output DataFrame contains the 'Coefficient_Price' column.
        """
        modeler = DemandModeler(model_choice='linear')
        model_df = modeler.fit_models(self.data, self.product_summary)
        self.assertIn('Intercept', model_df.columns)
        self.assertIn('Coefficient_Price', model_df.columns)

if __name__ == '__main__':
    unittest.main()
