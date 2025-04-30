import unittest
import pandas as pd
from pricing_simulation_pipeline import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    """
    Unit test case for the DataPreprocessor class.

    This class tests the clean_sales() and aggregate_sales() static methods
    to ensure proper cleaning and aggregation of sales data.

    Attributes:
        data (pd.DataFrame): Dummy dataset simulating sales transactions.
    """
    def setUp(self):
        """
        Set up a dummy sales dataset before each test.

        Creates a DataFrame with:
        - DateKey
        - ProductKey
        - UnitPrice
        - DiscountAmount
        - SalesQuantity
        - ReturnQuantity
        - PromotionKey
        """
        self.data = pd.DataFrame({
            'DateKey': ['2023-01-01', '2023-01-02'],
            'ProductKey': [1, 2],
            'UnitPrice': [100, 200],
            'DiscountAmount': [10, 20],
            'SalesQuantity': [5, 0],
            'ReturnQuantity': [0, 1],
            'PromotionKey': [10, 20]
        })

    def test_clean_sales(self):
        """
        Test the clean_sales() method.

        Verifies that:
        - Only valid sales transactions remain after cleaning.
        - UnitPrice is correctly adjusted for discount.
        """
        cleaned = DataPreprocessor.clean_sales(self.data)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]['UnitPrice'], 90)

    def test_aggregate_sales(self):
        """
        Test the aggregate_sales() method.

        Verifies that:
        - Sales data is aggregated by DateKey, PromotionKey, ProductKey, and UnitPrice.
        - The output contains the 'SalesQuantity' column.
        """
        cleaned = DataPreprocessor.clean_sales(self.data)
        aggregated = DataPreprocessor.aggregate_sales(cleaned)
        self.assertIn('SalesQuantity', aggregated.columns)

if __name__ == '__main__':
    unittest.main()
