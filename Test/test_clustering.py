import unittest
import pandas as pd
from pricing_simulation_pipeline import ClusterAnalyzer

class TestClusterAnalyzer(unittest.TestCase):
    """
    Unit test case for the ClusterAnalyzer class.

    This class tests the elbow_method() to ensure it returns the correct number 
    of WCSS (Within-Cluster Sum of Squares) values corresponding to cluster counts from 1 to 10.

    Attributes:
        data (pd.DataFrame): Dummy dataset simulating product features (AvgPrice, PriceStdDev, TotalSales).
        analyzer (ClusterAnalyzer): Instance of the ClusterAnalyzer to be tested.
    """
    def setUp(self):
        """
        Set up the dummy dataset and ClusterAnalyzer instance before each test.

        Creates a DataFrame with 10 sample products, each having:
        - Average Price
        - Price Standard Deviation
        - Total Sales
        """
        self.data = pd.DataFrame({
            'AvgPrice': [100, 200, 150, 180, 120, 220, 250, 190, 140, 130],
            'PriceStdDev': [10, 15, 12, 14, 11, 18, 20, 16, 13, 12],
            'TotalSales': [1000, 800, 1200, 900, 1100, 700, 650, 850, 1050, 950]
        })
        self.analyzer = ClusterAnalyzer()

    def test_elbow_method(self):
        """
        Test the elbow_method() to ensure it returns exactly 10 WCSS values.

        Verifies that:
        - The returned list has length 10 (for k=1 to k=10 clusters).
        """
        wcss = self.analyzer.elbow_method(self.data)
        self.assertEqual(len(wcss), 10)  # 1 to 10 clusters

if __name__ == '__main__':
    unittest.main()
