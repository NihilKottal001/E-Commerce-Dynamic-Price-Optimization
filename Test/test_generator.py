import unittest
import pandas as pd
from pricing_simulation_pipeline import SummaryGenerator

class TestSummaryGenerator(unittest.TestCase):
    """
    Unit test case for the SummaryGenerator class.

    This class tests the generate_summary() method to ensure:
    - The output DataFrame includes a 'Regret' column.
    - The 'Regret' values are sorted in ascending order (monotonic increasing).

    Attributes:
        data (pd.DataFrame): Dummy dataset simulating pricing simulation results.
    """
    def setUp(self):
        """
        Set up dummy pricing simulation results before each test.

        Creates a small DataFrame with:
        - ProductKey
        - PricesUsed
        - EarningPhaseRevenue
        - OptimalHindsightRevenue
        - Regret
        """
        self.data = pd.DataFrame({
            'ProductKey': [1, 2],
            'PricesUsed': [[10, 11], [12]],
            'EarningPhaseRevenue': [1000, 800],
            'OptimalHindsightRevenue': [1100, 900],
            'Regret': [100, 100]
        })

    def test_generate_summary(self):
        """
        Test the generate_summary() method to ensure correct output structure and sorting.

        Verifies that:
        - The 'Regret' column is present in the output DataFrame.
        - The 'Regret' values are sorted in ascending order.
        """
        summary = SummaryGenerator.generate_summary(self.data)
        self.assertIn('Regret', summary.columns)
        self.assertTrue(summary['Regret'].is_monotonic_increasing)

if __name__ == '__main__':
    unittest.main()
