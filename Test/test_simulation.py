import unittest
import pandas as pd
from pricing_simulation_pipeline import PricingSimulator

class TestPricingSimulator(unittest.TestCase):
    """
    Unit test case for the PricingSimulator class.

    This class tests the simulate() method to ensure it correctly generates 
    revenue and pricing outputs for products based on given demand functions and simulation phases.

    Attributes:
        sim_data (pd.DataFrame): Dummy simulation sales data.
        phases (list of tuples): Learning and earning phase date ranges.
        demand_functions (dict): Mapping of (cluster_id, promotion_type) to simple demand models.
        product_summary (pd.DataFrame): Mapping of products to cluster IDs.
    """
    def setUp(self):
        """
        Set up dummy simulation sales data, phases, demand functions, and product summary.

        Creates:
        - A simple simulation dataset over two learning/earning phases.
        - Basic demand functions mapping clusters and promotion types.
        - Product-to-cluster assignment.
        """
        self.sim_data = pd.DataFrame({
            'DateKey': pd.date_range(start='2023-01-01', periods=4),
            'PromotionKey': [10, 10, 20, 20],
            'ProductKey': [1, 1, 1, 1],
            'UnitPrice': [10, 11, 12, 13],
            'SalesQuantity': [100, 110, 90, 95],
            'PromotionType': ['Promotion A', 'Promotion A', 'Promotion B', 'Promotion B']
        })
        self.phases = [('2023-01-01', '2023-01-02'), ('2023-01-03', '2023-01-04')]
        self.demand_functions = {
            (0, 'Promotion A'): {'Intercept': 200, 'Coefficient_Price': -5},
            (0, 'Promotion B'): {'Intercept': 180, 'Coefficient_Price': -4}
        }
        self.product_summary = pd.DataFrame({'ProductKey': [1], 'Cluster': [0]})

    def test_simulate(self):
        """
        Test the simulate() method.

        Verifies that:
        - The simulation result DataFrame contains the 'EarningPhaseRevenue' column.
        """
        simulator = PricingSimulator(phases=self.phases, demand_functions=self.demand_functions)
        result_df = simulator.simulate(self.sim_data, self.product_summary)
        self.assertIn('EarningPhaseRevenue', result_df.columns)

if __name__ == '__main__':
    unittest.main()
