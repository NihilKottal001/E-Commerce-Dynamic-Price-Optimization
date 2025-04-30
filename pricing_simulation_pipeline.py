import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyodbc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split

# ----------------------------------
# 1. Database Connection
# ----------------------------------

class DatabaseConnector:

    def __init__(self, server, database):
        
        """
        Initialize the DatabaseConnector with server and database names.

        Args:
            server (str): SQL Server address or name.
            database (str): Name of the database to connect to.
        """
        self.server = server
        self.database = database

    def fetch_tables(self):
        """
        Connect to the database and fetch required tables.

        Queries the following:
        - 'FactOnlineSales' table
        - 'FactSales' table
        - 'DimPromotion' table

        Returns:
            tuple: A tuple containing three pandas DataFrames:
                - sales (DataFrame): FactOnlineSales table with selected columns.
                - simulation (DataFrame): FactSales table with selected columns.
                - promotion (DataFrame): DimPromotion table with all columns.

        Raises:
            pyodbc.Error: If the connection to the database fails.
        """
        cnxn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};Trusted_Connection=yes;', timeout=30)
        sales = pd.read_sql_query('SELECT DateKey, ProductKey, UnitPrice, DiscountAmount, SalesQuantity, ReturnQuantity, PromotionKey FROM FactOnlineSales', cnxn)
        simulation = pd.read_sql_query('SELECT DateKey, ProductKey, UnitPrice, DiscountAmount, SalesQuantity, ReturnQuantity, PromotionKey FROM FactSales', cnxn)
        promotion = pd.read_sql_query('SELECT * FROM DimPromotion', cnxn)
        cnxn.close()
        return sales, simulation, promotion

# ----------------------------------
# 2. Data Preprocessing
# ----------------------------------

class DataPreprocessor:
    
    
    @staticmethod
    def clean_sales(sales):
        """
        Cleans the sales dataset by adjusting unit prices and filtering out invalid transactions.

        Adjusts 'UnitPrice' by subtracting 'DiscountAmount'.
        Removes rows where:
        - SalesQuantity <= 0
        - UnitPrice <= 0
        - ReturnQuantity > 0

        Args:
            sales (pd.DataFrame): The raw sales data.

        Returns:
            pd.DataFrame: The cleaned sales data.
        """
        sales['UnitPrice'] -= sales['DiscountAmount']
        return sales[(sales['SalesQuantity'] > 0) & (sales['UnitPrice'] > 0) & (sales['ReturnQuantity'] <= 0)]

    @staticmethod
    def aggregate_sales(sales):
        """
        Aggregates the cleaned sales dataset.

        Groups by:
        - DateKey
        - PromotionKey
        - ProductKey
        - UnitPrice

        and sums the 'SalesQuantity'.

        Args:
            sales (pd.DataFrame): The cleaned sales data.

        Returns:
            pd.DataFrame: Aggregated sales data.
        """
        sales['DateKey'] = pd.to_datetime(sales['DateKey'])
        return sales.groupby(['DateKey', 'PromotionKey', 'ProductKey', 'UnitPrice']).agg({'SalesQuantity': 'sum'}).reset_index()

    @staticmethod
    def clean_simulation(simulation):
        """
        Cleans the simulation dataset by adjusting unit prices and removing invalid transactions.

        Same cleaning rules as sales data:
        - Adjust 'UnitPrice' using 'DiscountAmount'.
        - Remove rows with invalid sales or returns.

        Args:
            simulation (pd.DataFrame): The raw simulation data.

        Returns:
            pd.DataFrame: The cleaned simulation data.
        """
        simulation['UnitPrice'] -= simulation['DiscountAmount']
        return simulation[(simulation['SalesQuantity'] > 0) & (simulation['UnitPrice'] > 0) & (simulation['ReturnQuantity'] <= 0)]

    @staticmethod
    def aggregate_simulation(simulation):
        """
        Aggregates the cleaned simulation dataset.

        Groups by:
        - DateKey
        - PromotionKey
        - ProductKey
        - UnitPrice

        and sums the 'SalesQuantity'.

        Args:
            simulation (pd.DataFrame): The cleaned simulation data.

        Returns:
            pd.DataFrame: Aggregated simulation data.
        """
        simulation['DateKey'] = pd.to_datetime(simulation['DateKey'])
        return simulation.groupby(['DateKey', 'PromotionKey', 'ProductKey', 'UnitPrice']).agg({'SalesQuantity': 'sum'}).reset_index()

    @staticmethod
    def create_promotion_table(promotion):
        """
        Creates a simplified promotion table by categorizing promotion names.

        Categorization:
        - 'Promotion A' if 'North America' in name
        - 'Promotion B' if 'Asian' in name
        - 'Promotion C' if 'Europe' in name
        - 'No Promotion' otherwise

        Args:
            promotion (pd.DataFrame): The promotion metadata table.

        Returns:
            pd.DataFrame: A table with PromotionKey and PromotionType.
        """
        promotion['PromotionType'] = promotion['PromotionName'].apply(
            lambda x: 'Promotion A' if 'North America' in x else 
                      'Promotion B' if 'Asian' in x else 
                      'Promotion C' if 'Europe' in x else 'No Promotion')
        return promotion[['PromotionKey', 'PromotionType']]

# ----------------------------------
# 3. Clustering
# ----------------------------------

class ClusterAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()

    def elbow_method(self, features):
        scaled = self.scaler.fit_transform(features)
        wcss = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled)
            wcss.append(kmeans.inertia_)
        return wcss

    def create_clusters(self, product_summary, n_clusters):
        scaled = self.scaler.transform(product_summary[['AvgPrice', 'PriceStdDev', 'TotalSales']])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        product_summary['Cluster'] = kmeans.fit_predict(scaled)
        return product_summary

# ----------------------------------
# 4. Demand Modeling
# ----------------------------------

class DemandModeler:
    def __init__(self, model_choice='linear'):
        """
        Initializes the ClusterAnalyzer class.

        Sets up a StandardScaler instance for feature scaling.
        """
        self.model_choice = model_choice

    def fit_models(self, merged_sales, product_summary):
        """
        Assigns clusters to products based on selected features.

        Scales the input features and fits KMeans using the specified number of clusters.
        Adds a new column 'Cluster' to the product summary DataFrame.

        Args:
            product_summary (pd.DataFrame): DataFrame containing product metrics (AvgPrice, PriceStdDev, TotalSales).
            n_clusters (int): The number of clusters to form.

        Returns:
            pd.DataFrame: The product summary DataFrame with an added 'Cluster' column indicating cluster labels.
        """
        demand_model_outputs = []
        for cluster_id in product_summary['Cluster'].unique():
            cluster_products = product_summary[product_summary['Cluster'] == cluster_id]['ProductKey']
            cluster_data = merged_sales[merged_sales['ProductKey'].isin(cluster_products)]
            valid_promos = cluster_data.groupby('PromotionType')['UnitPrice'].nunique()
            valid_promos = valid_promos[valid_promos > 1].index

            for promo in valid_promos:
                promo_df = cluster_data[cluster_data['PromotionType'] == promo]
                X, y = promo_df[['UnitPrice']], promo_df['SalesQuantity']

                if self.model_choice == 'linear':
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                    model = Ridge(alpha=1).fit(X_train, y_train)
                    intercept, coef = model.intercept_, model.coef_[0]
                else:
                    daily = promo_df.groupby('DateKey').agg({'SalesQuantity': 'sum', 'UnitPrice': 'mean'}).dropna()
                    model = ARIMA(endog=daily['SalesQuantity'], exog=daily[['UnitPrice']], order=(1,0,1)).fit()
                    intercept = model.params.get('const', 0)
                    coef = model.params.get('UnitPrice', 0)

                demand_model_outputs.append({'Cluster': cluster_id, 'PromotionType': promo, 'Intercept': intercept, 'Coefficient_Price': coef})
        return pd.DataFrame(demand_model_outputs)

# ----------------------------------
# 5. Pricing Simulation
# ----------------------------------

class PricingSimulator:
    def __init__(self, phases, demand_functions):
        """
        Initialize the PricingSimulator.

        Args:
            phases (list of tuples): List of (start_date, end_date) pairs defining simulation phases.
            demand_functions (dict): Demand models with intercept and price coefficient for each (cluster_id, promotion_type).
        """
        self.phases = phases
        self.demand_functions = demand_functions

    def simulate(self, original_simulation_df, product_summary):
        """
        Runs the pricing simulation for each product over all defined phases.

        For each product:
        - Simulates daily learning phase price adjustments.
        - Chooses the best price learned during learning phase for the earning phase.
        - Calculates total revenue earned using simulated prices.
        - Computes regret compared to optimal hindsight revenue.

        Args:
            original_simulation_df (pd.DataFrame): Simulation dataset with product sales and promotions.
            product_summary (pd.DataFrame): Product features with assigned clusters.

        Returns:
            pd.DataFrame: A DataFrame containing:
                - ProductKey
                - List of PricesUsed during phases
                - EarningPhaseRevenue (revenue generated by learned prices)
                - OptimalHindsightRevenue (best possible revenue in hindsight)
                - Regret (difference between optimal and actual revenue)
        """
        results = []
        for product_id in original_simulation_df['ProductKey'].unique():
            product_df = original_simulation_df[original_simulation_df['ProductKey'] == product_id].copy()
            if product_df.empty:
                continue
            cluster_id = product_summary[product_summary['ProductKey'] == product_id]['Cluster'].values[0]
            initial_price = product_df['UnitPrice'].mean()
            current_price = initial_price
            total_actual_revenue = 0
            prices_used = []
            best_price_over_learning = None
            best_learning_revenue = -float('inf')

            for i, (start, end) in enumerate(self.phases):
                phase_df = product_df[(product_df['DateKey'] >= start) & (product_df['DateKey'] <= end)]
                if phase_df.empty:
                    continue

                if i % 2 == 0:
                    for day in sorted(phase_df['DateKey'].unique()):
                        daily_df = phase_df[phase_df['DateKey'] == day]
                        if daily_df.empty:
                            continue
                        observed_demand = daily_df['SalesQuantity'].mean()
                        promo_type = daily_df['PromotionType'].mode()[0]

                        model = self.demand_functions.get((cluster_id, promo_type))
                        if not model:
                            continue

                        proposed_price = (observed_demand - model['Intercept']) / model['Coefficient_Price'] if model['Coefficient_Price'] != 0 else current_price
                        min_price = initial_price * 0.7
                        max_price = initial_price * 1.05
                        price = proposed_price if min_price <= proposed_price <= max_price else current_price
                        current_price = price

                        predicted_demand = model['Intercept'] + model['Coefficient_Price'] * price
                        revenue = price * predicted_demand
                        total_actual_revenue += revenue
                        prices_used.append(price)

                        if revenue > best_learning_revenue:
                            best_learning_revenue = revenue
                            best_price_over_learning = price
                else:
                    promo_type = phase_df['PromotionType'].mode()[0]
                    model = self.demand_functions.get((cluster_id, promo_type))
                    if not model or best_price_over_learning is None:
                        continue
                    predicted_demand = model['Intercept'] + model['Coefficient_Price'] * best_price_over_learning
                    total_demand = predicted_demand * len(phase_df)
                    total_actual_revenue += best_price_over_learning * total_demand
                    prices_used.extend([best_price_over_learning] * len(phase_df))

            regret = self.calculate_regret(product_df, cluster_id, initial_price)
            results.append({
                'ProductKey': product_id,
                'PricesUsed': prices_used,
                'EarningPhaseRevenue': round(total_actual_revenue, 2),
                'OptimalHindsightRevenue': round(regret['optimal'], 2),
                'Regret': round(regret['optimal'] - total_actual_revenue, 2)
            })
        return pd.DataFrame(results)

    def calculate_regret(self, product_df, cluster_id, initial_price):
        """
        Calculates the optimal hindsight revenue for a product.

        Simulates different prices within a defined price range during each phase,
        and picks the price that maximizes revenue assuming perfect foresight.

        Args:
            product_df (pd.DataFrame): The sales records of a single product.
            cluster_id (int): Cluster ID to select the appropriate demand function.
            initial_price (float): Initial average price for the product.

        Returns:
            dict: A dictionary with the optimal revenue ('optimal' key).
        """
        optimal_hindsight_revenue = 0
        for i, (start, end) in enumerate(self.phases):
            phase_df = product_df[(product_df['DateKey'] >= start) & (product_df['DateKey'] <= end)]
            if phase_df.empty:
                continue
            promo_type = phase_df['PromotionType'].mode()[0]
            model = self.demand_functions.get((cluster_id, promo_type))
            if not model:
                continue
            best_rev = 0
            for price in np.linspace(initial_price * 0.7, initial_price * 1.05, 20):
                pred_demand = model['Intercept'] + model['Coefficient_Price'] * price
                rev = price * pred_demand * len(phase_df)
                best_rev = max(best_rev, rev)
            optimal_hindsight_revenue += best_rev
        return {'optimal': optimal_hindsight_revenue}

# ----------------------------------
# 6. Visualization
# ----------------------------------

class Visualizer:
    @staticmethod
    def plot_elbow(wcss):
        """
        Plots the Elbow method chart showing Within-Cluster Sum of Squares (WCSS) 
        for different numbers of clusters.

        Args:
            wcss (list): List of WCSS values for each number of clusters (k).
        """
        plt.figure(figsize=(8, 4))
        plt.plot(range(1, 11), wcss, marker='o')
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_3d_clusters(product_summary):
        """
        Creates a 3D scatter plot of clustered products.

        Args:
            product_summary (pd.DataFrame): DataFrame containing 'AvgPrice', 'PriceStdDev', 'TotalSales', and 'Cluster' columns.
        """
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(product_summary['AvgPrice'], product_summary['PriceStdDev'], product_summary['TotalSales'], c=product_summary['Cluster'], cmap='viridis')
        ax.set_xlabel('Avg Price')
        ax.set_ylabel('Price Std Dev')
        ax.set_zlabel('Total Sales')
        plt.colorbar(scatter)
        plt.show()

    @staticmethod
    def plot_sales_trend(visual_data):
        """
        Plots the annual sales trend by promotion type using line plots.

        Args:
            visual_data (pd.DataFrame): DataFrame with 'DateKey', 'SalesQuantity', and 'PromotionType' columns.
        """
        g = sns.FacetGrid(visual_data, col='PromotionType', col_wrap=2, height=4)
        g.map_dataframe(sns.lineplot, x='DateKey', y='SalesQuantity')
        g.set_titles("{col_name}")
        g.set_axis_labels("Date", "Sales")
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle('Annual Sales Trend by Promotion Type', fontsize=16)
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
        plt.show()

class RevenueComparer:
    @staticmethod
    def compare(actual_revenues, algo_revenues, phase_labels, product_id):
        """
        Plots a bar chart comparing actual revenue vs algorithm-predicted revenue across phases.

        Args:
            actual_revenues (list): List of actual revenues for each phase.
            algo_revenues (list): List of algorithm-predicted revenues for each phase.
            phase_labels (list): List of phase labels (e.g., 'Phase 1', 'Phase 2', etc.).
            product_id (int): Product identifier for title labeling.
        """
        x = range(len(phase_labels))
        bar_width = 0.35
        plt.figure(figsize=(12, 6))
        plt.bar(x, actual_revenues, width=bar_width, label='Actual Revenue', color='skyblue')
        plt.bar([i + bar_width for i in x], algo_revenues, width=bar_width, label='Algorithm Revenue', color='orange')
        plt.xlabel('Phases')
        plt.ylabel('Revenue ($)')
        plt.title(f'Revenue Comparison for Product {product_id}')
        plt.xticks([i + bar_width / 2 for i in x], phase_labels)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def compare_product(pricing_results_df, aggregated_simulation, phases):
        """
        Asks the user to select a product ID, and then compares actual vs algorithm revenue
        phase-by-phase for the selected product.

        Args:
            pricing_results_df (pd.DataFrame): Results DataFrame containing 'ProductKey' and 'PricesUsed'.
            aggregated_simulation (pd.DataFrame): Aggregated simulation sales data.
            phases (list of tuples): List of (start_date, end_date) pairs defining simulation phases.
        """
        
        product_id = input("Enter the ProductKey to compare revenues: ")

        try:
            product_id = int(product_id)
        except:
            print("Invalid ProductKey format. Must be an integer.")
            return

        if product_id not in pricing_results_df['ProductKey'].values:
            print("ProductKey not found in simulation results.")
            return

        product_df = aggregated_simulation[aggregated_simulation["ProductKey"] == product_id].copy()
        algo_prices = pricing_results_df[pricing_results_df["ProductKey"] == product_id]["PricesUsed"].values[0]

        actual_revenues = []
        algo_revenues = []
        phase_labels = []
        algo_price_idx = 0

        for i, (start, end) in enumerate(phases):
            phase_df = product_df[(product_df["DateKey"] >= start) & (product_df["DateKey"] <= end)]
            if phase_df.empty:
                actual_revenues.append(0)
                algo_revenues.append(0)
            else:
                actual_revenue = (phase_df["UnitPrice"] * phase_df["SalesQuantity"]).sum()
                actual_revenues.append(actual_revenue)

                if i % 2 == 0:
                    price_used = algo_prices[algo_price_idx]
                    algo_price_idx += 1
                else:
                    price_used = algo_prices[algo_price_idx - 1] if algo_price_idx > 0 else 0

                predicted_demand = phase_df["SalesQuantity"].mean()
                estimated_revenue = price_used * predicted_demand * len(phase_df)
                algo_revenues.append(estimated_revenue)

            phase_labels.append(f"Phase {i+1}")

        RevenueComparer.compare(actual_revenues, algo_revenues, phase_labels, product_id)
        
        
class SummaryGenerator:
    @staticmethod
    def generate_summary(pricing_results_df):
        """
        Generates a summary table for products that experimented with multiple unique prices.

        Filters products that used more than one unique price and sorts them in ascending order by regret.

        Args:
            pricing_results_df (pd.DataFrame): Simulation results DataFrame containing 'PricesUsed', 'EarningPhaseRevenue', 'OptimalHindsightRevenue', and 'Regret'.

        Returns:
            pd.DataFrame: A sorted summary table with columns:
                - ProductKey
                - UniquePricesUsed
                - EarningPhaseRevenue
                - OptimalHindsightRevenue
                - Regret
        """
        def has_multiple_unique_prices(prices):
            return len(set(prices)) > 1

        filtered_df = pricing_results_df[pricing_results_df['PricesUsed'].apply(has_multiple_unique_prices)].copy()

        summary_table = pd.DataFrame({
            'ProductKey': filtered_df['ProductKey'],
            'UniquePricesUsed': filtered_df['PricesUsed'].apply(lambda x: list(set(x))),
            'EarningPhaseRevenue': filtered_df['EarningPhaseRevenue'],
            'OptimalHindsightRevenue': filtered_df['OptimalHindsightRevenue'],
            'Regret': filtered_df['Regret']
        })

        summary_table = summary_table.sort_values(by='Regret', ascending=True)
        return summary_table