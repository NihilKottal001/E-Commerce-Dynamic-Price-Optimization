# DYNAMIC PRICING: OPTIMIZED PRICING STRATEGY USING MULTI-ARMED BANDIT THEORY

# Project Overview

My project is focused on developing a dynamic pricing model that utilizes both Multi-Armed Bandit (MAB) techniques and hierarchical modeling. The primary goal of this initiative is to create adaptive pricing strategies that can effectively tackle different business scenarios. These strategies are designed to:

- Maximize Revenue: Optimize pricing to increase revenue without compromising competitive positioning.
- Increase Sales Volume: Implement pricing adjustments that can boost sales, particularly useful for promotions or in peak demand periods.
- Clear Old Inventory: Use strategic price reductions to manage and reduce excess stock efficiently.
- Enhance Customer Satisfaction: Develop pricing strategies that maintain fairness and foster customer loyalty, thereby enhancing overall customer satisfaction.

This project implements a **dynamic pricing strategy** based on the **multi-armed bandit (MAB)** framework and the **m-price-change (mPC) policy** proposed by Cheung, Simchi-Levi, and Wang (2017). The objective is to learn optimal pricing decisions under demand uncertainty with minimal price experimentation while minimizing regret.

---

## 📘 Reference

Cheung, W., Simchi-Levi, D., & Wang, H. (2017).  
*Dynamic Pricing and Demand Learning with Limited Price Experimentation*.  
Operations Research, 65(6), 1722–1731.

---

## 📊 Project Objective

- Implement dynamic pricing with **limited price changes**.
- Learn **demand response models** with minimal exploration.
- Evaluate **actual revenue vs hindsight-optimal revenue**.
- Minimize **regret** while ensuring revenue performance.

---

## 🛠 Methodology Overview

1. **Data Source**  
   - Used Microsoft's `ContosoRetailDW` sample database (FactSales, FactOnlineSales, DimPromotion).

2. **Preprocessing**  
   - Cleaned unit prices and removed invalid transactions.
   - Aggregated sales by date, product, and promotion.

3. **Product Clustering**  
   - Clustered products using KMeans based on:
     - Average Price
     - Price Standard Deviation
     - Total Sales

4. **Demand Modeling**  
   - Fitted **Ridge Regression** & **ARIMA** models per (Cluster, PromotionType).
   - Captured price elasticity via model coefficients.

5. **Price Simulation (MAB Structure)**  
   - **Learning Phase**: Explored two prices per product.
   - **Earning Phase**: Fixed best-performing price.
   - Modeled as a **two-armed bandit** per product.

6. **Hindsight Optimal Revenue & Regret**  
   - Simulated revenue over a grid of prices.
   - Computed regret: Regret = OptimalHindsightRevenue - EarningPhaseRevenue

---

## 🧰 Tools & Libraries Used

- **Language**: Python 3.x  
- **Data Access**: `pyodbc` (SQL Server connection)  
- **Data Handling**: `pandas`, `numpy`  
- **Modeling**:  
  - `scikit-learn` for Ridge regression and KMeans  
  - `statsmodels` for ARIMA modeling  
- **Visualization**:  
  - `matplotlib`, `seaborn` for plots  
- **Testing**: `unittest` for validating key components

---

## 📈 Key Outputs

- **Summary Table**: ProductKey, UniquePricesUsed, Revenue, Regret
- **Visualizations**:
  - Revenue comparison (actual vs optimal)
  - Regret per product
  - Clustering (3D scatter)
  - Sales trend by promotion

---

## 🧪 Testing

- Unit tests implemented for:
  - Data cleaning and aggregation
  - Model training
  - Pricing simulation logic
- Ensures robustness and correctness.
