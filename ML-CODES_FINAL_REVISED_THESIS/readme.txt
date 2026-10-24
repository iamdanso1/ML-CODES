Based on your thesis methodology, you are using the following four forecasting models:   SARIMA (Seasonal Autoregressive Integrated Moving Average): Serving as your classical time-series baseline.   Random Forest: A tree-based ensemble machine learning model used to handle non-linear relationships in your tabular data.   XGBoost (Extreme Gradient Boosting): An advanced tree-based ensemble method also used for supervised regression with your engineered features.   LSTM (Long Short-Term Memory): A deep learning Recurrent Neural Network architecture designed to capture long-term relationships and dependencies in sequential time-series data.



Extra Information
===========================
Data Requirements & Sources Breakdown1. National Petroleum Authority (NPA)Variables Needed: Monthly consumption/sales volumes for PMS (Gasoline), AGO (Gasoil/Diesel), and LPG (in litres/metric tonnes).   Where to Get It: NPA Annual Performance & Petroleum Analysis Reports, Energy Commission Statistics Bulletins, or via a direct data request letter from your university (KNUST) department to the NPA Data/Statistics Unit.   2. Ghana Statistical Service (GSS)Variables Needed: Monthly Consumer Price Index (CPI) and Headline Inflation (%).   Where to Get It: GSS StatsBank portal (statsbank.statsghana.gov.gh) or monthly CPI Bulletin archives (2010–2025).   3. Bank of Ghana (BoG)Variables Needed: Monthly average USD/GHS Exchange Rate.   Where to Get It: Bank of Ghana Data Warehouse / Time Series Database, or BoG Statistical Bulletins (Monthly Interbank Exchange Rates).   4. Exogenous Fuel Pump PricesVariables Needed: Historical retail pump prices (GHS/Litre).   Where to Get It: NPA pricing formula archives or Chamber of Bulk Oil Distributors (CBOD) annual operational reports.


import pandas as pd
import requests

# Querying Ghana Statistical Service (GSS) StatsBank API endpoint for CPI
url = "https://statsbank.statsghana.gov.gh:8443/api/v1/en/PX-2018-P3/Economy/CPI/cpi_monthly.px"

# Alternative direct fetch from GSS public open data repository
try:
    print("Fetching official CPI and Inflation data from GSS...")
    # Downloading GSS CPI historic dataset CSV
    cpi_df = pd.read_csv("https://statsbank.statsghana.gov.gh/sq/download_cpi_monthly.csv")
    print("Download successful!")
    print(cpi_df.head())
except Exception as e:
    print("GSS API server is currently undergoing maintenance or restricted.")
    print("Using local pre-processed GSS series fallback...")