import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm, multivariate_normal
from numpy.linalg import pinv
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Set the title for the Streamlit app
st.title("Simple Kelly Criterion")

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Kelly Optimal Betting Fraction",
    "Optimal Kelly Fraction Simulation (1 stock & risk free instrument)",
    "Optimal Portfolio of Correlated Stocks"
])

with tab1:
# Title for the app
  st.sidebar.header("Expected Growth Rate vs Capital Fraction in Stock")

  # Let the user adjust the parameters
  param1 = st.slider("Parameter 1 (Growth multiplier)", min_value=0.1, max_value=3.0, value=1.7, step=0.1)
  param2 = st.slider("Parameter 2 (Loss multiplier)", min_value=0.1, max_value=3.0, value=0.7, step=0.1)

  # Generate the data based on user input
  u = np.linspace(0, 1, num=100)
  xgr = 0.5 * (np.log(1 + param1 * u) + np.log(1 - param2 * u))
  # Create a DataFrame
  df = pd.DataFrame({'perc_capital': u, 'expected_growth': xgr})
  # Plot using Seaborn
  st.subheader("Expected Growth Rate vs Capital Fraction")
  fig, ax = plt.subplots()
  sns.lineplot(data=df, x='perc_capital', y='expected_growth', ax=ax)
  plt.xlabel("Percentage of Capital")
  plt.ylabel("Expected Growth Rate")
  st.pyplot(fig)


with tab2:
  # Sidebar for user input in Tab 1
  st.sidebar.header("Kelly Fraction Simulation Parameters")

  sp500_top50 = [
      'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'BRK-B', 'NVDA', 'TSLA', 'META',
      'UNH', 'JNJ', 'V', 'XOM', 'WMT', 'JPM', 'MA', 'PG', 'HD', 'BAC', 'PFE',
      'ABBV', 'KO', 'DIS', 'PEP', 'NFLX', 'CSCO', 'NKE', 'LLY', 'MCD', 'VZ',
      'MRK', 'TMO', 'CVX', 'ABT', 'CMCSA', 'DHR', 'AVGO', 'COST', 'ACN', 'NEE',
      'WFC', 'LIN', 'TXN', 'ADBE', 'MDT', 'HON', 'UNP', 'ORCL', 'PM'
  ]
  # Ticker selection from sp500_top50
  tickers = st.sidebar.multiselect("Select Tickers",
                                   sp500_top50,
                                   default=["AAPL", "MSFT", "GOOGL"])

  selected_stock = tickers[0]
  n_simulations = st.sidebar.slider("Number of Simulations",
                                    min_value=100,
                                    max_value=5000,
                                    value=1000)
  n_months = st.sidebar.slider("Number of Months",
                               min_value=10,
                               max_value=120,
                               value=30)
  n_steps = st.sidebar.slider("Number of Kelly Steps",
                              min_value=5,
                              max_value=50,
                              value=25)

  run_simulation = st.sidebar.button("Run Simulation")

  if run_simulation:
    # Download the selected stock data
    dax = yf.download(selected_stock, start='1900-1-1', interval='1mo')
    dax['monthly_returns'] = (dax['Close'] / dax['Close'].shift(1)
                              ) - 1  # Calculate the monthly returns
    mu = dax['monthly_returns'].mean()
    sigma = dax['monthly_returns'].std()
    r = 0.03 / 12.0  # Monthly risk-free rate

    # Initialize a list to store the grouped DataFrames for each Kelly fraction
    dff_list = []

    # Iterate through all Kelly scenarios to find the optimal Kelly fraction
    for u in range(n_steps):
      kelly_fraction = u / n_steps  # Calculate the Kelly fraction for this step

      # Initialize a list to store DataFrames for each simulation
      df_simulations = []

      for i in range(n_simulations):
        # Simulate monthly returns
        rvs = norm.rvs(loc=mu, scale=sigma, size=n_months)

        # Initialize wealth list
        w = [1.0] * n_months

        for j in range(1, n_months):
          # Clip returns to avoid extreme values
          if rvs[j] < -0.99:
            rvs[j] = -0.99
          elif rvs[j] > 0.99:
            rvs[j] = 0.99

          # Calculate portfolio return with Kelly fraction and risk-free rate
          port_ret = (kelly_fraction * (rvs[j - 1] - r)) + (1 + r)
          w[j] = w[j - 1] * port_ret

        # Create a DataFrame for this simulation
        dfa = pd.DataFrame({
            'returns': rvs,
            'wealth': w,
            'month': np.arange(1, n_months + 1),
            'sim': i,
            'kelly': kelly_fraction,
            'log_wealth': np.log(w)
        })

        # Append the simulation DataFrame to the list
        df_simulations.append(dfa)

      # Concatenate all simulation DataFrames for this Kelly fraction
      df_concat = pd.concat(df_simulations, ignore_index=True)

      # Group by 'kelly' and 'month' and calculate the mean
      df_grouped = df_concat.groupby(['kelly', 'month']).mean().reset_index()

      # Append the grouped DataFrame to the dff_list
      dff_list.append(df_grouped)

    # Concatenate all grouped DataFrames from different Kelly fractions
    dff = pd.concat(dff_list, ignore_index=True)

    col1, col2 = st.columns(2)

    with col1:
      fig1, ax1 = plt.subplots()
      sns.lineplot(data=df_concat,
                   x='month',
                   y='log_wealth',
                   hue='sim',
                   ax=ax1)
      ax1.set_title("Kelly Fraction Simulation Results (Simulations)")
      ax1.set_xlabel("Month")
      ax1.set_ylabel("Log Wealth")
      st.pyplot(fig1)

    with col2:
      fig2, ax2 = plt.subplots()
      sns.lineplot(data=dff, x='month', y='log_wealth', hue='kelly', ax=ax2)
      ax2.set_title("Kelly Fraction Simulation Results (Averaged)")
      ax2.set_xlabel("Month")
      ax2.set_ylabel("Log Wealth")
      st.pyplot(fig2)

with tab3:
  # Sidebar for user input in Tab 2
  st.sidebar.header("Optimal Portfolio Parameters")


  ticker1 = tickers[0]
  ticker2 = tickers[1]
  ticker3 = tickers[2]

  start_date = st.sidebar.date_input("Start Date",
                                     value=pd.to_datetime('2013-01-01'))
  end_date = st.sidebar.date_input("End Date",
                                   value=pd.to_datetime('2023-04-30'))

  run_analysis = st.sidebar.button("Run Analysis")

  if run_analysis:
    # Download historical stock data using yfinance
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    # Calculate daily returns for each stock
    rets_fresenius = data[tickers[0]].pct_change().dropna()
    rets_deutsche_bank = data[tickers[1]].pct_change().dropna()
    rets_commerz_bank = data[tickers[2]].pct_change().dropna()

    # Plotting scatter plots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Scatter plot for Fresenius vs Deutsche Bank
    axs[0].scatter(rets_fresenius, rets_deutsche_bank, alpha=0.5)
    axs[0].plot(np.unique(rets_fresenius),
                np.poly1d(np.polyfit(rets_fresenius, rets_deutsche_bank,
                                     1))(np.unique(rets_fresenius)),
                color='red')
    axs[0].set_title(f'{ticker1} vs {ticker2}')

    # Scatter plot for Fresenius vs Commerz Bank
    axs[1].scatter(rets_fresenius, rets_commerz_bank, alpha=0.5)
    axs[1].plot(np.unique(rets_fresenius),
                np.poly1d(np.polyfit(rets_fresenius, rets_commerz_bank,
                                     1))(np.unique(rets_fresenius)),
                color='red')
    axs[1].set_title(f'{ticker1} vs {ticker3}')

    # Scatter plot for Commerz Bank vs Deutsche Bank
    axs[2].scatter(rets_commerz_bank, rets_deutsche_bank, alpha=0.5)
    axs[2].plot(np.unique(rets_commerz_bank),
                np.poly1d(np.polyfit(rets_commerz_bank, rets_deutsche_bank,
                                     1))(np.unique(rets_commerz_bank)),
                color='red')
    axs[2].set_title(f'{ticker2} vs {ticker3}')

    plt.tight_layout()
    st.pyplot(fig)

    # Calculate and display correlations
    correlation_fd = np.corrcoef(rets_fresenius, rets_deutsche_bank)[0, 1]
    correlation_fc = np.corrcoef(rets_fresenius, rets_commerz_bank)[0, 1]
    st.write(
        f'Correlation between {ticker1} and {ticker2}: {correlation_fd}')
    st.write(
        f'Correlation between {ticker1} and {ticker3} {correlation_fc}')
    st.write(
        f'Correlation between {ticker2} and {ticker3} {correlation_fc}')