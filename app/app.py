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
from scipy.optimize import minimize
from scipy.stats import binom

# Set the title for the Streamlit app
st.title("Simple Kelly Criterion")

st.sidebar.header("Select Up to 3 Stocks for your portfolio")

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
# Error handling to ensure at least 3 tickers are selected
if len(tickers) < 3:
    st.sidebar.error("Please select at least 3 tickers.")
    # Set default tickers if not enough are selected
    tickers = ["AAPL", "MSFT", "GOOGL"]

ticker1 = tickers[0]
ticker2 = tickers[1]
ticker3 = tickers[2]

start_date = st.sidebar.date_input("Start Date",
                                    value=pd.to_datetime('2013-01-01'))
end_date = st.sidebar.date_input("End Date",
                                  value=pd.to_datetime('2023-04-30'))
# Create tabs
tab1, tab2, tab3 = st.tabs([
    "Kelly Optimal Betting Fraction",
    "Optimal Kelly Fraction Simulation",
    "Optimal Portfolio of Correlated Stocks"
])

with tab1:
# Title for the app
  # Brief explanation of the Kelly formula
  kelly_formula = r'''
  ## Kelly Criterion Derivation
  ### Expected Geometric Growth Rate
  $$ 
  r = (1+fb)^{p} * (1-fa)^{1-p}
  $$
  
  Where $f$ is the fraction of the portfolio to be invested, $p$ is the probability of success, $b$ is the growth multiplier, and $a$ is the loss multiplier.

  Now take ARGMAX of the log expected growth rate.
  $$
  ARGMAX  p * log(1 + fb) + (1-p) * log(1 - fa)
  $$

  We can then solve for $f$ to get the optimal fraction of the portfolio to be invested by taking the derivative of the above equation with respect to $f$ and setting it to zero.
  $$
  \frac{d}{df} [p * log(1 + fb) + (1-p) * log(1 - fa)] = 0
  $$
  Finally after rearranging we get, giving us the kelly fraction:
  $$
  f^{*} = \frac{p}{a} - \frac{1-p}{b}
  $$
  '''
  st.write(kelly_formula)
  # Let the user adjust the parameters
  param1 = st.slider("Parameter 1 (Growth multiplier)", min_value=0.1, max_value=3.0, value=1.5, step=0.05)
  param2 = st.slider("Parameter 2 (Loss multiplier)", min_value=0.05, max_value=1.0, value=0.5, step=0.05)
  param3 = st.slider("Parameter 3 (Probability of Success)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
  
  # Generate the data based on user input
  u = np.linspace(0, 1, num=100)
  xgr = param3 * np.log(1 + param1 * u) + (1-param3) * np.log(1 - param2 * u) 
  # Create a DataFrame
  df = pd.DataFrame({'perc_capital': u, 'expected_growth': xgr})
  # Plot using Seaborn
  st.subheader("Expected Growth Rate vs Capital Fraction")
  fig, ax = plt.subplots()
  sns.lineplot(data=df, x='perc_capital', y='expected_growth', ax=ax)
  plt.xlabel("Percentage of Capital")
  plt.ylabel("Expected Growth Rate")
  st.pyplot(fig)

  # lets do an additional exercise by solving for the kelly optimal percent of capital to invest based on the scenario we present
  # and write a function in python to calculate and plot the full kelly, half kelly, and fully invested strategies, 
  # similar to the above, but done in a nice python function where inputs are provided
  param4 = st.slider("Parameter 4 (Number of Periods)", min_value=10, max_value=1000, value=50, step=10)
  param5 = st.slider("Parameter 5 (Percent Kelly to Compare)", min_value=0.05, max_value=1.0, value=0.5, step=0.05)
  def log_util(u, p1, r1, r2):
    # inverse log utility so we can use the scipy minimize function
    return -1.0 *(p1*np.log(1+r1*u) + (1-p1)*np.log(1-r2*u))

  def minimize_util(p1, r1, r2):
    kr = minimize(log_util, 0, args = (p1, r1, r2), method = 'BFGS')
    return kr.x[0]

  def wealth(p1, r1, r2, n_trades, wealth_1, f):
    # n is the number of trades, wealth_1 is starting wealth, f is the kelly fraction to be considered
    k = minimize_util(p1, r1, r2)
    wealth, wealth_k, wealth_fk = [wealth_1] * n_trades, [wealth_1] * n_trades, [wealth_1] * n_trades
    periods = np.linspace(0, n_trades-1, num=n_trades)
    outcomes = binom.rvs(1, p1, size = n_trades)
    for i in range(1, len(outcomes)):
      if outcomes[i] == 0:
        wealth[i] = wealth[i-1] * (1-r2)
        wealth_k[i] = wealth_k[i-1] * k * (1-r2) + wealth_k[i-1] * (1-k)
        wealth_fk[i] = wealth_fk[i-1] * f*k * (1-r2) + wealth_fk[i-1] * (1-f*k)
      else:
        wealth[i] = wealth[i-1] * (1+r1)
        wealth_k[i] = wealth_k[i-1] * k * (1+r1) + wealth_k[i-1] * (1-k)
        wealth_fk[i] = wealth_fk[i-1] * f*k * (1+r1) + wealth_fk[i-1] * (1-f*k)
    df = pd.DataFrame(data = {'Invest Everything':wealth,
                            'Invest Kelly Optimal Amount':wealth_k,
                            f'Invest {param5 * 100}% of Kelly Optimal Amount':wealth_fk,
                            'periods':periods})
    return df

  df = wealth(param3, param1, param2, param4, 1, param5) # make this selectable in the streamlit application
  dfm = df.melt('periods', var_name = 'wealth_fraction', value_name = 'wealth')
  st.subheader("Expected Wealth w/ Different Kelly Fractions")
  fig2, ax2 = plt.subplots()
  sns.lineplot(data=dfm, x='periods', y='wealth', hue='wealth_fraction', ax=ax2)
  plt.xlabel("Investment Periods")
  plt.ylabel("Expected Wealth")
  st.pyplot(fig2)

with tab2:
  tab2_text = '''
  ## Kelly Fraction Simulation
  Simulate the log wealth of different kelly fractions for a given stock and risk free rate.
  Select a stock, a risk free rate, periods (months), simulations, and kelly steps to view simulation results.
  '''
  st.write(tab2_text)
  # Sidebar for user input in Tab 1
  selected_stock = st.multiselect("Select Tickers",
                                  tickers)
  risk_free_rate = st.slider("Risk Free Rate",
                                    min_value=0.0,
                                    max_value=0.15,
                                    value=0.03,
                                    step=0.01)
  n_simulations = st.slider("Number of Simulations",
                                    min_value=100,
                                    max_value=5000,
                                    value=1000)
  n_months = st.slider("Number of Months",
                               min_value=10,
                               max_value=120,
                               value=30)
  n_steps = st.slider("Number of Kelly Steps",
                              min_value=5,
                              max_value=50,
                              value=25)

  run_simulation = st.button("Run Simulation")
  # Error handling to ensure at least 3 tickers are selected

  if selected_stock:
    # Download the selected stock data
    dax = yf.download(selected_stock, start=start_date, end=end_date, interval='1mo')
    dax['monthly_returns'] = (dax['Close'] / dax['Close'].shift(1)
                              ) - 1  # Calculate the monthly returns
    mu = dax['monthly_returns'].mean()
    sigma = dax['monthly_returns'].std()
    normal_returns = norm.rvs(loc=mu, scale=sigma, size = 1000)
    col11, col12 = st.columns(2)
    with col11:
      fig11, ax11 = plt.subplots()
      sns.lineplot(data=dax,
                   x='Date',
                   y='Close',
                   hue='sim',
                   ax=ax1)
      ax11.set_title("Daily Close Price")
      ax11.set_xlabel("Month")
      ax11.set_ylabel("Price at Close")
      st.pyplot(fig11)
    with col12:
      fig12, ax12 = plt.subplots()
      sns.displot(df['monthly_returns'])
      sns.kdeplot(normal_returns, ax=ax1)
      ax12.set_title("Distribution of Monthly Returns")
      ax12.set_xlabel("Monthly Returns")
      ax12.set_ylabel("Density")
      st.pyplot(fig12)
  if run_simulation:

    r = risk_free_rate / 12.0  # Monthly risk-free rate

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
  st.header("Optimal Portfolio Parameters")
  run_analysis = st.button("Run Analysis")

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