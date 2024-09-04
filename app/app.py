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
st.title("Financial Simulations")

# Create tabs
tab1, tab2 = st.tabs([
    "Optimal Kelly Fraction Simulation (1 stock & risk free instrument)",
    "Optimal Portfolio of Correlated Stocks"
])

with tab1:
  # Sidebar for user input in Tab 1
  st.sidebar.header("Kelly Fraction Simulation Parameters")

  sp500_top50 = [
      'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'BRK-B', 'NVDA', 'TSLA', 'META',
      'UNH', 'JNJ', 'V', 'XOM', 'WMT', 'JPM', 'MA', 'PG', 'HD', 'BAC', 'PFE',
      'ABBV', 'KO', 'DIS', 'PEP', 'NFLX', 'CSCO', 'NKE', 'LLY', 'MCD', 'VZ',
      'MRK', 'TMO', 'CVX', 'ABT', 'CMCSA', 'DHR', 'AVGO', 'COST', 'ACN', 'NEE',
      'WFC', 'LIN', 'TXN', 'ADBE', 'MDT', 'HON', 'UNP', 'ORCL', 'PM'
  ]

  selected_stock = st.sidebar.selectbox("Select a Stock for DAX Variable:",
                                        sp500_top50)
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

with tab2:
  # Sidebar for user input in Tab 2
  st.sidebar.header("Optimal Portfolio Parameters")

  # Ticker selection from sp500_top50
  tickers = st.sidebar.multiselect("Select Tickers",
                                   sp500_top50,
                                   default=["AAPL", "MSFT", "GOOGL"])

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
    axs[0].set_title(f'{ticker[0]} vs {tikcer[1]}')

    # Scatter plot for Fresenius vs Commerz Bank
    axs[1].scatter(rets_fresenius, rets_commerz_bank, alpha=0.5)
    axs[1].plot(np.unique(rets_fresenius),
                np.poly1d(np.polyfit(rets_fresenius, rets_commerz_bank,
                                     1))(np.unique(rets_fresenius)),
                color='red')
    axs[1].set_title(f'{ticker[1]} vs {tikcer[2]}')

    # Scatter plot for Commerz Bank vs Deutsche Bank
    axs[2].scatter(rets_commerz_bank, rets_deutsche_bank, alpha=0.5)
    axs[2].plot(np.unique(rets_commerz_bank),
                np.poly1d(np.polyfit(rets_commerz_bank, rets_deutsche_bank,
                                     1))(np.unique(rets_commerz_bank)),
                color='red')
    axs[2].set_title(f'{ticker[1]} vs {tikcer[2]}')

    plt.tight_layout()
    st.pyplot(fig)

    # Calculate and display correlations
    correlation_fd = np.corrcoef(rets_fresenius, rets_deutsche_bank)[0, 1]
    correlation_fc = np.corrcoef(rets_fresenius, rets_commerz_bank)[0, 1]
    st.write(
        f'Correlation between {tickers[0]} and {tickers[1]}: {correlation_fd}')
    st.write(
        f'Correlation between {tickers[0]} and {tickers[2]} {correlation_fc}')

    # Function to estimate the covariance matrix of excess returns
    def estimate_sigma(in_sample_returns, risk_free_return):
        n_assets = in_sample_returns.shape[0]
        n_observations = in_sample_returns.shape[1]
        cov_matrix = np.zeros((n_assets, n_assets))

        for k in range(n_assets):
            for j in range(n_assets):
                cov_matrix[k, j] = np.mean((in_sample_returns[k, :] - risk_free_return) * (in_sample_returns[j, :] - risk_free_return))

        return cov_matrix

    # Parameters for the portfolio optimization
    risk_free_return = st.sidebar.slider("Risk-Free Return", 0.01, 0.05, 0.02)
    exp_rets = np.array([0.09, 0.09])
    s1, s2 = 0.16, 0.16
    rho = 0.5
    cov_mat = np.array([[s1, np.sqrt(s1 * s2) * rho],
                        [np.sqrt(s1 * s2) * rho, s2]])

    # Generate a large sample to estimate sigma
    N = 1000
    temp = multivariate_normal.rvs(mean=exp_rets, cov=cov_mat, size=N)
    sigma = estimate_sigma(temp.T, risk_free_return)

    # Optimal portfolio via Nekrasov's formula
    u = (1 + risk_free_return) * pinv(sigma) @ (exp_rets - risk_free_return)

    # Simulation of portfolio allocations
    path_len = 100
    sim_len = 1000
    rets = temp[:path_len]
    b = np.zeros((path_len, sim_len))
    wealth = np.zeros((path_len, sim_len))

    for i in range(sim_len):
      for j in range(path_len):
        b[j, i] = np.dot(u, rets[j]) + (1 - np.sum(u))
        wealth[j, i] = np.prod(b[:j + 1, i])

    # Create 3D scatter plot
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
    for i in range(100):
      fig.add_trace(
          go.Scatter3d(x=rets[:path_len, 0],
                       y=rets[:path_len, 1],
                       z=wealth[:, i],
                       mode='markers',
                       marker=dict(size=3),
                       name=f'Sim {i + 1}'))

    fig.update_layout(scene=dict(xaxis_title='Stock 1 Returns',
                                 yaxis_title='Stock 2 Returns',
                                 zaxis_title='Wealth'),
                      title="Wealth Simulation in 3D")

    st.plotly_chart(fig)