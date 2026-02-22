import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

try:
    from app.data import download_monthly_prices, download_close_prices
    from app.kelly import expected_log_growth, simulate_binary_wealth_paths
    from app.models import KellyInputs, SimulationConfig
    from app.sim import simulate_kelly_paths
except ModuleNotFoundError:
    # Support `streamlit run app/app.py`, where `app.py` shadows the package name.
    from data import download_monthly_prices, download_close_prices
    from kelly import expected_log_growth, simulate_binary_wealth_paths
    from models import KellyInputs, SimulationConfig
    from sim import simulate_kelly_paths


@st.cache_data(show_spinner=False)
def cached_download_monthly_prices(ticker, start_date, end_date):
    return download_monthly_prices(ticker, start_date, end_date)


@st.cache_data(show_spinner=False)
def cached_download_close_prices(tickers, start_date, end_date):
    return download_close_prices(tickers, start_date, end_date)


@st.cache_data(show_spinner=False)
def cached_simulate_kelly_paths(config: SimulationConfig):
    return simulate_kelly_paths(config)


st.title("Simple Kelly Criterion")

st.sidebar.header("Select exactly 3 stocks for your portfolio")

sp500_top50 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "BRK-B", "NVDA", "TSLA", "META",
    "UNH", "JNJ", "V", "XOM", "WMT", "JPM", "MA", "PG", "HD", "BAC", "PFE",
    "ABBV", "KO", "DIS", "PEP", "NFLX", "CSCO", "NKE", "LLY", "MCD", "VZ",
    "MRK", "TMO", "CVX", "ABT", "CMCSA", "DHR", "AVGO", "COST", "ACN", "NEE",
    "WFC", "LIN", "TXN", "ADBE", "MDT", "HON", "UNP", "ORCL", "PM",
]

tickers = st.sidebar.multiselect(
    "Select Tickers",
    sp500_top50,
    default=["AAPL", "MSFT", "GOOGL"],
    max_selections=3,
)

if len(tickers) != 3:
    st.sidebar.error("Please select exactly 3 tickers.")
    tickers = ["AAPL", "MSFT", "GOOGL"]

ticker1, ticker2, ticker3 = tickers

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2013-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp.today().date())

if start_date >= end_date:
    st.sidebar.error("Start Date must be before End Date.")

tab1, tab2, tab3 = st.tabs(
    [
        "Kelly Optimal Betting Fraction",
        "Optimal Kelly Fraction Simulation",
        "Visualizing Correlated Stocks",
    ]
)

with tab1:
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

    param1 = st.slider("Parameter 1 (Growth multiplier)", min_value=0.1, max_value=3.0, value=1.5, step=0.05)
    param2 = st.slider("Parameter 2 (Loss multiplier)", min_value=0.05, max_value=1.0, value=0.5, step=0.05)
    param3 = st.slider("Parameter 3 (Probability of Success)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)

    u = np.linspace(0, 1, num=100)
    xgr = expected_log_growth(u, param3, param1, param2)
    df = pd.DataFrame({"perc_capital": u, "expected_growth": xgr})

    st.subheader("Expected Growth Rate vs Capital Fraction")
    fig, ax = plt.subplots()
    sns.lineplot(data=df, x="perc_capital", y="expected_growth", ax=ax)
    plt.xlabel("Percentage of Capital")
    plt.ylabel("Expected Growth Rate")
    st.pyplot(fig)

    param4 = st.slider("Parameter 4 (Number of Periods)", min_value=10, max_value=1000, value=50, step=10)
    param5 = st.slider("Parameter 5 (Percent Kelly to Compare)", min_value=0.05, max_value=1.0, value=0.5, step=0.05)

    kelly_inputs = KellyInputs(
        prob_success=param3,
        gain_multiplier=param1,
        loss_multiplier=param2,
        n_periods=param4,
        fractional_kelly=param5,
    )
    wealth_all, wealth_k, wealth_fk, _ = simulate_binary_wealth_paths(kelly_inputs)

    wealth_df = pd.DataFrame(
        {
            "Invest Everything": wealth_all,
            "Invest Kelly Optimal Amount": wealth_k,
            f"Invest {param5 * 100:.0f}% of Kelly Optimal Amount": wealth_fk,
            "periods": np.arange(param4),
        }
    )

    dfm = wealth_df.melt("periods", var_name="wealth_fraction", value_name="wealth")
    st.subheader("Expected Wealth w/ Different Kelly Fractions")
    fig2, ax2 = plt.subplots()
    sns.lineplot(data=dfm, x="periods", y="wealth", hue="wealth_fraction", ax=ax2)
    plt.xlabel("Investment Periods")
    plt.ylabel("Expected Wealth")
    st.pyplot(fig2)

with tab2:
    st.write(
        """
  ## Kelly Fraction Simulation
  Simulate the log wealth of different kelly fractions for a given stock and risk free rate.
  Select a stock, a risk free rate, periods (months), simulations, and kelly steps to view simulation results.
  """
    )

    selected_stock = st.selectbox("Select A Ticker", tickers)

    mu = None
    sigma = None
    stock_df = None

    if selected_stock and start_date < end_date:
        stock_df = cached_download_monthly_prices(selected_stock, start_date, end_date)
        if stock_df.empty:
            st.warning("No monthly data returned for the selected ticker/date range.")
        else:
            st.table(stock_df.head())
            mu = stock_df["monthly_returns"].mean()
            sigma = stock_df["monthly_returns"].std()

            if pd.isna(mu) or pd.isna(sigma) or sigma <= 0:
                st.warning("Not enough valid return observations to estimate simulation parameters.")
            else:
                normal_returns = np.random.normal(loc=mu, scale=sigma, size=1000)
                col11, col12 = st.columns(2)
                with col11:
                    fig11, ax11 = plt.subplots()
                    sns.lineplot(x=stock_df["Date"], y=stock_df["Close"], ax=ax11)
                    ax11.set_title("Monthly Close Price")
                    ax11.set_xlabel("Month")
                    ax11.set_ylabel("Price at Close")
                    plt.xticks(rotation=45)
                    st.pyplot(fig11)
                with col12:
                    fig12, ax12 = plt.subplots()
                    sns.histplot(stock_df["monthly_returns"], ax=ax12, kde=False, stat="density", bins=30)
                    sns.kdeplot(normal_returns, ax=ax12, color="red", label="Normal Approximation")
                    ax12.set_title("Distribution of Monthly Returns")
                    ax12.set_xlabel("Monthly Returns")
                    ax12.set_ylabel("Density")
                    st.pyplot(fig12)

    risk_free_rate = st.slider("Risk Free Rate", min_value=0.0, max_value=0.15, value=0.03, step=0.01)
    n_simulations = st.slider("Number of Simulations", min_value=100, max_value=5000, value=1000)
    n_months = st.slider("Number of Months", min_value=10, max_value=120, value=30)
    n_steps = st.slider("Number of Kelly Steps", min_value=5, max_value=50, value=25)

    run_simulation = st.button("Run Simulation")

    if run_simulation:
        if mu is None or sigma is None or pd.isna(mu) or pd.isna(sigma) or sigma <= 0:
            st.error("Simulation requires valid return moments (mu, sigma). Select a ticker/range with enough data.")
        else:
            sim_config = SimulationConfig(
                mu=float(mu),
                sigma=float(sigma),
                risk_free_rate=risk_free_rate,
                n_simulations=n_simulations,
                n_months=n_months,
                n_steps=n_steps,
            )
            per_sim_df, averaged_df = cached_simulate_kelly_paths(sim_config)

            col1, col2 = st.columns(2)

            with col1:
                fig1, ax1 = plt.subplots()
                sns.lineplot(data=per_sim_df, x="month", y="log_wealth", hue="sim", ax=ax1)
                ax1.set_title("Kelly Fraction Simulation Results (Selected Kelly)")
                ax1.set_xlabel("Month")
                ax1.set_ylabel("Log Wealth")
                st.pyplot(fig1)

            with col2:
                fig3, ax3 = plt.subplots()
                sns.lineplot(data=averaged_df, x="month", y="log_wealth", hue="kelly", ax=ax3)
                ax3.set_title("Kelly Fraction Simulation Results (Averaged)")
                ax3.set_xlabel("Month")
                ax3.set_ylabel("Log Wealth")
                st.pyplot(fig3)

with tab3:
    st.header("Correlation Analysis")

    if start_date >= end_date:
        st.warning("Please choose a valid date range.")
    else:
        close_data = cached_download_close_prices(tickers, start_date, end_date)

        if close_data.empty or any(t not in close_data.columns for t in tickers):
            st.warning("Unable to load close prices for all selected tickers.")
        else:
            ret1 = close_data[tickers[0]].pct_change().dropna()
            ret2 = close_data[tickers[1]].pct_change().dropna()
            ret3 = close_data[tickers[2]].pct_change().dropna()

            correlation_12 = np.corrcoef(ret1, ret2)[0, 1]
            correlation_13 = np.corrcoef(ret1, ret3)[0, 1]
            correlation_23 = np.corrcoef(ret2, ret3)[0, 1]

            fig_corr, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].scatter(ret1, ret2, alpha=0.5)
            axs[0].plot(np.unique(ret1), np.poly1d(np.polyfit(ret1, ret2, 1))(np.unique(ret1)), color="red")
            axs[0].set_title(f"{ticker1} vs {ticker2} (Correlation: {correlation_12:.2f})")

            axs[1].scatter(ret1, ret3, alpha=0.5)
            axs[1].plot(np.unique(ret1), np.poly1d(np.polyfit(ret1, ret3, 1))(np.unique(ret1)), color="red")
            axs[1].set_title(f"{ticker1} vs {ticker3} (Correlation: {correlation_13:.2f})")

            axs[2].scatter(ret3, ret2, alpha=0.5)
            axs[2].plot(np.unique(ret3), np.poly1d(np.polyfit(ret3, ret2, 1))(np.unique(ret3)), color="red")
            axs[2].set_title(f"{ticker2} vs {ticker3} (Correlation: {correlation_23:.2f})")

            plt.tight_layout()
            st.pyplot(fig_corr)
