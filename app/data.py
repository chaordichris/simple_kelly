import pandas as pd
import yfinance as yf


def download_monthly_prices(ticker: str, start_date, end_date) -> pd.DataFrame:
    data = yf.download(ticker, start=start_date, end=end_date, interval="1mo", auto_adjust=False)
    if data.empty:
        return pd.DataFrame()

    out = data.reset_index().copy()
    if "Close" not in out.columns:
        return pd.DataFrame()

    out["monthly_returns"] = out["Close"].pct_change()
    out = out.dropna(subset=["monthly_returns", "Close"])
    return out


def download_close_prices(tickers, start_date, end_date) -> pd.DataFrame:
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    if data.empty:
        return pd.DataFrame()

    if "Close" not in data:
        return pd.DataFrame()

    close_data = data["Close"].copy()
    if isinstance(close_data, pd.Series):
        close_data = close_data.to_frame(name=tickers[0] if isinstance(tickers, list) and tickers else "Close")

    return close_data.dropna(how="all")
