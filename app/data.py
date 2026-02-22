import pandas as pd
import yfinance as yf


def _extract_close_series(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Return a normalized close-price Series from yfinance output."""
    if "Close" in data.columns:
        close_obj = data["Close"]
        if isinstance(close_obj, pd.Series):
            return close_obj
        if isinstance(close_obj, pd.DataFrame):
            if ticker in close_obj.columns:
                return close_obj[ticker]
            return close_obj.iloc[:, 0]

    if isinstance(data.columns, pd.MultiIndex):
        # Handle alternate level ordering from yfinance outputs.
        cols = data.columns
        if "Close" in cols.get_level_values(0):
            close_df = data.xs("Close", axis=1, level=0)
            if isinstance(close_df, pd.Series):
                return close_df
            if ticker in close_df.columns:
                return close_df[ticker]
            return close_df.iloc[:, 0]
        if "Close" in cols.get_level_values(-1):
            close_df = data.xs("Close", axis=1, level=-1)
            if isinstance(close_df, pd.Series):
                return close_df
            if ticker in close_df.columns:
                return close_df[ticker]
            return close_df.iloc[:, 0]

    raise KeyError("Close")


def download_monthly_prices(ticker: str, start_date, end_date) -> pd.DataFrame:
    data = yf.download(ticker, start=start_date, end=end_date, interval="1mo", auto_adjust=False)
    if data.empty:
        return pd.DataFrame()

    try:
        close_series = _extract_close_series(data, ticker)
    except KeyError:
        return pd.DataFrame()

    out = close_series.rename("Close").to_frame().reset_index()
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
