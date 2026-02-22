import pandas as pd

from app.data import download_monthly_prices


def test_download_monthly_prices_handles_empty(monkeypatch):
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("app.data.yf.download", fake_download)
    out = download_monthly_prices("AAPL", "2020-01-01", "2020-12-31")
    assert out.empty


def test_download_monthly_prices_handles_multiindex_columns(monkeypatch):
    idx = pd.date_range("2020-01-31", periods=3, freq="M")
    cols = pd.MultiIndex.from_tuples(
        [("Close", "AAPL"), ("Open", "AAPL")],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame([[100.0, 99.0], [105.0, 104.0], [102.0, 101.0]], index=idx, columns=cols)

    def fake_download(*args, **kwargs):
        return raw

    monkeypatch.setattr("app.data.yf.download", fake_download)
    out = download_monthly_prices("AAPL", "2020-01-01", "2020-12-31")

    assert not out.empty
    assert "Close" in out.columns
    assert "monthly_returns" in out.columns
