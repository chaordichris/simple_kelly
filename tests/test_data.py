import pandas as pd

from app.data import download_monthly_prices


def test_download_monthly_prices_handles_empty(monkeypatch):
    def fake_download(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr("app.data.yf.download", fake_download)
    out = download_monthly_prices("AAPL", "2020-01-01", "2020-12-31")
    assert out.empty
