"""Download and clean current NSE prices for the BTP universe."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import jarque_bera

from config import (
    BENCHMARK_NAME,
    BENCHMARK_TICKER,
    DATA_DIR,
    END_DATE,
    MIN_HISTORY_FRAC,
    NSE_UNIVERSE,
    OUTPUT_DIR,
    START_DATE,
)


@dataclass
class MarketData:
    prices: pd.DataFrame
    returns: pd.DataFrame
    benchmark_prices: pd.Series
    benchmark_returns: pd.Series
    betas: pd.Series
    meta: pd.DataFrame
    start: str
    end: str
    ohlcv: dict[str, pd.DataFrame] | None = None

    @property
    def tickers(self) -> list[str]:
        return list(self.returns.columns)


def _extract_field(raw: pd.DataFrame, field: str) -> pd.DataFrame:
    """yfinance may use (Price, Ticker) or (Ticker, Price) columns."""
    if not isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns:
            return raw[[field]].copy()
        raise KeyError(f"{field} not in download columns: {list(raw.columns)}")

    level0 = set(raw.columns.get_level_values(0))
    level1 = set(raw.columns.get_level_values(1))
    if field in level0:
        out = raw[field].copy()
    elif field in level1:
        out = raw.xs(field, axis=1, level=1).copy()
    else:
        raise KeyError(f"{field} not in MultiIndex columns: {raw.columns.tolist()[:8]}")
    if isinstance(out, pd.Series):
        out = out.to_frame()
    return out


def download_ohlcv(
    tickers: list[str],
    start: str = START_DATE,
    end: str | None = END_DATE,
) -> pd.DataFrame:
    end = end or date.today().isoformat()
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True,
        threads=False,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no rows. Check network or tickers.")
    return raw


def _coverage_ok(series: pd.Series, min_frac: float) -> bool:
    if series.empty:
        return False
    return float(series.notna().mean()) >= min_frac


def build_market_data(
    start: str = START_DATE,
    end: str | None = END_DATE,
    universe: dict | None = None,
    benchmark: str = BENCHMARK_TICKER,
    min_history_frac: float = MIN_HISTORY_FRAC,
) -> MarketData:
    universe = universe or NSE_UNIVERSE
    tickers = list(universe.keys())
    end = end or date.today().isoformat()

    raw = download_ohlcv(tickers + [benchmark], start=start, end=end)
    close = _extract_field(raw, "Close")

    missing = [t for t in tickers + [benchmark] if t not in close.columns]
    if missing:
        raise RuntimeError(f"Missing close prices for: {missing}")

    asset_px = close[tickers].replace(0, np.nan)
    bench_px = close[benchmark].replace(0, np.nan)

    keep = [t for t in tickers if _coverage_ok(asset_px[t], min_history_frac)]
    dropped = sorted(set(tickers) - set(keep))
    if dropped:
        print(f"Dropped short-history names: {dropped}")
    if not keep:
        raise RuntimeError("No NSE name passed the history filter.")

    aligned = pd.concat([asset_px[keep], bench_px.rename(benchmark)], axis=1).dropna(how="any")
    if aligned.empty:
        raise RuntimeError("No overlapping NSE trading days after alignment.")

    prices = aligned[keep]
    benchmark_prices = aligned[benchmark]
    ohlcv = {}
    for field in ("Open", "High", "Low", "Volume"):
        try:
            panel = _extract_field(raw, field)[keep].reindex(aligned.index)
            ohlcv[field] = panel
        except (KeyError, ValueError):
            continue
    ohlcv["Close"] = prices
    returns = prices.pct_change().dropna()
    benchmark_returns = benchmark_prices.pct_change().dropna()
    returns, benchmark_returns = returns.align(benchmark_returns, join="inner", axis=0)

    bench_var = float(benchmark_returns.var(ddof=1))
    betas = pd.Series(
        {t: float(returns[t].cov(benchmark_returns) / bench_var) for t in keep},
        name="beta",
    )

    rows = []
    for t in keep:
        info = universe.get(t, {})
        px = prices[t]
        r = returns[t]
        rows.append(
            {
                "ticker": t,
                "name": info.get("name", t),
                "sector": info.get("sector", "Unknown"),
                "last_price": float(px.iloc[-1]),
                "last_date": px.index[-1].date().isoformat(),
                "n_days": int(r.shape[0]),
                "ann_return": float((1 + r.mean()) ** 252 - 1),
                "ann_vol": float(r.std(ddof=1) * np.sqrt(252)),
                "beta": float(betas[t]),
            }
        )
    meta = pd.DataFrame(rows)

    print(
        f"NSE sample {returns.index.min().date()} → {returns.index.max().date()} | "
        f"{returns.shape[0]} days × {returns.shape[1]} names | benchmark {BENCHMARK_NAME}"
    )
    ohlcv = {k: v.loc[returns.index] for k, v in ohlcv.items()}
    return MarketData(
        prices=prices.loc[returns.index],
        returns=returns,
        benchmark_prices=benchmark_prices.loc[returns.index],
        benchmark_returns=benchmark_returns,
        betas=betas,
        meta=meta,
        start=returns.index.min().date().isoformat(),
        end=returns.index.max().date().isoformat(),
        ohlcv=ohlcv,
    )


def jarque_bera_table(returns: pd.DataFrame, meta: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for col in returns.columns:
        series = returns[col].dropna()
        stat, p_value = jarque_bera(series)
        name = col
        sector = ""
        if meta is not None and col in set(meta["ticker"]):
            rec = meta.loc[meta["ticker"] == col].iloc[0]
            name = rec["name"]
            sector = rec["sector"]
        rows.append(
            {
                "ticker": col,
                "name": name,
                "sector": sector,
                "mean": float(series.mean()),
                "std": float(series.std(ddof=1)),
                "skew": float(series.skew()),
                "kurtosis": float(series.kurtosis() + 3.0),
                "jb_stat": float(stat),
                "p_value": float(p_value),
                "reject_normal_5pct": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows)


def load_market_cache() -> MarketData | None:
    price_path = DATA_DIR / "nse_prices.csv"
    ret_path = DATA_DIR / "nse_returns.csv"
    bench_path = DATA_DIR / "nifty50_returns.csv"
    meta_path = DATA_DIR / "universe.csv"
    beta_path = DATA_DIR / "betas.csv"
    if not all(p.exists() for p in (price_path, ret_path, bench_path, meta_path, beta_path)):
        return None
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
    returns = pd.read_csv(ret_path, index_col=0, parse_dates=True)
    bench = pd.read_csv(bench_path, index_col=0, parse_dates=True).iloc[:, 0]
    meta = pd.read_csv(meta_path)
    betas = pd.read_csv(beta_path, index_col=0).iloc[:, 0]
    bench_px = (1 + bench).cumprod()
    ohlcv = {"Close": prices.loc[returns.index]}
    for field in ("Open", "High", "Low", "Volume"):
        path = DATA_DIR / f"nse_{field.lower()}.csv"
        if path.exists():
            ohlcv[field] = pd.read_csv(path, index_col=0, parse_dates=True).reindex(returns.index)
    return MarketData(
        prices=prices,
        returns=returns,
        benchmark_prices=bench_px,
        benchmark_returns=bench,
        betas=betas,
        meta=meta,
        start=returns.index.min().date().isoformat(),
        end=returns.index.max().date().isoformat(),
        ohlcv=ohlcv,
    )


def save_market_cache(data: MarketData) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data.prices.to_csv(DATA_DIR / "nse_prices.csv")
    data.returns.to_csv(DATA_DIR / "nse_returns.csv")
    data.benchmark_returns.to_csv(DATA_DIR / "nifty50_returns.csv")
    data.meta.to_csv(DATA_DIR / "universe.csv", index=False)
    data.betas.to_csv(DATA_DIR / "betas.csv")
    if data.ohlcv:
        for field, panel in data.ohlcv.items():
            if field == "Close":
                continue
            panel.to_csv(DATA_DIR / f"nse_{field.lower()}.csv")


def get_data(start: str = START_DATE, end: str | None = END_DATE) -> tuple:
    """Backwards-compatible tuple used by the original CLI/app."""
    data = build_market_data(start=start, end=end)
    jb = jarque_bera_table(data.returns, data.meta)
    save_market_cache(data)
    jb.to_csv(OUTPUT_DIR / "jarque_bera.csv", index=False)
    return data.returns, data.benchmark_returns, data.betas.to_dict(), {}, jb


if __name__ == "__main__":
    market = build_market_data()
    jb = jarque_bera_table(market.returns, market.meta)
    save_market_cache(market)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jb.to_csv(OUTPUT_DIR / "jarque_bera.csv", index=False)
    print(market.meta.to_string(index=False))
    print("\nJarque–Bera")
    print(jb.to_string(index=False))
    print("\nData pipeline completed.")
