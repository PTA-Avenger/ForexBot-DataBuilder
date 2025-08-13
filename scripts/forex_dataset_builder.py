import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class BuildConfig:
    tickers: List[str]
    period: str
    interval: str
    output_dir: str
    lookahead_days: int = 5
    rsi_window: int = 14
    ema_window: int = 10
    bb_window: int = 20
    zscore_window: int = 10
    zscore_threshold: float = 1.0
    offline: bool = False
    synthetic_rows: int = 500
    n_workers: int = 8
    force: bool = False
    file_format: str = "csv"  # csv | parquet


def ensure_dirs(base_output_dir: str) -> dict:
    raw_dir = os.path.join(base_output_dir, "raw")
    processed_dir = os.path.join(base_output_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    return {"raw": raw_dir, "processed": processed_dir}


def _save_df(df: pd.DataFrame, path: str, file_format: str) -> None:
    if file_format == "csv":
        df.to_csv(path, index=False)
    elif file_format == "parquet":
        try:
            df.to_parquet(path, index=False)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Parquet output requires pyarrow or fastparquet. Install one or use --file-format csv.") from exc
    else:
        raise ValueError("Unsupported file format: %s" % file_format)


def download_price_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("yfinance is required. Install it via pip install yfinance") from exc

    hist = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if hist.empty:
        raise ValueError(f"No data returned for ticker {ticker} with period={period} interval={interval}")
    hist = hist.reset_index().rename(columns={"Date": "date"})
    hist["ticker"] = ticker
    return hist


def generate_synthetic_history(ticker: str, rows: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=rows, freq="B")
    returns = rng.normal(loc=0.0002, scale=0.007, size=rows)
    prices = 1.0 * (1 + pd.Series(returns)).cumprod() * 100
    highs = prices * (1 + np.abs(rng.normal(0, 0.003, size=rows)))
    lows = prices * (1 - np.abs(rng.normal(0, 0.003, size=rows)))
    opens = prices.shift(1).fillna(prices.iloc[0])
    volumes = rng.integers(1_000_000, 5_000_000, size=rows)
    df = pd.DataFrame(
        {
            "date": dates,
            "Open": opens.values,
            "High": highs.values,
            "Low": lows.values,
            "Close": prices.values,
            "Adj Close": prices.values,
            "Volume": volumes,
        }
    )
    df["ticker"] = ticker
    return df


def compute_features_for_symbol(df: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    try:
        import ta
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ta is required. Install it via pip install ta") from exc

    df = df.sort_values("date").copy()

    df["return_1d"] = df["Close"].pct_change()
    df["log_return_1d"] = np.log1p(df["return_1d"].fillna(0.0))

    rsi = ta.momentum.RSIIndicator(close=df["Close"], window=cfg.rsi_window)
    df["rsi"] = rsi.rsi()

    df["ema"] = df["Close"].ewm(span=cfg.ema_window, adjust=False).mean()
    df["ema_dist"] = (df["Close"] - df["ema"]) / df["ema"]

    macd = ta.trend.MACD(close=df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close=df["Close"], window=cfg.bb_window)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_pct"] = (df["Close"] - df["bb_low"]) / (df["bb_high"] - df["bb_low"])

    df["fwd_return_1d"] = df["Close"].pct_change().shift(-1)
    df[f"fwd_return_{cfg.lookahead_days}d"] = (
        df["Close"].pct_change(cfg.lookahead_days).shift(-cfg.lookahead_days)
    )

    rolling_mean = df["Close"].rolling(cfg.zscore_window).mean()
    rolling_std = df["Close"].rolling(cfg.zscore_window).std()
    df["zscore"] = (df["Close"] - rolling_mean) / rolling_std

    return df


def compute_technical_features(df: pd.DataFrame, cfg: BuildConfig, n_workers: Optional[int] = None) -> pd.DataFrame:
    # Compute per symbol to avoid cross-series leakage and enable parallelism
    groups = [g for _, g in df.groupby("ticker", sort=False)]
    if n_workers and n_workers > 1 and len(groups) > 1:
        results: List[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(compute_features_for_symbol, g, cfg): g["ticker"].iloc[0] for g in groups}
            for fut in as_completed(futures):
                results.append(fut.result())
        feat = pd.concat(results, ignore_index=True)
    else:
        feat = pd.concat([compute_features_for_symbol(g, cfg) for g in groups], ignore_index=True)
    return feat


def build_trend_dataset(feat_df: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    df = feat_df.copy()
    df["target"] = (df[f"fwd_return_{cfg.lookahead_days}d"] > 0).astype(int)
    feature_cols = [
        "return_1d",
        "log_return_1d",
        "rsi",
        "ema_dist",
        "macd",
        "macd_signal",
        "macd_diff",
        "bb_pct",
        "zscore",
    ]
    cols = ["date", "ticker", "Close"] + feature_cols + ["target"]
    df = df[cols].dropna().reset_index(drop=True)
    return df


def build_meanrev_dataset(feat_df: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    df = feat_df.copy()
    extreme_mask = df["zscore"].abs() >= cfg.zscore_threshold
    df = df.loc[extreme_mask].copy()
    df["target"] = ((df["zscore"] > 0) & (df["fwd_return_1d"] < 0) | (df["zscore"] < 0) & (df["fwd_return_1d"] > 0)).astype(int)
    feature_cols = [
        "return_1d",
        "log_return_1d",
        "rsi",
        "ema_dist",
        "macd",
        "macd_signal",
        "macd_diff",
        "bb_pct",
        "zscore",
    ]
    cols = ["date", "ticker", "Close"] + feature_cols + ["target"]
    df = df[cols].dropna().reset_index(drop=True)
    return df


def build_sentiment_stub(processed_dir: str, file_format: str) -> str:
    sentiment_path = os.path.join(processed_dir, f"sentiment.{file_format}")
    df = pd.DataFrame(columns=["date", "ticker", "text", "label"])
    _save_df(df, sentiment_path, file_format)
    return sentiment_path


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Build Forex datasets (trend, mean-reversion, sentiment stub)")
    parser.add_argument("--tickers", type=str, default="EURUSD=X,GBPUSD=X,USDJPY=X,AUDUSD=X", help="Comma-separated tickers")
    parser.add_argument("--period", type=str, default="2y", help="History period for yfinance (e.g., 1y, 2y, 5y, max)")
    parser.add_argument("--interval", type=str, default="1d", help="Candle interval for yfinance (e.g., 1d, 1h, 15m)")
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"), help="Base output directory for data")
    parser.add_argument("--lookahead-days", type=int, default=5, help="Lookahead period for trend target")
    parser.add_argument("--offline", action="store_true", help="Generate synthetic data instead of downloading with yfinance")
    parser.add_argument("--synthetic-rows", type=int, default=500, help="Number of rows per ticker when using --offline")
    parser.add_argument("--n-workers", type=int, default=8, help="Number of worker threads for download/feature computation")
    parser.add_argument("--force", action="store_true", help="Force re-download/rebuild even if raw files exist")
    parser.add_argument("--file-format", type=str, choices=["csv", "parquet"], default="csv", help="Output file format for processed datasets")

    args = parser.parse_args(argv)

    cfg = BuildConfig(
        tickers=[t.strip() for t in args.tickers.split(",") if t.strip()],
        period=args.period,
        interval=args.interval,
        output_dir=os.path.abspath(args.output_dir),
        lookahead_days=args.lookahead_days,
        offline=args.offline,
        synthetic_rows=args.synthetic_rows,
        n_workers=max(1, int(args.n_workers)),
        force=bool(args.force),
        file_format=str(args.file_format),
    )

    dirs = ensure_dirs(cfg.output_dir)

    def build_one_symbol(ticker: str) -> pd.DataFrame:
        raw_path = os.path.join(dirs["raw"], f"{ticker.replace('=','_')}_{cfg.interval}.csv")
        if not cfg.force and os.path.exists(raw_path):
            print(f"[dataset-builder] Using cached raw for {ticker}: {raw_path}")
            df = pd.read_csv(raw_path, parse_dates=["date"])
            df["ticker"] = ticker
            return df
        if cfg.offline:
            print(f"[dataset-builder] Generating synthetic data for {ticker} rows={cfg.synthetic_rows}...")
            hist = generate_synthetic_history(ticker, cfg.synthetic_rows)
        else:
            print(f"[dataset-builder] Downloading {ticker} period={cfg.period} interval={cfg.interval}...")
            hist = download_price_history(ticker, cfg.period, cfg.interval)
        hist.to_csv(raw_path, index=False)
        print(f"[dataset-builder] Saved raw to {raw_path} ({len(hist)} rows)")
        return hist

    # Parallel data acquisition
    all_hist: List[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=cfg.n_workers) as ex:
        futures = {ex.submit(build_one_symbol, ticker): ticker for ticker in cfg.tickers}
        for fut in as_completed(futures):
            all_hist.append(fut.result())

    full = pd.concat(all_hist, ignore_index=True)

    print("[dataset-builder] Computing technical features...")
    feat = compute_technical_features(full, cfg, n_workers=cfg.n_workers)

    print("[dataset-builder] Building trend dataset...")
    trend_df = build_trend_dataset(feat, cfg)
    trend_path = os.path.join(dirs["processed"], f"trend_dataset.{cfg.file_format}")
    _save_df(trend_df, trend_path, cfg.file_format)
    print(f"[dataset-builder] Wrote {trend_path} ({len(trend_df)} rows)")

    print("[dataset-builder] Building mean-reversion dataset...")
    meanrev_df = build_meanrev_dataset(feat, cfg)
    meanrev_path = os.path.join(dirs["processed"], f"meanrev_dataset.{cfg.file_format}")
    _save_df(meanrev_df, meanrev_path, cfg.file_format)
    print(f"[dataset-builder] Wrote {meanrev_path} ({len(meanrev_df)} rows)")

    sentiment_path = build_sentiment_stub(dirs["processed"], cfg.file_format)
    print(f"[dataset-builder] Wrote sentiment stub {sentiment_path}")

    print("[dataset-builder] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
