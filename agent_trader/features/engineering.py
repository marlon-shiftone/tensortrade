from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["return_1"] = enriched["close"].pct_change().fillna(0.0)
    enriched["return_5"] = enriched["close"].pct_change(5).fillna(0.0)
    enriched["range_pct"] = ((enriched["high"] - enriched["low"]) / enriched["close"]).replace(
        [np.inf, -np.inf], 0.0
    )
    enriched["sma_10_gap"] = (enriched["close"] / enriched["close"].rolling(10).mean() - 1.0).fillna(0.0)
    enriched["sma_30_gap"] = (enriched["close"] / enriched["close"].rolling(30).mean() - 1.0).fillna(0.0)
    enriched["volatility_10"] = enriched["return_1"].rolling(10).std().fillna(0.0)
    enriched["volume_zscore_20"] = (
        (enriched["volume"] - enriched["volume"].rolling(20).mean()) /
        enriched["volume"].rolling(20).std()
    ).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return enriched.dropna().copy()
