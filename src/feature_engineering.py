import numpy as np
import pandas as pd


def extract_features_from_m5_slice(df_m5_slice, lookback=24):
    """Menerima slice DataFrame minimal 24 candle M5 terakhir (OHLC)

    dan mengembalikan feature vector siap-inferensi (1, N_features).
    """
    prices = df_m5_slice['close'].values.astype(np.float64)
    highs = df_m5_slice['high'].values.astype(np.float64)
    lows = df_m5_slice['low'].values.astype(np.float64)

    if len(prices) < lookback:
        raise ValueError(
            f'Dibutuhkan minimal {lookback} candle M5 untuk feature engineering!'
        )

    # Ambil 24 candle terakhir
    win_close = prices[-lookback:]
    win_high = highs[-lookback:]
    win_low = lows[-lookback:]
    p0 = win_close[-1]

    # Z-Score Normalization
    w_mean = win_close.mean()
    w_std = win_close.std() if win_close.std() > 0 else 1e-6
    z_shape = (win_close - w_mean) / w_std

    # Volatility & Momentum Features
    atr_local = np.mean(win_high - win_low)
    atr_norm = atr_local / w_std
    volatility_ratio = w_std / w_mean

    mom_3 = (p0 - win_close[-3]) / w_std
    mom_6 = (p0 - win_close[-6]) / w_std
    mom_12 = (p0 - win_close[-12]) / w_std
    mom_24 = (p0 - win_close[0]) / w_std

    feat_vector = np.hstack([
        z_shape,
        [atr_norm, volatility_ratio, mom_3, mom_6, mom_12, mom_24],
    ]).reshape(1, -1)

    return feat_vector, p0, atr_local, w_std