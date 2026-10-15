import numpy as np


def extract_features_single_window(df_120m):
  """Menerima DataFrame 120 bar M1 terakhir.

  Kolom: ['close', 'high', 'low', 'spread_mean']
  """
  prices = df_120m['close'].values.astype(np.float64)
  highs = df_120m['high'].values.astype(np.float64)
  lows = df_120m['low'].values.astype(np.float64)
  spreads = df_120m['spread_mean'].values.astype(np.float64)

  p0 = prices[-1]
  w_mean = prices.mean()
  w_std = prices.std() if prices.std() > 0 else 1e-6
  z_shape = (prices - w_mean) / w_std

  atr_local = np.mean(highs - lows)
  atr_norm = atr_local / w_std
  spread_norm = spreads[-1] / (spreads.mean() + 1e-6)

  mom_5 = (p0 - prices[-5]) / w_std
  mom_15 = (p0 - prices[-15]) / w_std
  mom_30 = (p0 - prices[-30]) / w_std
  mom_60 = (p0 - prices[-60]) / w_std
  mom_120 = (p0 - prices[0]) / w_std

  m_signs = np.sign([mom_5, mom_15, mom_30, mom_60])
  is_aligned = (
      1.0 if np.all(m_signs == m_signs[0]) and m_signs[0] != 0 else 0.0
  )

  feat = np.hstack([
      z_shape,
      [atr_norm, spread_norm, is_aligned],
      [mom_5, mom_15, mom_30, mom_60, mom_120],
  ])

  return feat.reshape(1, -1), p0, w_std, atr_local