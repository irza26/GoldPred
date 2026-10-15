from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

# Setup Path Lokal Projek
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "XAUUSD_2021_2026_1min_master.parquet"
MODEL_OUTPUT_PATH = (
    BASE_DIR / "models" / "xauusd_hybrid_m1_engine.joblib"
)

LOOKBACK_M1 = 120
HORIZON_STEPS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]


def extract_features(df):
  prices = df["close"].values.astype(np.float64)
  highs = df["high"].values.astype(np.float64)
  lows = df["low"].values.astype(np.float64)
  spreads = df["spread_mean"].values.astype(np.float64)
  timestamps = df.index

  train_mask = timestamps < "2026-01-01"
  train_indices = np.where(train_mask)[0]

  X, Y_b60, Y_b15, Y_deltas = [], [], [], []
  max_h = max(HORIZON_STEPS)

  print("Memproses ekstraksi fitur untuk data training...")
  for idx in range(LOOKBACK_M1, len(train_indices) - max_h, 5):
    c_idx = train_indices[idx]
    w_start = c_idx - LOOKBACK_M1

    win_close = prices[w_start:c_idx]
    win_high = highs[w_start:c_idx]
    win_low = lows[w_start:c_idx]
    win_spr = spreads[w_start:c_idx]
    p0 = win_close[-1]

    # Z-Score Shape & Microstructure Features
    w_mean = win_close.mean()
    w_std = win_close.std() if win_close.std() > 0 else 1e-6
    z_shape = (win_close - w_mean) / w_std

    atr_local = np.mean(win_high - win_low)
    atr_norm = atr_local / w_std
    spread_norm = win_spr[-1] / (win_spr.mean() + 1e-6)

    # Multi-Scale Momentum Alignment
    mom_5 = (p0 - win_close[-5]) / w_std
    mom_15 = (p0 - win_close[-15]) / w_std
    mom_30 = (p0 - win_close[-30]) / w_std
    mom_60 = (p0 - win_close[-60]) / w_std
    mom_120 = (p0 - win_close[0]) / w_std

    m_signs = np.sign([mom_5, mom_15, mom_30, mom_60])
    is_aligned = (
        1.0 if np.all(m_signs == m_signs[0]) and m_signs[0] != 0 else 0.0
    )

    feat = np.hstack([
        z_shape,
        [atr_norm, spread_norm, is_aligned],
        [mom_5, mom_15, mom_30, mom_60, mom_120],
    ])

    label_60 = 1 if prices[c_idx + 60] > p0 else 0
    label_15 = 1 if prices[c_idx + 15] > p0 else 0
    fut_targets = np.array([prices[c_idx + s] for s in HORIZON_STEPS])
    target_deltas = (fut_targets - p0) / w_std

    X.append(feat)
    Y_b60.append(label_60)
    Y_b15.append(label_15)
    Y_deltas.append(target_deltas)

  return np.array(X), np.array(Y_b60), np.array(Y_b15), np.array(Y_deltas)


def run_training():
  print(f"Membaca dataset lokal: {DATA_PATH}")
  if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"File Parquet tidak ditemukan di {DATA_PATH}! Taruh file di folder"
        " 'data/'."
    )

  df_m1 = pd.read_parquet(DATA_PATH)
  X_train, Y_tr_b60, Y_tr_b15, Y_tr_deltas = extract_features(df_m1)

  print("1/3. Melatih Classifier +60m...")
  clf_60m = HistGradientBoostingClassifier(
      max_iter=250,
      learning_rate=0.03,
      max_leaf_nodes=45,
      min_samples_leaf=40,
      random_state=42,
  )
  clf_60m.fit(X_train, Y_tr_b60)

  print("2/3. Melatih Classifier +15m...")
  clf_15m = HistGradientBoostingClassifier(
      max_iter=250,
      learning_rate=0.03,
      max_leaf_nodes=45,
      min_samples_leaf=40,
      random_state=42,
  )
  clf_15m.fit(X_train, Y_tr_b15)

  print("3/3. Melatih Quantile Risk Envelopes (Q10 & Q90)...")
  models_q10, models_q90 = [], []
  for step in range(len(HORIZON_STEPS)):
    q10 = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.10,
        max_iter=100,
        learning_rate=0.05,
        random_state=42,
    )
    q90 = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.90,
        max_iter=100,
        learning_rate=0.05,
        random_state=42,
    )
    q10.fit(X_train, Y_tr_deltas[:, step])
    q90.fit(X_train, Y_tr_deltas[:, step])
    models_q10.append(q10)
    models_q90.append(q90)

  # Export Model Bundle Ke Folder models/
  MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
  artifact_bundle = {
      "clf_60m": clf_60m,
      "clf_15m": clf_15m,
      "models_q10": models_q10,
      "models_q90": models_q90,
      "lookback_m1": LOOKBACK_M1,
      "horizon_steps": HORIZON_STEPS,
      "conf_threshold": 60.0,
  }

  joblib.dump(artifact_bundle, MODEL_OUTPUT_PATH)
  print(f"\nSELESAI! Model tersimpan di: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
  run_training()