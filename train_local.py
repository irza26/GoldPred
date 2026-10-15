import os
import time
import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

# ==========================================
# 1. LOAD MASTER DATASET
# ==========================================
PARQUET_PATH = 'data/XAUUSD_2021_2026_1min_master.parquet'
MODEL_OUT_PATH = 'models/xauusd_m5_multihorizon_engine_2026.joblib'

print('Membaca master dataset...')
df = pd.read_parquet(PARQUET_PATH)

df_5m = (
    df.resample('5min')
    .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'})
    .dropna()
)

prices = df_5m['close'].values.astype(np.float64)
highs = df_5m['high'].values.astype(np.float64)
lows = df_5m['low'].values.astype(np.float64)
timestamps = df_5m.index

LOOKBACK = 24       # 2 Jam Lookback (24 x 5m)
HORIZON_60M = 12    # 60 Menit (12 Candle M5)

train_mask = timestamps < '2026-01-01'
test_mask = timestamps >= '2026-01-01'

train_indices = np.where(train_mask)[0]
test_indices = np.where(test_mask)[0]


# ==========================================
# 2. FEATURE ENGINEERING ENGINE
# ==========================================
def extract_robust_dataset(indices, stride=2):
    X, Y_bin_60, Y_reg_deltas = [], [], []

    for idx in range(LOOKBACK, len(indices) - HORIZON_60M, stride):
        c_idx = indices[idx]
        w_start = c_idx - LOOKBACK
        f_end = c_idx + HORIZON_60M

        win_close = prices[w_start:c_idx]
        win_high = highs[w_start:c_idx]
        win_low = lows[w_start:c_idx]
        p0 = win_close[-1]

        # Normalisasi Z-Score & Volatilitas
        w_mean = win_close.mean()
        w_std = win_close.std() if win_close.std() > 0 else 1e-6
        z_shape = (win_close - w_mean) / w_std

        atr_local = np.mean(win_high - win_low)
        atr_norm = atr_local / w_std
        volatility_ratio = w_std / w_mean

        mom_3 = (p0 - win_close[-3]) / w_std
        mom_6 = (p0 - win_close[-6]) / w_std
        mom_12 = (p0 - win_close[-12]) / w_std
        mom_24 = (p0 - win_close[0]) / w_std

        feat = np.hstack([
            z_shape,
            [atr_norm, volatility_ratio, mom_3, mom_6, mom_12, mom_24],
        ])

        # Target Biner (+60m > p0)
        act_60m = prices[c_idx + 12]
        label_60 = 1 if act_60m > p0 else 0

        # Target Delta Z-Score untuk Quantile Regressors
        fut_close = prices[c_idx:f_end]
        fut_z = (fut_close - w_mean) / w_std
        target_z_deltas = fut_z - z_shape[-1]

        X.append(feat)
        Y_bin_60.append(label_60)
        Y_reg_deltas.append(target_z_deltas)

    return np.array(X), np.array(Y_bin_60), np.array(Y_reg_deltas)


print('Membentuk Feature Matrix (Train & Test Out-of-Sample)...')
X_train, Y_tr_b60, Y_tr_deltas = extract_robust_dataset(train_indices, stride=2)
X_test, Y_te_b60, Y_te_deltas = extract_robust_dataset(test_indices, stride=1)


# ==========================================
# 3. TRAINING CLASSIFIER & QUANTILE REGRESSORS
# ==========================================
print('1. Melatih GBDT Classifier Engine (Direction & Confidence)...')
clf_60 = HistGradientBoostingClassifier(
    max_iter=200,
    learning_rate=0.02,
    max_leaf_nodes=45,
    min_samples_leaf=40,
    l2_regularization=1.0,
    random_state=42,
)
clf_60.fit(X_train, Y_tr_b60)

print('2. Melatih Quantile Risk Engine (Q10 & Q90 Envelopes)...')
models_q10, models_q90 = [], []
for step in range(HORIZON_60M):
    q10 = HistGradientBoostingRegressor(
        loss='quantile',
        quantile=0.10,
        max_iter=80,
        learning_rate=0.05,
        random_state=42,
    )
    q90 = HistGradientBoostingRegressor(
        loss='quantile',
        quantile=0.90,
        max_iter=80,
        learning_rate=0.05,
        random_state=42,
    )

    q10.fit(X_train, Y_tr_deltas[:, step])
    q90.fit(X_train, Y_tr_deltas[:, step])

    models_q10.append(q10)
    models_q90.append(q90)


# ==========================================
# 4. MICRO-WAVEFORM DYNAMIC FUNCTION
# ==========================================
def generate_dynamic_wave_trajectory(p0, bias, atr_local, horizon_steps=12):
    t = np.linspace(0, 1, horizon_steps)
    base_trend = (bias * 2) * atr_local * np.linspace(0.2, 1.8, horizon_steps)
    wave_component = np.sin(t * 2.5 * np.pi) * (atr_local * 0.35)
    damped_wave = wave_component * np.exp(-t * 0.3)
    return p0 + base_trend + damped_wave


# ==========================================
# 5. INFERENCE & EVALUASI KUANTITATIF (OUT-OF-SAMPLE 2026)
# ==========================================
print('\nMenghitung Evaluasi Kuantitatif pada Data Uji Murni 2026...')

probs_test = clf_60.predict_proba(X_test)[:, 1]
CONF_THRESHOLD = 54.0

high_conf_mask = (probs_test >= (CONF_THRESHOLD / 100.0)) | (
    probs_test <= (1.0 - CONF_THRESHOLD / 100.0)
)
valid_idx = np.where(high_conf_mask)[0]

horizons = ['1m', '5m', '10m', '15m', '30m', '60m']
horizon_indices = [0, 0, 1, 2, 5, 11]

eval_rows = []

for h_name, h_step in zip(horizons, horizon_indices):
    y_true_list, y_pred_list, p0_list, dir_pred_list = [], [], [], []

    for idx in valid_idx:
        c_idx = test_indices[idx * 1 + LOOKBACK]
        p0 = prices[c_idx]
        p_up = probs_test[idx]

        direction = 1 if p_up >= 0.5 else -1
        bias = (p_up - 0.5) if direction == 1 else (0.5 - p_up)

        win_high = highs[c_idx - LOOKBACK : c_idx]
        win_low = lows[c_idx - LOOKBACK : c_idx]
        atr_local = np.mean(win_high - win_low)

        wave_traj = generate_dynamic_wave_trajectory(p0, bias if direction == 1 else -bias, atr_local)

        if h_name == '1m':
            act_p = df.loc[timestamps[c_idx] :].iloc[1]['close']
        else:
            act_p = prices[c_idx + h_step + 1]

        y_true_list.append(act_p)
        y_pred_list.append(wave_traj[h_step])
        p0_list.append(p0)
        dir_pred_list.append(direction)

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)
    p0_arr = np.array(p0_list)
    dir_pred = np.array(dir_pred_list)

    dir_true = np.sign(y_true - p0_arr)
    acc = np.mean(dir_pred == dir_true) * 100
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    corr, _ = pearsonr(y_true - p0_arr, y_pred - p0_arr)

    eval_rows.append({
        'Horizon Window': f'+{h_name}',
        'Akurasi Arah (%)': f'{acc:.2f}%',
        'MAE ($ USD)': f'${mae:.2f}',
        'RMSE ($ USD)': f'${rmse:.2f}',
        'Pearson Correlation (r)': f'{corr:.4f}',
    })

df_eval = pd.DataFrame(eval_rows)

print('=' * 85)
print(f'MATRIKS EVALUASI DYNAMIC MICRO-WAVEFORM ENGINE (2026)')
print(f'Total Sinyal Teruji (Out-of-Sample 2026): {len(valid_idx)} Sinyal')
print('=' * 85)
print(df_eval.to_string(index=False))
print('=' * 85)


# ==========================================
# 6. SAVE JOBLIB PACKAGE
# ==========================================
export_bundle = {
    'model_engine': clf_60,
    'models_q10': models_q10,
    'models_q90': models_q90,
    'lookback_candles': LOOKBACK,
    'horizon_max_minutes': 60,
    'confidence_threshold_pct': CONF_THRESHOLD,
    'performance_metrics_2026': df_eval.to_dict(orient='records'),
}

os.makedirs('models', exist_ok=True)
joblib.dump(export_bundle, MODEL_OUT_PATH)

print(f'\n✅ MODEL ENGINE DYNAMIC MICRO-WAVEFORM RESMI DISIMPAN KE: "{MODEL_OUT_PATH}"')