import os
import joblib
import numpy as np
from src.feature_engineering import extract_features_from_m5_slice


class XAUUSDInferenceEngine:

    def __init__(self, model_path='models/xauusd_m5_multihorizon_engine_2026.joblib'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Model tidak ditemukan: {model_path}')

        self.bundle = joblib.load(model_path)
        self.clf_model = self.bundle['model_engine']
        self.models_q10 = self.bundle['models_q10']
        self.models_q90 = self.bundle['models_q90']
        self.lookback = self.bundle['lookback_candles']
        self.threshold = self.bundle['confidence_threshold_pct']

    def predict(self, df_m5_slice):
        feat_vector, p0, atr_local, w_std = extract_features_from_m5_slice(
            df_m5_slice, lookback=self.lookback
        )

        # 1. GBDT Classifier (Direction & Confidence)
        prob_up = self.clf_model.predict_proba(feat_vector)[0, 1]

        if prob_up >= 0.50:
            pred_direction = 'BUY'
            confidence_pct = prob_up * 100
            bias = prob_up - 0.50
        else:
            pred_direction = 'SELL'
            confidence_pct = (1.0 - prob_up) * 100
            bias = -(0.50 - prob_up)

        is_high_confidence = confidence_pct >= self.threshold

        # 2. Dynamic Micro-Waveform Trajectory (Sinusoidal Swings)
        horizon_steps = 12
        t = np.linspace(0, 1, horizon_steps)

        # Base Trend Vector (GBDT Magnitude)
        dir_mult = 1.0 if pred_direction == 'BUY' else -1.0
        base_trend = (bias * 2) * atr_local * np.linspace(0.2, 1.8, horizon_steps) * dir_mult

        # Micro-Oscillation (Siklus Swing M5)
        wave_component = np.sin(t * 2.5 * np.pi) * (atr_local * 0.35)
        damped_wave = wave_component * np.exp(-t * 0.3)

        expected_trajectory = p0 + base_trend + damped_wave

        # 3. Quantile Risk Envelopes Q10 - Q90
        q10_deltas = np.array([
            m.predict(feat_vector)[0] for m in self.models_q10
        ])
        q90_deltas = np.array([
            m.predict(feat_vector)[0] for m in self.models_q90
        ])

        q10_bounds = p0 + (q10_deltas * w_std)
        q90_bounds = p0 + (q90_deltas * w_std)

        steps_indices = [0, 0, 1, 2, 5, 11]
        steps_labels = ['+1m', '+5m', '+10m', '+15m', '+30m', '+60m']

        horizon_projections = {}
        for lbl, idx in zip(steps_labels, steps_indices):
            horizon_projections[lbl] = {
                'pred': expected_trajectory[idx],
                'q10': q10_bounds[idx],
                'q90': q90_bounds[idx],
            }

        return {
            'timestamp': str(df_m5_slice.index[-1]),
            'anchor_p0': float(p0),
            'atr': float(atr_local),
            'direction': pred_direction,
            'confidence_pct': float(confidence_pct),
            'is_high_confidence': bool(is_high_confidence),
            'projections': horizon_projections,
            'full_traj': expected_trajectory,
            'full_q10': q10_bounds,
            'full_q90': q90_bounds,
        }