from pathlib import Path
import joblib
import numpy as np
from src.feature_engineering import extract_features_single_window


class XAUUSDInferenceEngine:

  def __init__(self, model_name='xauusd_hybrid_m1_engine.joblib'):
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / 'models' / model_name

    if not model_path.exists():
      raise FileNotFoundError(
          f"Model file tidak ditemukan di {model_path}. Jalankan"
          " 'python train_local.py' dulu!"
      )

    bundle = joblib.load(model_path)
    self.clf_60m = bundle['clf_60m']
    self.clf_15m = bundle['clf_15m']
    self.models_q10 = bundle['models_q10']
    self.models_q90 = bundle['models_q90']
    self.horizon_steps = bundle['horizon_steps']
    self.conf_threshold = bundle.get('conf_threshold', 60.0)

  def predict(self, df_120m):
    feat, p0, w_std, atr_local = extract_features_single_window(df_120m)

    prob_60 = self.clf_60m.predict_proba(feat)[0, 1]
    prob_15 = self.clf_15m.predict_proba(feat)[0, 1]

    # Threshold Check (>60% atau <40%)
    conf_pct = (prob_60 if prob_60 >= 0.5 else 1.0 - prob_60) * 100.0
    is_high_conf = conf_pct >= self.conf_threshold

    direction_60 = 1 if prob_60 >= 0.5 else -1

    signal_label = 'NEUTRAL / WAIT'
    if is_high_conf:
      signal_label = 'BUY' if direction_60 == 1 else 'SELL'

    pred_path, q10_path, q90_path = [], [], []

    for step_idx, h_step in enumerate(self.horizon_steps):
      t_norm = h_step / 60.0
      blended_dir = (
          direction_60
          if h_step > 20
          else (1 if prob_15 >= 0.5 else -1)
      )
      blended_bias = (
          np.abs(prob_60 - 0.5) if h_step > 20 else np.abs(prob_15 - 0.5)
      )

      macro_slope = (
          (blended_bias * 6.0)
          * atr_local
          * (t_norm**1.2)
          * blended_dir
      )
      wave_comp = (
          (atr_local * 0.40)
          * np.sin(t_norm * 2.5 * np.pi)
          * blended_dir
      )

      step_delta = (
          (macro_slope + wave_comp) if is_high_conf else (wave_comp * 0.2)
      )
      pred_path.append(p0 + step_delta)

      q10_delta = self.models_q10[step_idx].predict(feat)[0] * w_std
      q90_delta = self.models_q90[step_idx].predict(feat)[0] * w_std
      q10_path.append(p0 + q10_delta)
      q90_path.append(p0 + q90_delta)

    return {
        'anchor_price': p0,
        'signal': signal_label,
        'confidence_pct': conf_pct,
        'is_high_conf': is_high_conf,
        'horizon_steps': self.horizon_steps,
        'pred_path': np.array(pred_path),
        'q10_path': np.array(q10_path),
        'q90_path': np.array(q90_path),
    }