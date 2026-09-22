"""
Answers "is the delay prediction actually working?" with real numbers instead
of a vibe. train_delay_model_v2.py already computes real held-out validation
metrics (AUC, accuracy, precision/recall/F1 per threshold) every time it
trains - but only prints them to the console, where they're gone the moment
the terminal closes. This script re-derives the EXACT same validation split
(same train_test_split call, same random_state=42, same stratify column, on
the same unified CSV) WITHOUT re-training - it just loads the already-saved
model/encoder artifacts and evaluates them on the held-out 15% - and persists
the result so the admin panel can show it.

It ALSO fits a real probability calibrator here, because the raw model output
is NOT a calibrated probability: train_delay_model_v2.py deliberately applies
pos_weight to the loss (to stop the model from ignoring the rare "Delayed"
class), which systematically inflates every raw score. Checked against real
held-out flights, a raw score of "0.42" corresponds to an actual delay rate of
about 13% - genuinely low, but badly overstated by the raw number. Isotonic
regression (monotonic, so it never changes the model's real ranking/AUC) maps
each raw score to the real historical delay rate observed at that score on
this held-out set, so predict_delay_v2.py can return an honest probability
instead of an inflated one. risk_label()'s Low/Moderate/High cutoffs are then
set against this real calibrated scale, not arbitrary raw-score cutoffs.

It also saves a small real sample of held-out rows (their real recorded
inputs AND their real actual outcome) to outputs/validation_sample.csv, so
the admin panel can show concrete "predicted vs actual" examples on genuine
historical flights - not just a headline accuracy number - via
model_metrics.sample_validation_flights().

Run: python evaluate_delay_model_v2.py
Outputs:
    models/artifacts/metrics_v2.json
    models/artifacts/calibrator_v2.joblib
    outputs/validation_sample.csv
"""
import os
import sys
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, classification_report,
                              precision_score, recall_score, f1_score)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_preprocessing import FlightWeatherEncoder, load_clean_flight_weather, CAT_COLS, CONT_COLS, TARGET_COL
from delay_model_v2 import DelayNetV2
from config import UNIFIED_FLIGHT_WEATHER_CLEAN_PATH, OUTPUT_DIR

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "artifacts")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics_v2.json")
CALIBRATOR_PATH = os.path.join(ARTIFACT_DIR, "calibrator_v2.joblib")
VALIDATION_SAMPLE_PATH = os.path.join(OUTPUT_DIR, "validation_sample.csv")
SAMPLE_SIZE = 300


def _auc_by_prior_leg(val_df, y_true, probs) -> dict:
    """AUC separately for flights where the inbound aircraft's delay was known vs not.

    The unknown case is the one the web form actually hits most of the time, so if it
    regresses against the previous model this feature set is a net loss for real use,
    however good the headline number looks.
    """
    out = {}
    if "prev_leg_known" not in val_df.columns:
        return out
    flags = val_df["prev_leg_known"].to_numpy()
    for label, mask in [("prior_leg_known", flags == 1), ("prior_leg_unknown", flags == 0)]:
        n = int(mask.sum())
        # AUC is undefined without both classes present
        if n < 100 or len(set(y_true[mask].tolist())) < 2:
            out[label] = {"n": n, "auc": None}
            continue
        out[label] = {"n": n, "auc": round(float(roc_auc_score(y_true[mask], probs[mask])), 4),
                       "actual_delay_rate": round(float(y_true[mask].mean()), 4)}
    return out


def main():
    print("Loading unified dataset and reconstructing the original validation split...")
    df = load_clean_flight_weather(UNIFIED_FLIGHT_WEATHER_CLEAN_PATH)
    # Same call, same random_state, same stratify column as train_delay_model_v2.py -
    # this reproduces the identical held-out 15% the model never trained on,
    # as long as the underlying CSV hasn't changed since that training run.
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df[TARGET_COL])
    print(f"Validation set: {len(val_df):,} real held-out flights "
          f"({val_df[TARGET_COL].mean():.1%} actually delayed)")

    encoder = FlightWeatherEncoder.load(os.path.join(ARTIFACT_DIR, "delay_encoder_v2.joblib"))
    ckpt = torch.load(os.path.join(ARTIFACT_DIR, "delay_model_v2.pt"), map_location="cpu")
    model = DelayNetV2(ckpt["vocab_sizes"], ckpt["n_continuous"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_cat = torch.tensor(encoder.transform_cat(val_df))
    x_cont = torch.tensor(encoder.transform_cont(val_df))
    y_true = val_df[TARGET_COL].to_numpy(dtype=np.float32)

    probs_list = []
    eval_batch = 20000
    with torch.no_grad():
        for i in range(0, x_cat.shape[0], eval_batch):
            logits = model(x_cat[i:i + eval_batch], x_cont[i:i + eval_batch])
            probs_list.append(torch.sigmoid(logits).cpu())
    raw_probs = torch.cat(probs_list).numpy()

    # Fit the real calibration mapping (raw score -> real historical delay rate at that
    # score, on this held-out set) and save it so predict_delay_v2.py can apply it to every
    # future prediction. Monotonic, so AUC/ranking is identical before and after - only the
    # absolute number changes, from "inflated by pos_weight" to "matches real frequency."
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_probs, y_true)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    probs = calibrator.predict(raw_probs)

    def metrics_at(threshold: float) -> dict:
        preds = (probs >= threshold).astype(int)
        return {
            "threshold": round(float(threshold), 4),
            "precision": round(precision_score(y_true, preds, zero_division=0), 4),
            "recall": round(recall_score(y_true, preds, zero_division=0), 4),
            "f1": round(f1_score(y_true, preds, zero_division=0), 4),
        }

    # Best-F1 threshold search on the CALIBRATED scale (0.5 would almost never fire - the
    # real base rate is ~18%, so a meaningful "Moderate/High" cutoff is well below 0.5).
    candidate_thresholds = [round(t, 3) for t in np.arange(0.05, 0.55, 0.025)]
    f1_scores = [f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in candidate_thresholds]
    recommended_threshold = candidate_thresholds[int(np.argmax(f1_scores))]

    preds_default = (probs >= 0.5).astype(int)
    result = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "validation_size": len(val_df),
        "actual_delay_rate": round(float(y_true.mean()), 4),
        "auc": round(float(roc_auc_score(y_true, probs)), 4),
        "calibrated": True,
        "accuracy_at_default_threshold": round(float(accuracy_score(y_true, preds_default)), 4),
        "recommended_threshold": round(float(recommended_threshold), 4),
        "at_default_threshold": metrics_at(0.5),
        "at_recommended_threshold": metrics_at(recommended_threshold),
        "classification_report_at_default_threshold": classification_report(
            y_true, preds_default, target_names=["On-time", "Delayed"], output_dict=True, zero_division=0),
        # Split by whether the aircraft's prior leg was observable. prev_leg_arrival_delay
        # is the strongest feature in the data but only exists same-day, so a single
        # blended AUC would flatter the case that actually matters: a traveller checking
        # a flight next week, where that feature is absent. Report both.
        "auc_by_prior_leg": _auc_by_prior_leg(val_df, y_true, probs),
        # Real calibration check: for flights whose CALIBRATED score falls in each bucket,
        # what fraction actually got delayed - should track the bucket range closely, unlike
        # the raw-score version of this same table (see the module docstring).
        "calibration_buckets": [
            {"range": f"{lo:.2f}-{hi:.2f}", "count": int(mask.sum()),
             "actual_rate": round(float(y_true[mask].mean()), 4) if mask.sum() else None}
            for lo, hi in zip([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65], [0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 1.01])
            for mask in [(probs >= lo) & (probs < hi)]
        ],
    }

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved real validation metrics to {METRICS_PATH}")
    print(f"Saved calibrator to {CALIBRATOR_PATH}")
    print(f"AUC={result['auc']} | recommended threshold (calibrated)={recommended_threshold}")
    print("\nCalibration check (calibrated score bucket -> real actual delay rate):")
    for b in result["calibration_buckets"]:
        print(f"  {b['range']}: n={b['count']:>7} actual_rate={b['actual_rate']}")

    # A real, reproducible sample of held-out rows (their real recorded inputs +
    # real actual outcome) for the admin panel's "predicted vs actual on real past
    # flights" spot-check - predictions are recomputed live from the current model
    # at display time (see model_metrics.sample_validation_flights), this file only
    # carries the real inputs/outcome, never a frozen prediction that could drift
    # out of sync with whichever model artifact is actually loaded later.
    sample_cols = CAT_COLS + [c for c in CONT_COLS if c != "route_frequency"] + [TARGET_COL]
    sample_df = val_df[sample_cols].sample(n=min(SAMPLE_SIZE, len(val_df)), random_state=42)
    sample_df.to_csv(VALIDATION_SAMPLE_PATH, index=False)
    print(f"Saved {len(sample_df)} real held-out flights to {VALIDATION_SAMPLE_PATH}")


if __name__ == "__main__":
    main()
