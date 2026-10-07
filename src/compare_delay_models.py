"""
Every real candidate model, scored on flights NONE of them have seen.

A random held-out split flatters any model trained on the same year: the same
day's weather and congestion sit on both sides of it. The honest question is
"trained on the past, how well does it rank flights that come later?", so the
test set is BTS months after every candidate's training year
(build_bts_dataset.py --test).

Default comparison (override via MODELS below or CLI args "label=dir label=dir"):
  2019 model                 original model, origin weather only in training
  live (current /predict)    whatever models/artifacts holds right now
  live, arrival wx withheld  the live model, but with dest_* zeroed out - a
                             robustness check for "what if the forecast fetch
                             fails", not a different model
  <candidate dirs from argv> any new artifact_dir to evaluate against the rest

Run: python compare_delay_models.py [label=dir ...]
"""
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, brier_score_loss

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_preprocessing import FlightWeatherEncoder, load_clean_flight_weather, TARGET_COL  # noqa: E402
from delay_model_v2 import DelayNetV2  # noqa: E402
from config import BTS_TEST_CLEAN_PATH  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEST = ["dest_temp_f", "dest_precip_in", "dest_pressure", "dest_visibility", "dest_wind_speed"]

MODELS = {
    "2019 model": os.path.join(ROOT, "models", "artifacts_pre_bts"),
    "live (current /predict)": os.path.join(ROOT, "models", "artifacts"),
}
for arg in sys.argv[1:]:
    label, _, path = arg.partition("=")
    if path:
        MODELS[label] = path
if len(MODELS) == 2 and len(sys.argv) <= 1:
    # No candidate named on the CLI - fall back to the usual "next" directory.
    MODELS["candidate"] = os.path.join(ROOT, "models", "artifacts_bts_v2")


def load(art_dir):
    enc = FlightWeatherEncoder.load(os.path.join(art_dir, "delay_encoder_v2.joblib"))
    ck = torch.load(os.path.join(art_dir, "delay_model_v2.pt"), map_location="cpu")
    model = DelayNetV2(ck["vocab_sizes"], ck["n_continuous"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    cal_path = os.path.join(art_dir, "calibrator_v2.joblib")
    import joblib
    cal = joblib.load(cal_path) if os.path.exists(cal_path) else None
    return enc, model, cal


def score(enc, model, cal, df):
    xcat, xcont = torch.tensor(enc.transform_cat(df)), torch.tensor(enc.transform_cont(df))
    out = []
    with torch.no_grad():
        for i in range(0, len(df), 50000):
            out.append(torch.sigmoid(model(xcat[i:i + 50000], xcont[i:i + 50000])))
    raw = torch.cat(out).numpy()
    return cal.predict(raw) if cal is not None else raw


def main():
    test = load_clean_flight_weather(BTS_TEST_CLEAN_PATH)
    y = test[TARGET_COL].to_numpy()
    months = sorted(int(m) for m in test["month"].astype(int).unique())
    print(f"Test: {len(test):,} BTS flights, months {months}, delay rate {y.mean():.1%}\n")
    print("Models in this comparison:")
    for label, path in MODELS.items():
        print(f"  {label}: {path}")
    print()

    results = {}
    for label, path in MODELS.items():
        if not os.path.exists(os.path.join(path, "delay_model_v2.pt")):
            print(f"  (skipping '{label}' - no model at {path})")
            continue
        enc, model, cal = load(path)
        results[label] = score(enc, model, cal, test)
        if label == "live (current /predict)":
            # Same model, weather fetch fails at serving time - a robustness
            # check, not a separate model, so it piggybacks on this one's load.
            served = test.copy()
            for c in DEST:
                served[c] = enc.cont_mean[c]
            served["dest_weather_known"] = 0
            results[f"{label}, arrival wx withheld"] = score(enc, model, cal, served)

    advance = test["prev_leg_known"].to_numpy() == 0
    slices = {"all flights": np.ones(len(test), bool),
              "checked in advance (prior leg unknown)": advance}
    report = {}
    for name, m in slices.items():
        print(f"== {name}: n={m.sum():,}, actual delay rate {y[m].mean():.1%}")
        print(f"   {'model':40s} {'AUC':>7s} {'Brier':>7s} {'mean pred':>10s}")
        for k, p in results.items():
            auc, brier = roc_auc_score(y[m], p[m]), brier_score_loss(y[m], p[m])
            report.setdefault(name, {})[k] = {"auc": round(auc, 4), "brier": round(brier, 4),
                                               "mean_pred": round(float(p[m].mean()), 4)}
            print(f"   {k:40s} {auc:7.4f} {brier:7.4f} {p[m].mean():10.1%}")
        print()

    out_dir = next((p for l, p in MODELS.items() if l not in
                    ("2019 model", "live (current /predict)")), MODELS["live (current /predict)"])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "comparison_vs_live.json"), "w") as f:
        json.dump({"test_months": [int(x) for x in months], "n": int(len(test)), **report}, f, indent=2)
    print(f"Saved -> {os.path.join(out_dir, 'comparison_vs_live.json')}")


if __name__ == "__main__":
    main()
