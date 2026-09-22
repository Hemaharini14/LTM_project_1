"""
Trains the delay-DURATION regressor: given that a flight is late, how late?

Pairs with the existing classifier rather than replacing it -
    P(delay)             <- delay_model_v2 (calibrated)
    E[minutes | delayed] <- this model
so the app can say "34% chance, and if it happens expect about 50 minutes",
which the binary classifier structurally cannot express.

Design choices forced by the data:

* log1p target. Delay minutes are badly skewed (median 43, p90 150). Plain MSE
  on raw minutes would let a few multi-hour delays dominate every gradient and
  push predictions high for everyone. Training on log1p and inverting with
  expm1 makes the model optimise proportional error, which is also how a
  traveller reads it - being 20 minutes off matters more on a 30-minute delay
  than on a 3-hour one.
* Huber loss, for the outliers that survive the log.
* Same encoder and architecture as the classifier, with the final unit read as
  a value rather than a logit, so both models see identical real features.

THE BASELINE MATTERS. delay_duration.py already answers this from real history
(median delay for this carrier+route). A neural net is only worth shipping if
it beats that, so evaluation compares both on the same held-out rows, and the
script says plainly which won. Delay duration is largely driven by things
absent from these features - crew, the inbound aircraft's own day, ATC flow -
so "the lookup is as good" is a real possible outcome, not a failure to hide.

Run: python train_delay_duration_model.py
Outputs (only written if the model beats the baseline):
    models/artifacts/duration_encoder.joblib
    models/artifacts/duration_model.pt
    models/artifacts/duration_metrics.json
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_preprocessing import FlightWeatherEncoder, CONT_COLS
from delay_model_v2 import DelayNetV2
from config import OUTPUT_DIR

TRAINING_PATH = os.path.join(OUTPUT_DIR, "delay_duration_training.csv")
ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "artifacts")
TARGET = "departure_delay"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _baseline_predictions(train_df: pd.DataFrame, val_df: pd.DataFrame) -> np.ndarray:
    """What delay_duration.py already does: the real median for the most specific
    group with enough history. This is the bar the model has to clear."""
    overall = train_df[TARGET].median()
    by_carrier_route = train_df.groupby(["carrier_code", "route"])[TARGET].agg(["median", "size"])
    by_carrier_route = by_carrier_route[by_carrier_route["size"] >= 30]["median"]
    by_route = train_df.groupby("route")[TARGET].agg(["median", "size"])
    by_route = by_route[by_route["size"] >= 30]["median"]

    cr = pd.MultiIndex.from_arrays([val_df["carrier_code"], val_df["route"]])
    preds = pd.Series(by_carrier_route.reindex(cr).to_numpy(), index=val_df.index)
    preds = preds.fillna(pd.Series(by_route.reindex(val_df["route"]).to_numpy(), index=val_df.index))
    return preds.fillna(overall).to_numpy(dtype=np.float32)


def _report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = np.abs(y_pred - y_true)
    within = lambda m: float((err <= m).mean())
    stats = {
        "mae_min": round(float(err.mean()), 2),
        "median_abs_err_min": round(float(np.median(err)), 2),
        "within_15_min": round(within(15), 4),
        "within_30_min": round(within(30), 4),
    }
    print(f"  {name:22s} MAE {stats['mae_min']:6.2f} min | median abs err "
          f"{stats['median_abs_err_min']:6.2f} min | within 15min {stats['within_15_min']:.1%} "
          f"| within 30min {stats['within_30_min']:.1%}")
    return stats


def main(epochs: int = 6, batch_size: int = 4096, lr: float = 1e-3):
    print(f"Device: {DEVICE}")
    df = pd.read_csv(TRAINING_PATH, low_memory=False)
    for col in ["carrier_code", "origin_airport", "destination_airport",
                "weekday", "month", "scheduled_hour"]:
        df[col] = df[col].astype(str)
    print(f"Loaded {len(df):,} real delayed flights")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
    print(f"Train {len(train_df):,} | Val {len(val_df):,}")

    y_val = val_df[TARGET].to_numpy(dtype=np.float32)

    print("\nBaseline (real historical median, what delay_duration.py already does):")
    baseline_pred = _baseline_predictions(train_df, val_df)
    baseline_stats = _report("historical median", y_val, baseline_pred)

    encoder = FlightWeatherEncoder().fit(train_df)
    model = DelayNetV2(encoder.vocab_sizes(), n_continuous=len(CONT_COLS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=1.0)

    def tensors(d):
        return (torch.tensor(encoder.transform_cat(d)),
                torch.tensor(encoder.transform_cont(d)),
                torch.tensor(np.log1p(d[TARGET].to_numpy(dtype=np.float32))))

    tr_cat, tr_cont, tr_y = tensors(train_df)
    va_cat, va_cont, _ = tensors(val_df)

    n = tr_cat.shape[0]
    steps = int(np.ceil(n / batch_size))
    print(f"\nTraining {epochs} epochs x {steps} steps on log1p(minutes)...")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total, t0 = 0.0, time.time()
        for step in range(steps):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            optimizer.zero_grad()
            out = model(tr_cat[idx].to(DEVICE), tr_cont[idx].to(DEVICE))
            loss = criterion(out, tr_y[idx].to(DEVICE))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, va_cat.shape[0], 20000):
                preds.append(model(va_cat[i:i + 20000].to(DEVICE),
                                    va_cont[i:i + 20000].to(DEVICE)).cpu())
        val_pred = np.expm1(torch.cat(preds).numpy())
        mae = float(np.abs(val_pred - y_val).mean())
        print(f"  epoch {epoch}/{epochs} train_loss={total / n:.4f} val_MAE={mae:.2f} min "
              f"({time.time() - t0:.0f}s)")

    print("\nFinal comparison on the same held-out flights:")
    _report("historical median", y_val, baseline_pred)
    model_stats = _report("neural regressor", y_val, val_pred)

    improvement = baseline_stats["mae_min"] - model_stats["mae_min"]
    print(f"\nModel beats baseline by {improvement:.2f} min MAE "
          f"({improvement / baseline_stats['mae_min']:+.1%})")

    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "train_size": len(train_df), "validation_size": len(val_df),
        "baseline": baseline_stats, "model": model_stats,
        "mae_improvement_min": round(improvement, 2),
        "model_wins": bool(improvement > 0.5),
    }

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, "duration_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if metrics["model_wins"]:
        encoder.save(os.path.join(ARTIFACT_DIR, "duration_encoder.joblib"))
        torch.save({"model_state": model.state_dict(), "vocab_sizes": encoder.vocab_sizes(),
                    "n_continuous": len(CONT_COLS)},
                   os.path.join(ARTIFACT_DIR, "duration_model.pt"))
        print(f"Saved model + encoder to {ARTIFACT_DIR}")
    else:
        print("NOT saving the model: it doesn't meaningfully beat the historical median, "
              "so shipping it would add complexity and a false impression of precision "
              "for no accuracy gain. delay_duration.py's lookup stays the answer.")


if __name__ == "__main__":
    main()
