"""
Trains DelayNetV2 on the cleaned flight+weather 2019 dataset
(outputs/cleaned_flight_weather_2019.csv from clean_monthly_flights.py).

Run: python train_delay_model_v2.py
Outputs:
  models/artifacts/delay_encoder_v2.joblib
  models/artifacts/delay_model_v2.pt

NOTE: this dataset is ~5M+ rows (much larger than Airlines.csv). Expect
training to take a few minutes per epoch on CPU. Set SAMPLE_FRAC below to
a smaller value (e.g. 0.2) for faster iteration while you're debugging,
then set it back to 1.0 for the final training run.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_preprocessing import FlightWeatherEncoder, load_clean_flight_weather, CONT_COLS, TARGET_COL
from delay_model_v2 import DelayNetV2
from config import MONTHLY_FLIGHT_WEATHER_CLEAN_PATH, OUTPUT_DIR

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "artifacts")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reduce this (e.g. 0.15) for a quick smoke-test run before committing to
# the full ~5M-row training pass.
SAMPLE_FRAC = 1.0


def main(epochs: int = 8, batch_size: int = 4096, lr: float = 1e-3):
    print(f"Device: {DEVICE}")
    print("Loading cleaned flight+weather data...")
    df = load_clean_flight_weather(MONTHLY_FLIGHT_WEATHER_CLEAN_PATH)
    print(f"Loaded {len(df):,} rows")

    missing_required = [c for c in CONT_COLS + [TARGET_COL] if c not in df.columns]
    if missing_required:
        raise ValueError(f"Required columns missing from cleaned data: {missing_required}. "
                          f"Did clean_monthly_flights.py run successfully?")

    if SAMPLE_FRAC < 1.0:
        df = df.sample(frac=SAMPLE_FRAC, random_state=42)
        print(f"SAMPLE_FRAC={SAMPLE_FRAC} -> using {len(df):,} rows for this run")

    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df[TARGET_COL])
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,}")
    print(f"Train delay rate: {train_df[TARGET_COL].mean():.3f}")

    encoder = FlightWeatherEncoder().fit(train_df)
    vocab_sizes = encoder.vocab_sizes()
    print("Vocab sizes:", vocab_sizes)

    def to_tensors(d):
        x_cat = torch.tensor(encoder.transform_cat(d))
        x_cont = torch.tensor(encoder.transform_cont(d))
        y = torch.tensor(d[TARGET_COL].to_numpy(dtype=np.float32))
        return x_cat, x_cont, y

    train_cat, train_cont, train_y = to_tensors(train_df)
    val_cat, val_cont, val_y = to_tensors(val_df)

    model = DelayNetV2(vocab_sizes, n_continuous=len(CONT_COLS)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Class imbalance fix: only ~18% of flights are delayed, so an unweighted
    # loss lets the model minimize loss by mostly predicting "on-time" and
    # still getting ~82% accuracy while catching almost no real delays
    # (this is exactly what happened in the first training run - 0.04 recall
    # on the Delayed class). pos_weight scales up the loss contribution from
    # positive (Delayed) examples so the model is actually pushed to learn
    # what separates them, not just to match the base rate.
    pos_rate = train_df[TARGET_COL].mean()
    pos_weight_value = (1 - pos_rate) / pos_rate
    print(f"Class balance: {pos_rate:.1%} delayed -> using pos_weight={pos_weight_value:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=DEVICE))

    n_train = train_cat.shape[0]
    steps_per_epoch = int(np.ceil(n_train / batch_size))

    best_auc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        t0 = time.time()
        for step in range(steps_per_epoch):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            xb_cat = train_cat[idx].to(DEVICE)
            xb_cont = train_cont[idx].to(DEVICE)
            yb = train_y[idx].to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb_cat, xb_cont)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch}", end="\r")

        model.eval()
        with torch.no_grad():
            # Evaluate in chunks to avoid a huge single forward pass on the val set
            val_probs_list = []
            eval_batch = 20000
            for i in range(0, val_cat.shape[0], eval_batch):
                vc = val_cat[i:i+eval_batch].to(DEVICE)
                vco = val_cont[i:i+eval_batch].to(DEVICE)
                logits = model(vc, vco)
                val_probs_list.append(torch.sigmoid(logits).cpu())
            val_probs = torch.cat(val_probs_list).numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            auc = roc_auc_score(val_y.numpy(), val_probs)
            acc = accuracy_score(val_y.numpy(), val_preds)

        print(f"Epoch {epoch}/{epochs} | train_loss={epoch_loss / n_train:.4f} "
              f"| val_auc={auc:.4f} | val_acc={acc:.4f} | {time.time()-t0:.1f}s")

        if auc > best_auc:
            best_auc = auc

    print("\nFinal validation report (threshold=0.5):")
    print(classification_report(val_y.numpy(), val_preds, target_names=["On-time", "Delayed"]))
    print(f"Best val AUC: {best_auc:.4f}")

    # threshold=0.5 is arbitrary - show the real precision/recall tradeoff
    # across thresholds so you can pick one that fits the actual use case
    # (e.g. for a travel-disruption alert, catching more real delays -
    # higher recall - usually matters more than avoiding false alarms).
    print("\nPrecision / Recall / F1 for 'Delayed' class at different thresholds:")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    from sklearn.metrics import precision_score, recall_score, f1_score
    best_f1, best_threshold = 0.0, 0.5
    for t in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
        preds_t = (val_probs >= t).astype(int)
        p = precision_score(val_y.numpy(), preds_t, zero_division=0)
        r = recall_score(val_y.numpy(), preds_t, zero_division=0)
        f1 = f1_score(val_y.numpy(), preds_t, zero_division=0)
        marker = ""
        if f1 > best_f1:
            best_f1, best_threshold = f1, t
            marker = "  <- best F1 so far"
        print(f"{t:>10.2f} {p:>10.3f} {r:>10.3f} {f1:>10.3f}{marker}")
    print(f"\nRecommended threshold (best F1): {best_threshold} -> use this instead of 0.5 "
          f"when deciding 'is this flight high-risk' downstream.")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    encoder.save(os.path.join(ARTIFACT_DIR, "delay_encoder_v2.joblib"))
    torch.save({
        "model_state": model.state_dict(),
        "vocab_sizes": vocab_sizes,
        "n_continuous": len(CONT_COLS),
        "recommended_threshold": best_threshold,
    }, os.path.join(ARTIFACT_DIR, "delay_model_v2.pt"))
    print(f"\nSaved model + encoder to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()