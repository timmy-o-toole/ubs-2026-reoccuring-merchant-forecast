"""Write the test submission.

    py -3.10 -m pipeline.make_submission             # -> data/submission_final.csv
    py -3.10 -m pipeline.make_submission --suffix x  # -> data/submission_x.csv

The model is trained on the train clients + pseudo-labelled pretrain clients
(never on valid). Its valid macro-F1 is printed first, then the same model
predicts the test clients.
"""

import argparse

import pandas as pd

from features import build_features
from pipeline.data import CUTOFF, LABEL_COL, LABELS, labelled_features, read_transactions, training_set
from pipeline.evaluate import build_model, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="final", help="output file data/submission_<suffix>.csv")
    args = parser.parse_args()
    out_path = f"data/submission_{args.suffix}.csv"

    X_train, y_train = training_set(labelled_features("train"))
    model = build_model().fit(X_train, y_train)

    valid = labelled_features("valid")
    print(f"model valid macro-F1: {macro_f1(valid[LABEL_COL], model.predict(valid[X_train.columns])):.4f}")

    sample = pd.read_csv("data/sample_submission.csv")
    test = sample[["client_id"]].merge(build_features(read_transactions("test"), CUTOFF),
                                       on="client_id", how="left")
    sub = pd.DataFrame({"client_id": sample["client_id"],
                        "predicted_next_recurring_merchant": model.predict(test[X_train.columns])})

    # Submission contract: exact client IDs, one row each, allowed labels.
    assert list(sub.columns) == list(sample.columns)
    assert len(sub) == len(sample) and sub["client_id"].is_unique
    assert set(sub["client_id"]) == set(sample["client_id"])
    assert sub["predicted_next_recurring_merchant"].isin(LABELS).all()

    sub.to_csv(out_path, index=False)
    print(f"wrote {out_path}: {len(sub)} rows")


if __name__ == "__main__":
    main()
