import argparse

import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Run predictions using trained model.")
    parser.add_argument("--data", required=True, help="Path to new data parquet file.")
    args = parser.parse_args()

    # 1. Load saved model
    clf = joblib.load("models/model.pkl")

    # 2. Load new data
    df_new = pd.read_parquet(args.data)

    # 3. Predict
    predictions = clf.predict(df_new)
    print("Predictions:", predictions)


if __name__ == "__main__":
    main()
