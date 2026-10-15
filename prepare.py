import pandas as pd

RAW_PATH = "transactions_raw.csv"


def clean_transactions(raw_df):
    df = raw_df.drop_duplicates().copy()                                # (a)
    for c in ["merchant_category", "city"]:                             # (b)
        df[c] = df[c].str.strip().str.title()
    df = df[df["amount_inr"].notna()]                                   # (c)
    df = df[(df["amount_inr"] >= 1) & (df["amount_inr"] <= 1_000_000)]  # (d)
    return df.reset_index(drop=True)                                    # (e)


def add_features(df):
    out = df.copy()
    out["amount_vs_avg_ratio"] = out["amount_inr"] / out["avg_amount_30d_inr"]
    out["is_night"] = (out["hour_of_day"] <= 5).astype(int)
    out["high_value"] = (out["amount_inr"] > 50_000).astype(int)
    return out


if __name__ == "__main__":
    raw = pd.read_csv(RAW_PATH)
    print("rows, columns as read:", raw.shape)
    print("exact duplicate rows:", raw.duplicated().sum())
    print("distinct raw spellings: category", raw["merchant_category"].nunique(),
          "| city", raw["city"].nunique())

    dd = raw.drop_duplicates()
    print("after dedup rows:", len(dd))
    print("missing amount_inr:", dd["amount_inr"].isna().sum())
    print("missing distance_from_home_km:", dd["distance_from_home_km"].isna().sum())
    print("amount <1 or >1,000,000:",
          ((dd["amount_inr"] < 1) | (dd["amount_inr"] > 1_000_000)).sum())

    before = raw.copy()
    clean = clean_transactions(raw)
    assert raw.equals(before), "input was modified!"
    print("\nclean rows:", len(clean))
    print(clean["merchant_category"].value_counts().to_dict())
    print(clean["city"].nunique(), "cities")
    print(clean["payment_mode"].value_counts().to_dict())
    print("amount min/max:", clean["amount_inr"].min(), clean["amount_inr"].max())
    print("missing distance:", clean["distance_from_home_km"].isna().sum())

    feat = add_features(clean)
    print("\nis_night sum:", feat["is_night"].sum())
    print("high_value sum:", feat["high_value"].sum())
    print("max ratio:", round(feat["amount_vs_avg_ratio"].max(), 2))

    n, f = len(feat), int(feat["is_fraud"].sum())
    print(f"\nfraud rows: {f} of {n} ({f/n:.2%})")
    print(f"never-fraud rule: accuracy {(n-f)/n:.2%}, recall 0")
    print("\nleakage table (chargeback_filed):")
    print(feat.groupby("chargeback_filed")["is_fraud"].agg(rows="count", fraud="sum"))

    # one-row check (Part D will do this)
    one = clean_transactions(raw).iloc[[0]]
    print("\none-row check:", add_features(clean_transactions(one)).shape)