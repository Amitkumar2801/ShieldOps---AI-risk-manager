"""
ShieldOps — Feature Engineering
================================
Turns raw customers.csv + orders.csv into two feature tables:

  1. customer_features  -> input to the RETURN ABUSE model
  2. order_features      -> input to the CHARGEBACK RISK model

Kept in a separate module so both train_models.py (training) and
app/app.py (live scoring) use the EXACT same feature logic — this avoids
train/serve skew, a very common real-world bug.
"""

import pandas as pd
import numpy as np


def build_customer_features(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """One row per customer: return-abuse signals."""
    orders = orders.copy()
    orders["order_value"] = orders["order_value"].astype(float)

    high_value_cutoff = orders["order_value"].quantile(0.75)

    agg = orders.groupby("customer_id").agg(
        total_orders=("order_id", "count"),
        total_returns=("is_returned", "sum"),
        avg_order_value=("order_value", "mean"),
        total_return_value=("order_value", lambda x: x[orders.loc[x.index, "is_returned"] == 1].sum()),
    ).reset_index()

    # avg days to return (only over actually-returned orders)
    returned = orders[orders["is_returned"] == 1].copy()
    returned["days_to_return"] = pd.to_numeric(returned["days_to_return"], errors="coerce")
    avg_days = returned.groupby("customer_id")["days_to_return"].mean().reset_index()
    avg_days.columns = ["customer_id", "avg_days_to_return"]

    # high value return ratio: of all returns, what fraction were high-value items
    returned["is_high_value"] = (returned["order_value"] >= high_value_cutoff).astype(int)
    hv = returned.groupby("customer_id")["is_high_value"].mean().reset_index()
    hv.columns = ["customer_id", "high_value_return_ratio"]

    feats = agg.merge(avg_days, on="customer_id", how="left")
    feats = feats.merge(hv, on="customer_id", how="left")

    feats["return_rate"] = feats["total_returns"] / feats["total_orders"].replace(0, np.nan)
    feats["avg_days_to_return"] = feats["avg_days_to_return"].fillna(0)
    feats["high_value_return_ratio"] = feats["high_value_return_ratio"].fillna(0)
    feats["return_rate"] = feats["return_rate"].fillna(0)

    # Multi-accounting signal: does this customer share an address with
    # another customer who ALSO has returns? (classic wardrobing ring signal)
    addr_map = customers[["customer_id", "address_id"]]
    addr_counts = addr_map["address_id"].value_counts()
    shared_addrs = set(addr_counts[addr_counts > 1].index)
    addr_map = addr_map.copy()
    addr_map["shares_address"] = addr_map["address_id"].isin(shared_addrs).astype(int)

    feats = feats.merge(addr_map[["customer_id", "shares_address"]], on="customer_id", how="left")
    feats["shares_address"] = feats["shares_address"].fillna(0)

    feats = feats.merge(customers[["customer_id", "name", "true_return_abuser"]],
                         on="customer_id", how="left")

    return feats


def build_order_features(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """One row per order: chargeback-risk signals."""
    orders = orders.copy()
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    orders = orders.sort_values(["customer_id", "order_date"])

    # Prior history BEFORE this order (no data leakage from the future)
    orders["customer_prior_orders"] = orders.groupby("customer_id").cumcount()
    orders["customer_prior_chargebacks"] = (
        orders.groupby("customer_id")["is_chargeback"].cumsum() - orders["is_chargeback"]
    )
    orders["customer_prior_returns"] = (
        orders.groupby("customer_id")["is_returned"].cumsum() - orders["is_returned"]
    )

    orders["payment_method_code"] = orders["payment_method"].astype("category").cat.codes
    orders["category_code"] = orders["category"].astype("category").cat.codes

    orders = orders.merge(customers[["customer_id", "address_id"]], on="customer_id", how="left")
    addr_counts = customers["address_id"].value_counts()
    shared_addrs = set(addr_counts[addr_counts > 1].index)
    orders["shares_address"] = orders["address_id"].isin(shared_addrs).astype(int)

    return orders


FEATURE_COLS_RETURN_ABUSE = [
    "total_orders", "total_returns", "return_rate", "avg_order_value",
    "avg_days_to_return", "high_value_return_ratio", "shares_address",
]

FEATURE_COLS_CHARGEBACK = [
    "order_value", "delivery_confirmed", "new_device",
    "customer_prior_orders", "customer_prior_chargebacks",
    "customer_prior_returns", "payment_method_code", "category_code",
    "shares_address",
]
