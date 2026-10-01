from __future__ import annotations 
from typing import List 
import numpy as np 
import pandas as pd 

def calculate_average_purchase_gap(customer_transactions: pd.DataFrame) :
    """Calculate the mean number of days between purchases."""
    dates = pd.to_datetime(customer_transactions["TransactionDate"],errors="coerce").dropna().sort_values()
    if len(dates) < 2 :
        return 0.0
    gaps = dates.diff().dt.total_seconds()/86400
    return float(gaps.dropna().mean())

def build_customer_features(
        transactions : pd.DataFrame ,
        reference_date : pd.Timestamp | None = None
) :
    """Aggregate transactions into RFM and behavioral customer features."""
    required = {
        "CustomerID" , "TransactionDate" , "ProductID" , "Category",
        "Quantity" , "TotalAmount"
    }
    missing = required - set(transactions.columns)
    if missing :
        raise ValueError(f"Missing columns for feature engineering : {missing}")
    if transactions.empty :
        raise ValueError("Cannot engineer features from empty data.")
    df = transactions.copy()
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"],errors="coerce")
    df = df.dropna(subset=["CustomerID" , "TransactionDate" , "TotalAmount"])
    if reference_date is None :
        reference_date = df["TransactionDate"].max() + pd.Timedelta(days=1)

    grouped = df.groupby("CustomerID")
    rfm = grouped.agg(
        LastPurchaseDate = ("TransactionDate","max") ,
        Frequency = ("TransactionID" , "nunique"),
        MonetaryValue = ("TotalAmount" , "sum")
    )
    rfm["Recency"] = (reference_date - rfm["LastPurchaseDate"]).dt.days

    behavioral = grouped.agg(
        TotalQuantityPurchased = ("Quantity" , "sum"),
        UniqueProducts = ("ProductID" , "nunique"),
        UniqueCategories = ("Category" , "nunique"),
    )
    behavioral["AOV"] = rfm["MonetaryValue"] /rfm["Frequency"].replace(0,np.nan)

    purchase_gaps = (
        df.groupby("CustomerID").apply(calculate_average_purchase_gap,include_groups=False).rename("AverageDaysBetweenPurchases")
    )

    features = pd.concat([rfm.drop(columns=["LastPurchaseDate"]),behavioral,purchase_gaps],axis = 1 )

    numeric = ["Recency" , "Frequency", "MonetaryValue" ,"AOV","TotalQuantityPurchased" , "UniqueProducts",
               "UniqueCategories" , "AverageDaysBetweenPurchases"]

    for c in numeric :
        features[c] = pd.to_numeric(features[c],errors="coerce")
    features = features.replace([np.inf,-np.inf],np.nan)
    features[numeric] = features[numeric].fillna(0)
    return features.reset_index() 

def rank_customer_features(customer_features:pd.DataFrame) :
    """Add percentile-based behavioral scores."""
    result = customer_features.copy()
    for c in ["MonetaryValue" , "Frequency" , "Recency" , "AOV"] :
        if c in result :
            rank = result[c].rank(pct=True,method="average")
            result[f"{c}_Score"] = 1 - rank if c=="Recency" else rank
    return result

def get_model_features(customer_features : pd.DataFrame) :
    """Return behavioral features used by the clustering model."""
    columns = [
        "Recency" , "Frequency" , "MonetaryValue" ,"AOV",
        "TotalQuantityPurchased" , "UniqueProducts" ,
        "UniqueCategories" , "AverageDaysBetweenPurchases"
    ]
    missing = [c for  c in columns if c not in customer_features.columns]
    if missing :
        raise ValueError(f"Missing model features : {missing}")
    return columns