from __future__ import annotations 
from typing import Dict , List , Optional 
import numpy as np 
import pandas as pd

REQUIRED_COLUMNS = [
    "TransactionID" , "CustomerID" , "TransactionDate",
    "ProductID" , "Category" , "Quantity" ,"UnitPrice"
]

def validate_transaction_data(df: pd.DataFrame , required_columns : Optional[List[str]]= None) :
    """Validate transaction data structure."""
    if not isinstance(df,pd.DataFrame) :
        raise TypeError("Input data must be a pandas DataFrame.")
    if df.empty :
        raise ValueError("Transaction dataset is empty.")
    columns = required_columns or REQUIRED_COLUMNS
    missing = [c for c in columns if c not in df.columns]
    if missing :
        raise ValueError(f"Missing required columns : {', '.join(missing)}")

def load_transactions(file_path : str) :
    """Load transaction data from CSV."""
    try :
        df = pd.read_csv(file_path)
    except FileNotFoundError as exc :
        raise FileNotFoundError(f"Transaction file not found: {file_path}") from exc
    except Exception as exc:
        raise ValueError(f"Unable to read transaction file: {exc}") from exc
    validate_transaction_data(df)
    return df 

def clean_transactions(df : pd.DataFrame) :
    """Clean transaction records and calculate transaction value."""
    validate_transaction_data(df)
    cleaned = df.drop_duplicates().copy()
    cleaned["TransactionDate"] = pd.to_datetime(cleaned["TransactionDate"],errors="coerce")
    for c in ["Quantity" , "UnitPrice"] :
        cleaned[c] = pd.to_numeric(cleaned[c],errors="coerce")
    for c in ["CustomerID" , "ProductID" , "Category"] :
        cleaned[c] = cleaned[c].astype("string").str.strip()
    cleaned = cleaned.dropna(subset=["CustomerID" , "TransactionDate" , "ProductID" ,"Category","Quantity","UnitPrice"])
    cleaned = cleaned[(cleaned["Quantity"]>0)&(cleaned["UnitPrice"]>0)]
    cleaned["TotalAmount"] = cleaned["Quantity"] * cleaned["UnitPrice"]
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna(subset=["TotalAmount"])
    if cleaned.empty :
        raise ValueError("No valid transaction records remain after cleaning.")
    return cleaned.reset_index(drop=True)

def transaction_summary(df: pd.DataFrame) :
    """Return high-level transaction statistics."""
    validate_transaction_data(df)
    amount = df["Quantity"] * df["UnitPrice"]
    return {
        "rows" : int(len(df)),
        "customers" : int(df["CustomerID"].nunique()),
        "products" : int(df["ProductID"].nunique()),
        "categories" : int(df["Category"].nunique()),
        "total_revenue" : float(amount.sum()),
        "average_order_value" : float(amount.mean()),
    }

def calculate_outlier_report(df:pd.DataFrame , columns : Optional[List[str]]=None) :
    """Calculate IQR and Z-score outlier statistics."""
    if df.empty :
        return pd.DataFrame()
    if columns is None :
        columns = [c for c in df.select_dtypes(include= np.number).columns if c != "TransactionID"]
    records = []
    for column in columns :
        if column not in df.columns :
            continue
        values = pd.to_numeric(df[column],errors="coerce").dropna()
        if values.empty :
            continue
        q1,q3 = values.quantile(.25) , values.quantile(.75)
        iqr = q3 - q1 
        lower , upper = q1 - 1.5*iqr , q3 + 1.5* iqr
        iqr_mask = (values < lower) | (values > upper)
        std = values.std()
        z_mask = pd.Series(False , index=values.index) if std == 0 or pd.isna(std) else (
            np.abs((values-values.mean())/std) > 3

        )
        records.append({
            "feature" : column , "q1" : q1 ,"q3" :q3,"iqr" :iqr,
            "lower_bound" : lower , "upper_bound" : upper,
            "iqr_outliers" : int(iqr_mask.sum()),
            "zscore_outliers" : int(z_mask.sum()) ,
            "outlier_percentage" : float(iqr_mask.mean() * 100),
        })
    return pd.DataFrame(records)

def cap_outliers_iqr(df:pd.DataFrame,columns :List[str],multiplier : float = 1.5) :
    """Cap numerical outliers using IQR boundaries."""
    result = df.copy()
    for column in columns :
        if column not in result.columns :
            continue
        values = pd.to_numeric(result[column],errors="coerce")
        q1 , q3 = values.quantile(.25) , values.quantile(.75)
        iqr = q3 - q1 
        if pd.isna(iqr) or iqr == 0 :
            continue
        result[column] = values.clip(q1-multiplier*iqr , q3 + multiplier*iqr)
    return result

def log_transform_features(df:pd.DataFrame,columns:List[str]) :
    """Apply log1p to non-negative skewed features."""
    result = df.copy()
    for column in columns :
        if column in result.columns :
            values = pd.to_numeric(result[column],errors="coerce")
            result[column] = np.log1p(np.clip(values,0,None))
    return result 
    
def prepare_customer_features(customer_df : pd.DataFrame , log_features : Optional[List[str]]=None,
    outlier_features : Optional[List[str]]=None):
    """Impute missing values, cap outliers and log-transform features."""
    if customer_df.empty :
        raise ValueError("Customer feature dataset is empty.")
    result = customer_df.copy()
    features = [
        c for c in result.columns
        if c != "CustomerID" and pd.api.types.is_numeric_dtype(result[c])
    ]
    result[features] = result[features].replace([np.inf,-np.inf],np.nan)
    for c in features :
        result[c] = result[c].fillna(result[c].median() if not pd.isna(result[c].median()) else 0 )
    if outlier_features :
        result = cap_outliers_iqr(result,outlier_features)
    if log_features :
        result = log_transform_features(result,log_features)
    return result 

def get_numeric_feature_columns(df:pd.DataFrame,exclude : Optional[List[str]]=None) :
    """Return numeric columns excluding specified columns."""
    excluded = set(exclude or [])
    return [c for c in df.select_dtypes(include=np.number).columns if c not in excluded]
