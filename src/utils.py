from __future__ import annotations 
import json
import os 
from typing import Any , Dict
import joblib 
import numpy as np 
import pandas as pd

def save_artifact(artifact : Any , file_path : str) :
    """Serialize an object using joblib."""
    directory = os.path.dirname(file_path)
    if directory :
        os.makedirs(directory,exist_ok=True)
    joblib.dump(artifact,file_path)

def load_artifact(file_path : str) :
    """Load a joblib artifact."""
    if not os.path.exists(file_path) :
        raise FileNotFoundError(f"Model artifact not found : {file_path}")
    return joblib.load(file_path)
 
def save_json(data : Dict[str,Any],file_path : str) :
    """Save metadata as JSON."""
    directory = os.path.dirname(file_path)
    if directory :
        os.makedirs(directory,exist_ok=True)
    with open(file_path , "w" ,encoding="utf-8") as file :
        json.dump(data,file,indent=4,default=str)

def load_json(file_path : str) :
    """Load JSON metadata."""
    if not os.path.exists(file_path) :
        raise FileNotFoundError(f"Metadata file not found :  {file_path}")
    with open(file_path,"r" , encoding="utf-8") as file :
        return json.load(file)
    
def find_similar_customers(
        customer_id : str ,
        customer_features : pd.DataFrame ,
        X_scaled : np.ndarray ,
        top_n : int = 5 
):
    """Find nearest customers using Euclidean distance."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than zero.")
    if "CustomerID" not in customer_features.columns :
        raise ValueError("customer_features must contain CustomerID.")
    ids = customer_features["CustomerID"].astype(str)
    matches = customer_features.index[ids == str(customer_id)].tolist()
    if not matches :
        raise ValueError(f"Customer ID '{customer_id}' was not found.")
    if len(customer_features) != len(X_scaled) :
        raise ValueError("Feature DataFrame and scaled matrix have different lengths.")
    target_vector = X_scaled[matches[0]]
    distances = np.linalg.norm(X_scaled - target_vector ,axis=1)
    result = pd.DataFrame({
        "CustomerID" : ids.values ,
        "Distance" : distances,
    })
    result = result[result["CustomerID"] != str(customer_id)]
    return result.sort_values("Distance").head(top_n).reset_index(drop=True)

def map_segment_label(profile_row : pd.Series) :
    """Map relative cluster behavior to a business label."""
    profiles = profile_row.attrs.get("all_profiles")
    if profiles is not None and not profiles.empty :
        r_med = profiles["AverageRecency"].median()
        f_med = profiles["AverageFrequency"].median()
        m_med = profiles["AverageMonetaryValue"].median()
    else :
        r_med = profile_row["AverageRecency"]
        f_med = profile_row["AverageFrequency"]
        m_med = profile_row["AverageMonetaryValue"]

    r = profile_row["AverageRecency"]
    f = profile_row["AverageFrequency"]
    m = profile_row["AverageMonetaryValue"]

    if m >= m_med and f >= f_med and r <= r_med :
        return "High-Value Loyal Customers"
    if r > r_med and m >= m_med :
        return "At-Risk Customers"
    if f < f_med and m < m_med :
        return "Low-Engagement Customers"
    if m >= m_med and f < f_med :
        return "High-Spend Occasional Customers"
    return "Regular Customers"

def create_segment_labels(profile:pd.DataFrame) :
    """Assign business labels to cluster profiles."""
    result = profile.copy()
    labels = []
    for _, row in result.iterrows() :
        row.attrs["all_profiles"] = result 
        labels.append(map_segment_label(row))
    result["SegmentLabel"] = labels
    return result 

def recommended_action (segment_label :str)  :
    """Return a recommended strategy for a business segment."""
    actions = {
        "High-Value Loyal Customers" : "Protect retention with loyalty rewards , early access , VIP offers and personalized recommendations." ,
        "At-Risk Customers" : "Launch win-back campaigns , personalized discounts and time-limited offers ",
        "Low-Engagement Customers" : "use low-cost reactivation and product-discovery campaigns.",
        "High-Spend Occasional Customers" : "Increase purchase frequency through cross-selling bundles and personalized follow-ups.",
        "Regular Customers" : "Maintain engagement with relevant recommendations , moderate promotions and loyalty-building campaigns",
    }
    return actions.get(segment_label,"Use targeted experimentation based on observed behavior.")


    