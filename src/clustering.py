from __future__ import annotations
from typing import List , Tuple
import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def scale_features (X: pd.DataFrame | np.ndarray) :
    """Standardize features using z-score standardization."""
    scaler = StandardScaler()
    return scaler , scaler.fit_transform(X)

def euclidean_distance(point_a : np.ndarray,point_b : np.ndarray) :
    """Calculate Euclidean distance between two vectors."""
    a , b = np.asarray(point_a,dtype=float) , np.asarray(point_b,dtype=float)
    if a.shape != b.shape :
        raise ValueError("Both vectors must have identical dimensions.")
    return float(np.linalg.norm(a-b))

def calculate_diff_k_scores(X_scaled : np.ndarray , k_values : List[int],random_state : int = 42,) :
    """Calculate K-Means inertia and silhouette scores."""
    if len(X_scaled) < 3 :
        raise ValueError("At least three observations are required.")
    records = []
    for k in k_values :
        if k < 2 or k >= len(X_scaled) :
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=20)
        labels = model.fit_predict(X_scaled)
        records.append({
            "K" : k,
            "Inertia" : model.inertia_,
            "SilhouetteScore" : silhouette_score(X_scaled,labels) if len(np.unique(labels))>1 else np.nan
        })
    return pd.DataFrame(records)

def train_kmeans(X_scaled:np.ndarray,n_clusters:int ,random_state : int = 42) :
    """Train the final K-Means model."""
    if n_clusters < 2 or n_clusters >= len(X_scaled) :
        raise ValueError("Invalid number of clusters.")
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=50,
        max_iter=500
    )
    return model , model.fit_predict(X_scaled)

def fit_pca(X_scaled : np.ndarray , n_components : int = 3) :
    """Fit PCA for dimensionality reduction."""
    max_components = min(X_scaled.shape[0],X_scaled.shape[1])
    n_components = min(n_components,max_components)
    if n_components < 2 :
        raise ValueError("PCA requires at least two components.")
    pca = PCA(n_components=n_components,random_state=42)
    return pca , pca.fit_transform(X_scaled)

def add_pca_coordinates(customer_features : pd.DataFrame ,X_pca : np.ndarray) :
    """Append PCA coordinates to customer records."""
    if len(customer_features) != len(X_pca) :
        raise ValueError("Customer rows and PCA rows must have equal length.")
    result = customer_features.copy()
    result["PC1"] , result["PC2"] = X_pca[:,0] , X_pca[:,1]
    if X_pca.shape[1] >=3 :
        result["PC3"] = X_pca[:,2]
    return result

def cluster_profile(raw_customer_df: pd.DataFrame, cluster_labels: np.ndarray, cluster_column: str = "Cluster") :
    """Create an aggregated profile for every cluster."""
    df = raw_customer_df.copy()
    df[cluster_column] = cluster_labels
    
    profile = df.groupby(cluster_column).agg(
        CustomerCount = ("CustomerID" , "count"),
        AverageRecency = ("Recency" , "mean"),
        AverageFrequency = ("Frequency" , "mean"),
        AverageMonetaryValue = ("MonetaryValue" ,"mean"),
        AverageAOV = ("AOV" , "mean"),
        AverageQuantity = ("TotalQuantityPurchased" ,"mean"),
        AverageUniqueProducts = ("UniqueProducts" , "mean"),
        AverageUniqueCategories = ("UniqueCategories" , "mean"),
        AveragePurchaseGap = ("AverageDaysBetweenPurchases" , "mean")
    
    ).reset_index()
    profile["Percentage"] = profile["CustomerCount"]/profile["CustomerCount"].sum()*100
    return profile.sort_values("AverageMonetaryValue",ascending=False).reset_index(drop=True)
