from __future__ import annotations
from typing import Dict , Tuple
import numpy as np 
import pandas as pd
from sklearn.cluster import AgglomerativeClustering , DBSCAN , KMeans
from sklearn.metrics import davies_bouldin_score , silhouette_score
 
def _safe_clustering_metrics(X: np.ndarray ,labels : np.ndarray) :
    """Calculate clustering metrics while handling DBSCAN noise."""
    labels = np.asarray(labels)
    valid_mask = labels != -1 
    if valid_mask.sum() < 3 :
        return{
            "SilhouetteScore" : np.nan ,
            "DaviesBouldinScore" : np.nan ,
            "NumberOfClusters" : 0,
            "NoisePercentage" : 100.0
        }
    valid_labels = labels[valid_mask]
    unique = np.unique(valid_labels)
    if len(unique) <2 :
        return {
            "SilhouetteScore" : np.nan ,
            "DaviesBouldingScore" : np.nan,
            "NumberOfClusters" : len(unique),
            "NoisePercentage" : float((~valid_mask).mean()*100)
        }
    return {
        "SilhouetteScore" :float(silhouette_score(X[valid_mask] , valid_labels)),
        "DaviesBouldinScore" : float(davies_bouldin_score(X[valid_mask],valid_labels)),
        "NumberOfClusters" : int(len(unique)),
        "NoisePercentage" : float((~valid_mask).mean() *100)
    }

def compare_clustering_models(
        X_scaled : np.ndarray,
        kmeans_clusters : int = 4, 
        dbscan_eps : float = 0.8 ,
        dbscan_min_samples : int = 8 ,
        random_state : int = 42
) :
    """Compare K-Means, Agglomerative and DBSCAN."""
    models = {
        "K-Means" : KMeans(n_clusters=kmeans_clusters,random_state=random_state,n_init=20),
        "Agglomerative" : AgglomerativeClustering(n_clusters=kmeans_clusters,linkage="ward"),
        "DBSCAN" : DBSCAN(eps=dbscan_eps,min_samples=dbscan_min_samples),
    }

    labels_dictionary = {}
    records = []
    for name , model in models.items() :
        labels = model.fit_predict(X_scaled)
        labels_dictionary[name] = labels
        metrics = _safe_clustering_metrics(X_scaled,labels)
        metrics["Model"] = name 
        records.append(metrics)
    results = pd.DataFrame(records)
    return results[[
        "Model" , "SilhouetteScore" , "DaviesBouldinScore" ,
        "NumberOfClusters" ,"NoisePercentage"
    ]],labels_dictionary