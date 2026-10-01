from __future__ import annotations 
from pathlib import Path 
import sys
import numpy as np 
import pandas as pd 

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT_ROOT))

from src.clustering import add_pca_coordinates , calculate_diff_k_scores , cluster_profile,fit_pca,scale_features,train_kmeans
from src.data_processing import clean_transactions ,load_transactions , prepare_customer_features
from src.feature_engineering import build_customer_features,get_model_features
from src.model_evaluation import compare_clustering_models
from src.utils import create_segment_labels,save_artifact,save_json

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RAW_DATA_PATH = DATA_DIR / "raw_transactions.csv"
CUSTOMER_FEATURES_PATH = MODELS_DIR / "customer_features.csv"
CLUSTERED_DATA_PATH = MODELS_DIR / "clustered_customers.csv"
DIFF_K_PATH = MODELS_DIR / "diff_k_scores.csv"
MODEL_COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
PROFILE_PATH = MODELS_DIR / "segment_profiles.csv"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
KMEANS_PATH = MODELS_DIR / "kmeans_model.joblib"
PCA_PATH = MODELS_DIR / "pca_model.joblib"
SIMILARITY_MATRIX_PATH = MODELS_DIR / "similarity_matrix.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

def generate_dummy_transactions(output_path : Path , n_customers : int = 500 ,n_transactions : int = 1000 ,random_state : int=42) :
    """Generate reproducible synthetic e-commerce transactions."""
    rng = np.random.default_rng(random_state)
    customer_ids = np.array([f"CUST_{i:04d}" for i in range(1,n_customers+1)])
    product_ids = np.array([f"PROD_{i:03d}" for i in range(1,101)])
    categories = np.array(["Electronics","Home" , "Fashion" ,"Beauty","Sports"])
    groups = rng.choice(["loyal" , "occasional","low" ,"at_risk"], size = n_customers,p=[.2,.3,.3,.2])
    group_map = dict(zip(customer_ids,groups))
    selected = rng.choice(customer_ids,size=n_transactions,replace=True)
    rows = []
    end_date = pd.Timestamp("2026-08-01")

    for transaction_id , customer_id in enumerate(selected,start=1) :
        group = group_map[customer_id]
        if group == "loyal" :
            days_ago , quantity , price = int(rng.exponential(60)),int(rng.poisson(3) + 1) , float(rng.lognormal(4.0,.45))
        elif group == "occasional" :
            days_ago ,quantity , price = int(rng.exponential(140)),int(rng.poisson(2)+1) , float(rng.lognormal(3.7,.50))
        elif group == "low" :
            days_ago ,quantity,price = int(rng.exponential(220)), int(rng.poisson(1) +1) , float(rng.lognormal(3.3,.55))
        else :
            days_ago , quantity ,price = int(rng.uniform(100,300)) , int(rng.poisson(2)+1) , float(rng.lognormal(3.9,.50))
        rows.append({
            "TransactionID" : transaction_id,
            "CustomerID" : customer_id,
            "TransactionDate" :end_date - pd.Timedelta(days = days_ago),
            "ProductID" : rng.choice(product_ids),
            "Category" : rng.choice(categories),
            "Quantity" : quantity ,
            "UnitPrice" : round(price,2),
        })
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents = True , exist_ok = True)
    df.to_csv(output_path,index=False)
    return df

def choose_best_k(elbow_results : pd.DataFrame) :
    """Select K with the highest valid silhouette score."""
    valid = elbow_results.dropna(subset=["SilhouetteScore"])
    if valid.empty :
        return 4
    return int(valid.loc[valid["SilhouetteScore"].idxmax(),"K"])

def main() :
    """Execute the complete ML pipeline."""
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    MODELS_DIR.mkdir(parents=True,exist_ok=True)

    print("[1/10] Loading/generating data...")

    if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 100 :
        transactions = load_transactions(str(RAW_DATA_PATH))
    else :
        print("Raw transaction file is empty .Generating synthetic transaction data...")
        transactions = generate_dummy_transactions(RAW_DATA_PATH)

    print("[2/10] Cleaning...")
    transactions = clean_transactions(transactions)

    print("[3/10] Feature engineering...") 
    customer_features = build_customer_features(transactions)
    model_features = get_model_features(customer_features)

    print("[4/10] Preprocessing...")
    raw_customer_features = customer_features.copy()
    log_features = [
        "MonetaryValue" , "AOV" , "TotalQuantityPurchased" ,
        "UniqueProducts" , "UniqueCategories" , "Frequency" ,
        "AverageDaysBetweenPurchases"
    ]
    processed = prepare_customer_features(
        customer_features,
        log_features=log_features,
        outlier_features=model_features
    )
    processed.to_csv(CUSTOMER_FEATURES_PATH,index=False)

    print("[5/10] Scaling...")
    scaler , X_scaled = scale_features(processed[model_features])
    save_artifact(scaler,str(SCALER_PATH))

    print("[6/10] Evaluating K...")
    k_values = list(range(2,min(10,len(processed)-1)+1))
    elbow = calculate_diff_k_scores(X_scaled,k_values)
    elbow.to_csv(DIFF_K_PATH,index=False)
    best_k = choose_best_k(elbow)
    print(f"Selected K: {best_k}")

    print("[7/10] Training K-Means...")
    kmeans, labels = train_kmeans(X_scaled,best_k)
    processed["Cluster"] = labels
    save_artifact(kmeans,str(KMEANS_PATH))

    print("[8/10] Comparing models...")
    comparison,_ = compare_clustering_models(X_scaled,best_k)
    comparison.to_csv(MODEL_COMPARISON_PATH,index=False)

    print("[9/10] PCA...")
    pca , X_pca = fit_pca(X_scaled,3)
    processed = add_pca_coordinates(processed,X_pca)
    save_artifact(pca , str(PCA_PATH))

    print("[10/10] Saving artifacts...")

    raw_customer_features["Cluster"] = labels
    raw_customer_features = add_pca_coordinates(raw_customer_features, X_pca)
    profile = create_segment_labels(cluster_profile(raw_customer_features, labels))
    profile.to_csv(PROFILE_PATH,index=False)
    raw_customer_features.to_csv(CLUSTERED_DATA_PATH , index=False)
    save_artifact(X_scaled,str(SIMILARITY_MATRIX_PATH))
    save_json({
        "feature_columns" : model_features ,
        "log_features" : log_features,
        "n_clusters" : best_k,
        "n_customers" : len(processed),
        "pca_explained_variance_ratio" : pca.explained_variance_ratio_.tolist(),
        "pca_total_explained_variance" : float(pca.explained_variance_ratio_.sum()),
    }, str(METADATA_PATH))

    print("\nPipeline completed successfully.")
    print(f"Customers: {len(processed):,}")
    print(f"Clusters : {best_k}")
    print(f"PCA variance explained : {pca.explained_variance_ratio_.sum():.2%}")

if __name__ == "__main__" :
    main()