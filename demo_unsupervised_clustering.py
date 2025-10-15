import pandas as pd
import numpy as np
from model.unsupervised_clustering import AdvancedUnsupervisedClustering

def main(training_data: str, output_filename: str):
    print("Advanced Unsupervised Clustering with Outlier Detection")
    print("="*60)
    # Create or load your dataset
    print("\nLoading data...")
    df = pd.read_csv(training_data)  #
    print(f"Dataset shape: {df.shape}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    
    FEATURE_WEIGHT = {
        "metric_name": 0.0,
        "min": 0.0,
        "max": 2.0,
        "mean": 0.0,
        "median": 0.0,
        "std_dev": 3.0,
        "skewness": 0.0,
        "kurtosis": 0.0,
        "p05": 1.0,
        "p25": 7.0,
        "p75": 7.0,
        "p95": 4.0,
        "iqr": 0.0,
        "null_count": 1.0,
        "nul_percentage": 0.0,
        "non_zero_records": 0.0,
        "zero_count": 2.0,
        "zero_percentage": 0.0,
        "total_records": 0.0
    }

    # Configuration
    CONFIG = {
        'outlier_fraction': 0.07,
        'weights': FEATURE_WEIGHT,
        'n_clusters': 3,
        'clustering_method': 'kmeans', # or dbscan
        'outlier_method': 'isolation_forest',  # 'isolation_forest', 'zscore', 'dbscan'
        'normalization_method': 'standard'  # 'standard', 'robust', 'minmax'
    }
    
    print(f"\n⚙️  Configuration:")
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    
    # Initialize clustering model
    clustering = AdvancedUnsupervisedClustering(**CONFIG)
    
    # Perform clustering
    print(f"\n🔍 Performing clustering with outlier detection...")
    clusters = clustering.fit_predict(df, auto_optimize=True)
    
    # Add clusters to original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    # Save results
    df_with_clusters.to_csv(output_filename, index=False)
    print(f"\n💾 Results saved to '{output_filename}'")
    
    # Display sample of results
    print(f"\n📋 Sample of clustered data (first 10 rows):")
    print(df.loc[(df_with_clusters["p25"] > 0.5)].head(10))
    
    # Final summary
    print(f"\n✅ Clustering completed successfully!")
    print(f"   Total data points: {len(df)}")
    print(f"   Clusters found: {len(np.unique(clusters))} (including outliers as cluster -1)")
    
    cluster_counts = df_with_clusters['cluster'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        if cluster == -1:
            print(f"   📛 Outliers: {count} points ({percentage:.1f}%)")
        else:
            print(f"   🔷 Cluster {cluster}: {count} points ({percentage:.1f}%)")

if __name__ == "__main__":
    dir_name = "report"
    training_data = f"{dir_name}/prometheus_profile_metric_profiles.csv"
    output_file = f"{dir_name}/clustered_data_with_outliers.csv"
    main(training_data, output_file)
    