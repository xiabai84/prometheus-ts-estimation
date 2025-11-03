# MetricProfileClustering

In real-world operational scenarios, after using the `PrometheusDataProfiler` to generate statistical analysis for `*_metric_profile.csv` files (where all data, except for label names, are numerical), we often find that the data dimensionality exceeds thousands of features. This makes it difficult to quickly identify which numerical metrics are exhibiting anomalies.

To enable users to rapidly pinpoint abnormal metrics, we have implemented three different algorithms in the detect_outliers method:
- Isolation Forest
- Z-Score
- DBSCAN

Building on this, we perform unsupervised clustering analysis on the remaining non-outlier data and use the Silhouette-Score to evaluate the clustering results, thereby determining the optimal number of clusters. 

The clustering algorithms implemented include K-Means and DBSCAN.

Since different metrics have varying requirements for feature importance, we further optimized traditional clustering algorithms by providing a method for feature weighting.

Code example:
```
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

CONFIG = {
    'outlier_fraction': 0.07,
    'weights': FEATURE_WEIGHT,
    'n_clusters': 3,
    'clustering_method': 'kmeans', # or dbscan
    'outlier_method': 'isolation_forest',  # 'isolation_forest', 'zscore', 'dbscan'
    'normalization_method': 'standard'  # 'standard', 'robust', 'minmax'
}

df = pd.read_csv(training_data)
clusters = clustering.fit_predict(df, auto_optimize=True)
# Add clusters to original dataframe
df_with_clusters = df.copy()
df_with_clusters['cluster'] = clusters
df_with_clusters.to_csv(output_filename, index=False)
```