# DTWClustering

In application operations, when an application fails, it often impacts the performance of multiple services in production. These services typically exhibit strong statistical correlations. Understanding their correlations can aid in Root Cause Analysis (RCA) of the problem. However, root cause analysis is particularly challenging due to operations teams' limited understanding of the code and business logic, while developers often lack familiarity with operational aspects and metrics.

To address these challenges, our library provides the `DTWClustering` class. This class utilizes the `Dynamic Time Warping (DTW)` algorithm at its core to calculate distances between pairs of time series data. Building on this, it employs an optimized time-series K-means algorithm to perform clustering analysis on the results. This helps identify strongly correlated metrics preceding failures, providing a starting point for in-depth analysis and investigation.

In practical scenarios, we typically extract approximately 30 minutes of metrics data preceding a failure for analysis, as this period often contains the root causes of the anomaly and holds the greatest statistical significance in terms of metric values.

Code example：
```
# Create and train model with window constraints
cluster_model = DTWClustering(
    n_clusters=3,
    metric="dtw",
    window_constraint=0.1,  # 10% window constraint
    max_iter=10,
    verbose=True
)
cluster_model.fit(data)

# Test get_metric_similarity_groups
print("\n🎯 Testing get_metric_similarity_groups():")
similarity_groups = cluster_model.get_metric_similarity_groups()
for i, group in enumerate(similarity_groups):
    print(f"  Group {i} ({len(group)} metrics): {group[:3]}...")

# Test get_topk_similar_columns
target_column = "metric_pattern0_0"
print(f"\n🎯 Testing get_topk_similar_columns for '{target_column}':")
similar_cols = cluster_model.get_topk_similar_columns(target_column, top_k=3, method='hybrid')
for item in similar_cols:
    cluster_flag = "🟢" if item['same_cluster'] else "🔵"
    print(f"  {cluster_flag} {item['column_name']} - Similarity: {item['similarity_score']:.3f}")

# visualization 
cluster_model.visualize_similar_columns(target_column, top_k=5)

# Test save_results_to_json
print(f"\n💾 Testing save_results_to_json():")
try:
    saved_files = cluster_model.save_results_to_json(
        filename_prefix='demo_clustering',
        output_dir='./demo_results'
    )
    print("✅ Results saved successfully!")
except Exception as e:
    print(f"❌ Error saving results: {e}")

return cluster_model
```
