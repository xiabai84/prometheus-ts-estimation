from model.ts_clustering import DTWClustering

# Demonstration function
def main():
    """Demonstrate the complete functionality including new methods"""
    print("=== Complete DTWClustering Demo ===")
    
    # Generate sample data
    def generate_sample_data():
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        data = {}
        patterns = [
            lambda i: np.sin(np.linspace(0 + i*0.5, 4*np.pi + i*0.5, 100)) + np.random.normal(0, 0.1, 100),
            lambda i: np.linspace(0, 10 + i, 100) + np.random.normal(0, 0.2, 100),
            lambda i: np.sin(np.linspace(0, 8*np.pi, 100)) * (1 + i*0.1) + np.random.normal(0, 0.15, 100),
        ]
        
        for i in range(15):
            pattern_type = i % len(patterns)
            data[f'metric_pattern{pattern_type}_{i}'] = patterns[pattern_type](i)
        
        return pd.DataFrame(data, index=dates)
    
    data = generate_sample_data()
    print(f"Generated data shape: {data.shape}")
    
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

if __name__ == "__main__":
    model = main()
    
    # Window Size Guidelines:

    # window=None: Full DTW, most accurate but O(n²) complexity

    # Absolute values: window_constraint=10 (10 time steps)
    # Percentage: window_constraint=0.1 (10% of series length)
    # Preset types: 'sakoe_chiba'
    # warp_penalty=0.5,           # For soft-DTW
    
    # example
    # model.visualize_similar_columns("target_metric", top_k=5)