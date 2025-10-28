import pandas as pd
import numpy as np
from model.comparison import MetricComparator

def main():
    """
    Use case:
    in SRE daily tasks, if a system goes wrong, we would like to compare the regulary metrics with metrics back to two hours of alerting time.
    """
    np.random.seed(42)
    # Create comprehensive test data
    normal_data = []
    fault_data = []
    
    # Common metrics with good data
    for service in ['api-gateway', 'user-service', 'order-service', 'payment-service']:
        for i in range(100):
            normal_data.append({
                'timestamp': pd.Timestamp.now() - pd.Timedelta(hours=i),
                'service_name': service,
                'response_time': np.random.normal(100, 10),
                'throughput': np.random.normal(1000, 100)
            })
        
        for i in range(80):
            multiplier = 2.5 if service == 'user-service' else 1.2 if service == 'payment-service' else 1.0
            fault_data.append({
                'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=i*10),
                'service_name': service,
                'response_time': np.random.normal(100 * multiplier, 15),
                'throughput': np.random.normal(1000 * multiplier, 150)
            })
    
    # Metrics only in fault data
    for service in ['new-service', 'cache-service']:
        for i in range(60):
            fault_data.append({
                'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=i*5),
                'service_name': service,
                'response_time': np.random.normal(200, 30),
                'throughput': np.random.normal(500, 80)
            })
    
    normal_df = pd.DataFrame(normal_data)
    fault_df = pd.DataFrame(fault_data)
    
    print("Initial data overview:")
    print(f"Normal data: {normal_df.shape}")
    print(normal_df)
    print(f"Fault data: {fault_df.shape}")
    
    # Initialize and analyze
    comparator = MetricComparator()
    
    comparator.prepare_data(
        normal_df=normal_df,
        fault_df=fault_df,
        value_col='response_time',
        metric_col='service_name',
        min_data_points=10,
        handle_nulls='drop',
        handle_zeros='drop'
    )
    
    print("\n" + "="*50)
    print("METRIC INFORMATION")
    print("="*50)
    
    info = comparator.get_metric_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    # Run all analyses
    print("\n" + "="*50)
    print("RUNNING COMPLETE ANALYSIS")
    print("="*50)
    
    # 1. Basic statistical comparison
    stats_results = comparator.basic_statistical_comparison()
    # print(f"Basic comparison: {len(stats_results)} metrics")
    if not stats_results.empty:
        print(f"\nStatistical results for {len(stats_results)} metrics:")
        print(stats_results[['metric', 'mean_change_percent', 'p75_change_percent', 'p_value', 'effect_size']].head().to_string(index=False))
    

    # 2. Generate summary report
    summary = comparator.generate_summary_report()
    print(f"\nSummary report:")
    print(summary.to_string(index=False))

    # 3. Get most changed metrics
    print("\n" + "="*50)
    print("TOP CHANGED METRICS")
    print("="*50)
    top_metrics = comparator.get_most_changed_metrics(5)
    if not top_metrics.empty:
        print(top_metrics[['metric', 'mean_change_percent', 'p75_change_percent', 'p_value', 'effect_size']].to_string(index=False))

    # 4. Get percentile analysis
    print("\n" + "="*50)
    print("75TH PERCENTILE ANALYSIS")
    print("="*50)
    
    percentile_analysis = comparator.get_percentile_analysis(top_n=2)
    if not percentile_analysis.empty:
        print("Top metrics by 75th percentile change:")
        print(percentile_analysis.to_string(index=False))

    # 5. Anomaly detection
    anomaly_results = comparator.anomaly_detection_comparison(contamination=0.3)
    
    if not anomaly_results.empty:
        print(f"Anomaly detection: {len(anomaly_results)} metrics analyzed")
        
        # Show anomalous metrics
        anomalous_metrics = comparator.get_anomalous_metrics()
        if not anomalous_metrics.empty:
            print(f"\nAnomalous metrics detected:")
            print(anomalous_metrics[['metric', 'anomaly_score', 'change_percent']].to_string(index=False))
        else:
            print("No anomalous metrics detected")

        percentile_anomalies = anomaly_results[
            (anomaly_results['is_anomalous']) & 
            (anomaly_results['percentile_driven_anomaly'])
        ]
        print("\nPercentile driven anomaly:")
        print(percentile_anomalies)

    # 6. Fault-only metrics details
    print("\n" + "="*50)
    print("FAULT-ONLY METRICS DETAILS")
    print("="*50)
    
    fault_only_details = comparator.get_fault_only_metrics_details()
    if not fault_only_details.empty:
        print("Details of metrics only found in fault data:")
        print(fault_only_details.to_string(index=False))

    # 6. Create visualization
    print("\nGenerating visualization...")
    comparator.visualize_comparison(top_n=4, save_path="fixed_analysis.png")
    
    print(f"\n✅ All tests completed successfully!")

if __name__ == "__main__":
    main()
