# MetricComparator

When an application experiences anomalies, certain problematic metrics often exhibit significant abnormalities, such as Java heap usage, response times of specific services or databases, or a sudden surge in the invocation count of a service over the past five minutes.

To quickly identify how application metrics behave under abnormal conditions, we often need to compare them with statistics from their normal state. However, the data captured by Prometheus is time-series data and contains a substantial number of zero values. Moreover, some metrics are only collected during anomalous events. To efficiently consolidate this information, we implemented the `MetricComparator` class. This class incorporates multiple statistical testing algorithms, such as `p-value`, `t-statistic`, and `cohen's d`, and also utilizes the `isolation_forest` algorithm with mean and `75th percentile` for anomaly detection analysis. 

Ultimately, it aggregates anomalous metrics during system failures and generates a report.

Code example：
```
normal_df = pd.DataFrame(normal_data)
fault_df = pd.DataFrame(fault_data)

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
```