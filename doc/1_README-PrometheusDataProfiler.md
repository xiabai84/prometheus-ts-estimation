# PrometheusDataProfiler

The PrometheusDataProfiler is primarily used to process Prometheus metric CSV data downloaded from Grafana Dashboards and generate statistical analysis reports.

The exported data format is as follows:
```
timestamp,http_requests_total,cpu_usage,memory_usage,disk_io,network_bytes
2024-01-01 00:00:00,5.00,6.00,5.00,2,6.00
2024-01-01 01:00:00,8.00,15.00,9.00,6,12.00
```

The default analysis includes the following statistics:
- metric_name
- mean
- median
- std_dev
- skewness
- kurtosis
- min
- max
- p05
- p25
- p75
- p95
- iqr
- null_count
- null_percentage
- non_zero_records
- zero_count
- zero_percentage
- total_records

The code implementation is as follows:
```
output_dir = "report"
name_prefix = "prometheus_metric"
input_csv_file = generate_sample_prometheus_data(file_name="report/sample_prometheus_data.csv")
profiler = PrometheusDataProfiler(input_csv_file)
profiler.load_data()
report_profiles = profiler.generate_profiling_report()
profiler.export_report_to_csv(report=report_profiles, output_dir=output_dir, output_prefix=name_prefix)
```

After calling the `generate_profiling_report` and `export_report_to_csv` methods, a set of CSV format reports will be saved in the output_dir. The most important statistical report is the `*_metric_profiles.csv` file, which contains monitoring data for a specific metric including all labels. This data can be directly used for preliminary screening and analysis with tools like Excel. After understanding the general data distribution, to automate the generation of specific screening results next time, we can use the tool `ComprehensiveReportBuilder` from this library.

Its usage is as follows (prerequisite is that the `*_metric_profiles.csv` file has been generated):

```
def setup_report_builder(metric_profile: pd.DataFrame):
    builder = ComprehensiveReportBuilder(metric_profile)
    report_columns = ["metric_name", "mean", "median", "std_dev", "skewness", "kurtosis", "min","max","p05","p25","p75","p95","iqr","total_records"]
    
    # Analyse der max. Ausführungszeit
    (builder
        .create_filter_group(group_name="max_response_time", description="Max response time greater than 1.5 seconds")
        .add_column_filter('max', '>', 1.5)
        .set_sorting(['max'], ascending=False)
        .select_columns(report_columns))

    # Analyse der instabilen Services
    (builder
        .create_filter_group(group_name="instabile_service", description="Instable services using std_dev, skewness, p25")
        .add_column_filter('std_dev', '>', 0.1)
        .add_column_filter('skewness', '>', 0.3)
        .add_column_filter('p25', '>', 0.5)
        .add_sort_column('std_dev', ascending=False)  # Highest rating first
        .add_sort_column('skewness', ascending=False)  # Then highest profit
        .select_columns(report_columns))
    
    # Analyse der p25 percentile
    (builder
        .create_filter_group(group_name="p25", description="25th Percentile greater than 0.5 seconds")
        .add_column_filter('p25', '>', 0.5)
        .set_sorting(['p25'], ascending=False)
        .select_columns(report_columns))
    
    return builder
```
As shown in the code above, we can add multiple report groups with different statistical dimensions. Each `filter_group` will generate a corresponding report file prefixed with that `filter_group` name. The `ComprehensiveReportBuilder` uses the Builder Design Pattern, which allows us to flexibly configure the data display through this code design.

After configuring the `ComprehensiveReportBuilder`, we can generate the desired reports:
```
# take results (dataframe report_profiles) from previous steps
df = pd.DataFrame(report_profiles['metric_profiles'])
report_builder = setup_report_builder(df)
filtered_dfs = report_builder.build_all()
export_path = report_builder.save_csv(file_prefix="prometheus_metric", reports=filtered_dfs, dir="report", use_iso_suffix=True)
instable_data = pd.read_csv(f"{export_path}/prometheus_metric_instabile_service.csv")
cols = instable_data.metric_name.to_list()
profiler.virtualize_skewness_kurtosis(columns=cols, figsize=(18, 12), save_path=f"{export_path}/instable_service_distribution.png", bw_method=0.5)
```

If you're interested in the data distribution of a particular `filter_group` report, you can call the `virtualize_skewness_kurtosis` method to visualize it.
