# prometheus-ts-estimation

This program is used to estimate the metric-data, which is downloaded from Grafana-Dashboard. The estimation is divided with following steps.

## 1 Calculation of statistical indicators
By running prometheus_data_profiling.py the program will calculate the basic statistical indicators, which relies on the given time-series input-csv-file.

## 2 Data Profiling
With this library we can easily calculate the basic statistics of form Prometheus exported time series data.

Code example
```
name_prefix = "report/prometheus_profile"
csv_file = generate_sample_prometheus_data(file_name="report/sample_prometheus_data.csv")
# csv_file = "file-name.csv"
profiler = PrometheusDataProfiler(csv_file)
profiler.load_data()
report_profiles = profiler.generate_profiling_report()
profiler.export_report_to_csv(report_profiles, name_prefix)
```

## 3 Setting constrains for generating reports

There are two ways to set groups for reporting.

### 3.1 Builder pattern
TBD

```
...
business_builder = ComprehensiveReportBuilder(df)

# Define various business constraints
(business_builder
 .create_filter_group("q1_targets")
 .add_column_filter('sales_amount', '>=', 2000)
 .add_column_filter('profit', '>=', 300)
 .add_column_filter('rating', '>=', 3)
 
 .create_filter_group("high_risk_orders") 
 .add_column_filter('profit', '<', 0)
 .add_column_filter('rating', '<=', 2)
 ...
```

### 3.2 Passing config as dictionary

```
def create_dynamic_filters():
    """Create filters based on dynamic conditions"""
    dynamic_builder = ComprehensiveReportBuilder(df)
    
    # These could come from user input, configuration, etc.
    dynamic_constraints = [
        {'group': 'user_defined_1', 'column': 'sales_amount', 'op': '>', 'value': 2500},
        {'group': 'user_defined_1', 'column': 'region', 'op': 'in', 'value': ['North', 'South']},
        {'group': 'user_defined_2', 'column': 'product_category', 'op': '==', 'value': 'Clothing'},
        {'group': 'user_defined_2', 'column': 'profit', 'op': '>', 'value': 100},
    ]
    
    for constraint in dynamic_constraints:
        dynamic_builder.create_filter_group(constraint['group'])
        dynamic_builder.add_column_filter(
            constraint['column'], 
            constraint['op'], 
            constraint['value']
        )
    
    return dynamic_builder.build_all()

dynamic_results = create_dynamic_filters()
print("Dynamic Filter Results:")
for group, result in dynamic_results.items():
    print(f"  {group}: {len(result)} rows")
```

## 4 Estimation of Outliers
By using unsupervised_clustering.py it will use "KMeans" clustering algorithm and additional "isolation_forest" to detect outliers and the potential clusters. Due to different data distribution it is to recommend to use "stardard normalization" algorithm to normalize data before training the mode. This can be done by setting the parameter "CONFIG".

In the current implementation we also considered to assign weights (feature importance), to make a better prediction.
For example:
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
```

## 5 Generating reports from multiple input files

In order to simplify genarating reports from multiple sources for example from multiple results of prometheus queries saving with csv-format, have can do it by using `ProfilingPipelineDirector`.
This class can register a set of builder-configurations by extending a `IReportBuilderConfig[ComprehensiveReportBuilder]`. Currently we only implemented `ComprehensiveReportBuilder`, which already contains rich statistical indicators e.g. max, min, median, skewness, kurtosis, n-th-percentile, zero_counts etc..
They are very useful for analysing time-series-data. 

Ref: `3.1 Builder pattern`

Under custom_builder_configs we provide a example `PrometheusMetricReportBuilderCofnig` for putting cosntraints on the default `ComprehensiveReportBuilder`.

To undertand how to use it, see the example below:

```
report_dir = "report"
director = ProfilingPipelineDirector()

prometheus_metric_config = PrometheusMetricReportBuilderConfig(
    csv_file=f"{report_dir}/1_sample_prometheus_data.csv",
    file_name="mx_activity_max",
    builder_name="mx_activity_max",
    generate_visualization=True
)
prometheus_count_config = PrometheusMetricReportBuilderConfig(
    csv_file=f"{report_dir}/2_sample_prometheus_data.csv",
    file_name="mx_activity_count",
    builder_name="mx_activity_count",
    generate_visualization=False
)
# register custom builders
director.register_builder_config(builder_config=prometheus_metric_config)
director.register_builder_config(builder_config=prometheus_count_config)

director.process_multiple_files(report_dir=report_dir)

# Print summary
director.print_summary()
```

In this exmpale we have two csv files as input, both of them will use the same conditions to creating reports for its own datasets.

In real-world szenario it make sense to implement multiple BuilderConfiguration classes for different bussiness scopes.

## 6 Fazit


