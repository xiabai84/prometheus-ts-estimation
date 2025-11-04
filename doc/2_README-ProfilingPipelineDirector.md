# ProfilingPipelineDirector

As mentioned earlier, using the `PrometheusDataProfiler` can automatically generate statistical reports for specific `*metric_profile.csv` files. Building on this, we can further automate the aggregation and analysis of application metrics using the `ProfilingPipelineDirector` to improve code readability and reusability.

Consider the following scenario in application operations:
When an application is deployed in an environment, it can expose thousands of metrics through the Prometheus endpoint. We need to perform statistical analysis on a subset of key metrics to gain deeper insights into the application’s performance during operations. Using the previous `PrometheusDataProfiler` approach would result in a large amount of repetitive code. Here, we aim to leverage the `register_builder_config` method of the `ProfilingPipelineDirector` to dynamically register these data sources and analysis reports. As the name suggests, the `PipelineDirector` provides a standardized workflow to help accomplish this highly repetitive task.

The specific steps are as follows:

First, inherit the `IReportBuilderConfig[ComprehensiveReportBuilder]` class and implement its setup_filters method according to business requirements.

```
from model.builder_config import IReportBuilderConfig
from model.report_builder import ComprehensiveReportBuilder

class PrometheusMetricReportBuilderConfig(IReportBuilderConfig[ComprehensiveReportBuilder]):
    """Builder for strict analysis configuration."""
    
    def setup_filters(self, builder: ComprehensiveReportBuilder) -> ComprehensiveReportBuilder:
        base_columns = ["metric_name", "mean", "median", "std_dev", "skewness", "kurtosis", 
                       "min", "max", "p05", "p25", "p75", "p95", "iqr", "total_records"]
        
        # Analyse der max. Ausführungszeit
        (builder
            .create_filter_group(group_name="max_response_time", description="Max response time greater than 1.5 seconds")
            .add_column_filter('max', '>', 1.5)
            .set_sorting(['max'], ascending=False)
            .select_columns(base_columns))

        # Analyse der instabilen Services
        (builder
            .create_filter_group(group_name="instabile_service", description="Instable services using std_dev, skewness, p25")
            .add_column_filter('std_dev', '>', 0.1)
            .add_column_filter('skewness', '>', 0.3)
            .add_column_filter('p25', '>', 0.5)
            .add_sort_column('std_dev', ascending=False)  # Highest rating first
            .add_sort_column('skewness', ascending=False)  # Then highest profit
            .add_visualization()
            .select_columns(base_columns))
        
        # Analyse der p25 percentile
        (builder
            .create_filter_group(group_name="p25", description="25th Percentile greater than 0.5 seconds")
            .add_column_filter('p25', '>', 0.5)
            .set_sorting(['p25'], ascending=False)
            .select_columns(base_columns))

        # Analyse der p75 percentile
        (builder
            .create_filter_group(group_name="p75", description="75th Percentile greater than 1.0 seconds")
            .add_column_filter('p75', '>', 1.0)
            .set_sorting(['p75'], ascending=False)
            .select_columns(base_columns))

        # Analyse der p95 percentile
        (builder
            .create_filter_group(group_name="p95", description="95th Percentile greater than 2.0 seconds")
            .add_column_filter('p95', '>', 1.0)
            .set_sorting(['p95'], ascending=False)
            .add_visualization()
            .select_columns(base_columns))
        
        return builder
    
```

Register the data sources into the `ProfilingPipelineDirector` object using the `register_builder_config` method. 
In this example we will reuse the class `PrometheusMetricReportBuilderConfig` to apply the same configuration for multiple input-data.

At the end we can use the `process_multiple_files` method to generate analyses and reports for all data sources with a single command.


Code example：
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
    generate_visualization=True
)
# register custom builders
director.register_builder_config(builder_config=prometheus_metric_config)
director.register_builder_config(builder_config=prometheus_count_config)

director.process_multiple_files(report_dir=report_dir)

# Print summary
director.print_summary()
```
