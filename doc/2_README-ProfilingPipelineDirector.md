# ProfilingPipelineDirector

As mentioned earlier, using the `PrometheusDataProfiler` can automatically generate statistical reports for specific `*metric_profile.csv` files. Building on this, we can further automate the aggregation and analysis of application metrics using the `ProfilingPipelineDirector` to improve code readability and reusability.

Consider the following scenario in application operations:
When an application is deployed in an environment, it can expose thousands of metrics through the Prometheus endpoint. We need to perform statistical analysis on a subset of key metrics to gain deeper insights into the application’s performance during operations. Using the previous `PrometheusDataProfiler` approach would result in a large amount of repetitive code. Here, we aim to leverage the `register_builder_config` method of the `ProfilingPipelineDirector` to dynamically register these data sources and analysis reports. As the name suggests, the `PipelineDirector` provides a standardized workflow to help accomplish this highly repetitive task.

The specific steps are as follows:

First, inherit the IReportBuilderConfig[ComprehensiveReportBuilder] class and implement its setup_filters method according to business requirements.

Register the data sources into the `ProfilingPipelineDirector` object using the `register_builder_config` method.

Use the `process_multiple_files` method to generate analyses and reports for all data sources with a single command.


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
