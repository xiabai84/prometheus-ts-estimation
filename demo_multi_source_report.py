from model.pipeline import ProfilingPipelineDirector
from custom_builder_configs.PrometheusMetricReportBuilderConfig import PrometheusMetricReportBuilderConfig

def main():
    """Main execution using the Builder pattern."""
    # Define destination directory for multiple reports
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


if __name__ == "__main__":
    main()