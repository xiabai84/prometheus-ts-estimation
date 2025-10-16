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
    