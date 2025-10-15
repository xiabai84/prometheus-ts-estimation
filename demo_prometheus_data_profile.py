import pandas as pd
from model.data_profiler import PrometheusDataProfiler
from model.report_builder import ComprehensiveReportBuilder
from utils.generate_sample_data import generate_sample_prometheus_data

if __name__ == "__main__":

    name_prefix = "report/prometheus_profile"
    csv_file = generate_sample_prometheus_data(file_name="report/sample_prometheus_data.csv")
    # csv_file = "file-name.csv"
    profiler = PrometheusDataProfiler(csv_file)
    
    profiler.load_data()
    report_profiles = profiler.generate_profiling_report()
    profiler.export_report_to_csv(report_profiles, name_prefix)
    df = pd.DataFrame(report_profiles['metric_profiles'])
    
    report_builder = ComprehensiveReportBuilder(df)
    report_columns = ["metric_name", "mean", "median", "std_dev", "skewness", "kurtosis", "min","max","p05","p25","p75","p95","iqr","total_records"]
    
    # Analyse der max. Ausführungszeit
    (report_builder
        .create_filter_group(group_name="max_response_time", description="Max response time greater than 1.5 seconds")
        .add_column_filter('max', '>', 1.5)
        .set_sorting(['max'], ascending=False)
        .select_columns(report_columns))

    # Analyse der instabilen Services
    (report_builder
        .create_filter_group(group_name="instabile_service", description="Instable services using std_dev, skewness, p25")
        .add_column_filter('std_dev', '>', 0.1)
        .add_column_filter('skewness', '>', 0.3)
        .add_column_filter('p25', '>', 0.5)
        .add_sort_column('std_dev', ascending=False)  # Highest rating first
        .add_sort_column('skewness', ascending=False)  # Then highest profit
        .select_columns(report_columns))
    
    # Analyse der p25 percentile
    (report_builder
        .create_filter_group(group_name="p25", description="25th Percentile greater than 0.5 seconds")
        .add_column_filter('p25', '>', 0.5)
        .set_sorting(['p25'], ascending=False)
        .select_columns(report_columns))

    # Analyse der p75 percentile
    (report_builder
        .create_filter_group(group_name="p75", description="75th Percentile greater than 1.0 seconds")
        .add_column_filter('p75', '>', 1.0)
        .set_sorting(['p75'], ascending=False)
        .select_columns(report_columns))

    # Analyse der p95 percentile
    (report_builder
        .create_filter_group(group_name="p95", description="95th Percentile greater than 2.0 seconds")
        .add_column_filter('p95', '>', 1.0)
        .set_sorting(['p95'], ascending=False)
        .select_columns(report_columns))

    filtered_dfs = report_builder.build_all()
    report_builder.save_csv(file_prefix="prometheus_metric", reports=filtered_dfs, dir="report", use_iso_suffix=False)
    
    instable_data = pd.read_csv(f"report/prometheus_metric_instabile_service.csv")
    cols = instable_data.metric_name.to_list()
    profiler.virtualize_skewness_kurtosis(columns=cols, figsize=(18, 12), save_path="instable_service_distribution.png")
