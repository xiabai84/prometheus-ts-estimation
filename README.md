# prometheus-ts-estimation

This program is used to estimate the metric-data, which is downloaded from Grafana-Dashboard. The estimation is divided with following steps.

## 1 Calculation of statistical indicators
By running prometheus_data_profiling.py the program will calculate the basic statistical indicators, which relies on the given time-series input-csv-file.

## 2 Estimation of Outliers
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
