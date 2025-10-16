import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from model.data_profiler import PrometheusDataProfiler
from model.report_builder import ComprehensiveReportBuilder
from model.builder_config import IReportBuilderConfig


class ProfilingPipelineDirector:
    """Director class that orchestrates the profiling pipeline using builders."""
    
    def __init__(self, base_output_dir: str = "report"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.results = []
        self.current_datetime = datetime.now()
        # Builder registry
        self._builder_configs = {}
    
    def register_builder_config(self, builder_config: IReportBuilderConfig[ComprehensiveReportBuilder]) -> None:
        """Register a new builder."""
        self._builder_configs[builder_config.builder_name] = builder_config
    
    def get_builder_config(self, name: str) -> IReportBuilderConfig[ComprehensiveReportBuilder]:
        """Get builder by name."""
        if name not in self._builder_configs:
            raise ValueError(f"Unknown builder: {name}. Available: {list(self._builder_configs.keys())}")
        return self._builder_configs[name]
    
    def process_file(self, csv_file: str, file_name: str, builder_name: str, 
                   generate_visualization: bool = True, output_dir="", use_iso_suffix=True) -> Dict[str, Any]:
        """Process a single CSV file using the specified builder."""

        print(f"Processing {file_name} with {builder_name} configuration...")
        output_dir = Path(output_dir)

        if use_iso_suffix:
            time_iso_format = self.current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            # output_dir = os.path.join(output_dir, time_iso_format)
            output_dir = output_dir.joinpath(output_dir, time_iso_format)
        
        output_dir = output_dir.joinpath(builder_name)
        directory = Path(output_dir)
        directory.mkdir(parents=True)

        try:
            # Initialize profiler
            profiler = PrometheusDataProfiler(csv_file)
            profiler.load_data()
            
            # Generate profiling report
            name_prefix = f"profile_{file_name}"
            report_profiles = profiler.generate_profiling_report()
            profiler.export_report_to_csv(report_profiles, output_dir, name_prefix)
            
            # Create DataFrame and setup report builder
            df = pd.DataFrame(report_profiles['metric_profiles'])
            report_builder = ComprehensiveReportBuilder(df)
            
            # Apply builder configuration
            builder_config = self.get_builder_config(builder_name)
            builder_config.setup_filters(report_builder)
            
            # Build and save reports
            filtered_dfs = report_builder.build_all()
            export_path = report_builder.save_csv(
                file_prefix=f"{file_name}",
                reports=filtered_dfs,
                dir=output_dir
            )
            
            visualization_paths = []
            visualization_groups = report_builder.visualizations

            for vis_group in visualization_groups:
                if generate_visualization and vis_group in filtered_dfs:
                    visualization_path = self._generate_visualization(
                        profiler=profiler, 
                        group_name=vis_group, 
                        export_path=export_path, 
                        file_name=file_name
                    )
                    visualization_paths.append(visualization_path)
            
            result = {
                "file_name": file_name,
                "builder_name": builder_name,
                "export_path": export_path,
                "reports_generated": list(filtered_dfs.keys()),
                "visualization_path": visualization_paths,
                "success": True
            }
            
            self.results.append(result)
            return result
            
        except Exception as e:
            import shutil
            shutil.rmtree(directory)
            error_result = {
                "file_name": file_name,
                "builder_name": builder_name,
                "success": False,
                "error": str(e)
            }
            self.results.append(error_result)
            raise
    
    def _generate_visualization(self, profiler: PrometheusDataProfiler, group_name: str, export_path: str, 
                              file_name: str) -> Optional[str]:
        """Generate visualization for unstable services."""
        try:
            instable_data = pd.read_csv(f"{export_path}/{file_name}_{group_name}.csv")
            cols = instable_data.metric_name.to_list()
            viz_path = f"{export_path}/{file_name}_{group_name}_distribution.png"
            profiler.virtualize_skewness_kurtosis(
                columns=cols, 
                figsize=(18, 12), 
                save_path=viz_path, 
                bw_method=0.5
            )
            return viz_path
        except Exception as e:
            print(f"Warning: Could not generate visualization: {e}")
            return None
    
    def process_multiple_files(self, report_dir: str) -> List[Dict[str, Any]]:
        """Process multiple files with their configurations."""
        
        for _, config in self._builder_configs.items():
            self.process_file(
                csv_file=config.csv_file,
                file_name=config.file_name,
                builder_name=config.builder_name,
                generate_visualization=config.generate_visualization,
                output_dir=report_dir
            )
        return self.results
    
    def print_summary(self) -> None:
        """Print processing summary."""
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        print(f"\n{'='*50}")
        print("PROCESSING SUMMARY")
        print(f"{'='*50}")
        print(f"Successful: {len(successful)} files")
        print(f"Failed: {len(failed)} files")
        
        for result in successful:
            print(f"\n✓ {result['file_name']} ({result['builder_name']})")
            print(f"  Reports: {result['reports_generated']}")
            if result['visualization_path']:
                print(f"  Visualization: {result['visualization_path']}")
        
        if failed:
            print(f"\nFailed files:")
            for fail in failed:
                print(f"  ✗ {fail['file_name']}: {fail['error']}")

