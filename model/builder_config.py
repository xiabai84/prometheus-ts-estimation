from typing import TypeVar, Generic
from abc import ABC, abstractmethod
from model.report_builder import ComprehensiveReportBuilder

# Define generic type variable constrained to ComprehensiveReportBuilder
T = TypeVar('T', bound=ComprehensiveReportBuilder)

class IReportBuilderConfig(ABC, Generic[T]):
    """Generic interface for report builders that only work with ComprehensiveReportBuilder or its subclasses."""

    def __init__(self, csv_file: str, file_name: str, builder_name, generate_visualization: bool = True):
        """
        Initialize the report builder with file and configuration parameters.
        
        Args:
            csv_file: Path to the CSV file to process
            file_name: Name identifier for the file
            builder_name: Name of the builder configuration
            generate_visualization: Whether to generate visualizations
        """
        self.csv_file = csv_file
        self.file_name = file_name
        self.builder_name = builder_name
        self.generate_visualization = generate_visualization

    @abstractmethod
    def setup_filters(self, report_builder: T) -> T:
        """
        Setup filters and configurations on a ComprehensiveReportBuilder instance.
        
        Args:
            report_builder: An instance of ComprehensiveReportBuilder (or subclass) to configure
        """
        pass
    