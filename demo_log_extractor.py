from model.log_extractor import LogExtractor
from typing import Dict, Any

# Demonstration
def demonstrate_metadata_extraction():
    """Demonstrate the extract_with_metadata method."""
    
    test_text = """
    2024-01-15 10:30:15 ERROR: com.mendix.MendixRuntimeException: Business logic error
    2024-01-15 10:30:20 WARNING: MendixRuntimeException: Also appears without package
    2024-01-15 10:30:25 INFO: java.lang.NullPointerException: Null reference
        at com.example.Processor.handle(Processor.java:123)
    2024-01-15 10:30:30 ERROR: FileNotFoundError: File not found
    2024-01-15 10:30:35 raise CustomBusinessException("validation failed")
    """
    
    extractor = LogExtractor(
        include_package_names=True,
        prefer_simple_names=True,
        keywords=["error", "exception"]
    )
    
    print("Metadata Extraction Demonstration")
    print("=" * 60)
    print("Test Text:")
    print(test_text)
    print("\n" + "=" * 60)
    
    # Get comprehensive metadata
    metadata = extractor.extract_with_metadata(test_text)
    
    print("1. Findings:")
    for finding in metadata['findings']:
        print(f"   - {finding}")
    
    print(f"\n2. Summary:")
    for key, value in metadata['summary'].items():
        print(f"   {key}: {value}")
    
    print(f"\n3. Frequency:")
    for item, count in metadata['frequency'].items():
        print(f"   {item}: {count}")
    
    print(f"\n4. Line Occurrences:")
    for item, lines in metadata['line_occurrences'].items():
        print(f"   {item}: lines {lines}")
    
    print(f"\n5. Categorized:")
    for category, items in metadata['categorized'].items():
        if items:
            print(f"   {category}: {items}")
    
    print(f"\n6. Package Analysis Summary:")
    for key, value in metadata['package_analysis']['summary'].items():
        print(f"   {key}: {value}")

def print_detailed_report(metadata: Dict[str, Any]):
    """Print a formatted report from metadata."""
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS REPORT")
    print("=" * 60)
    
    print(f"\nSUMMARY:")
    print(f"Total unique findings: {metadata['summary']['total_findings']}")
    print(f"Total occurrences: {metadata['summary']['total_occurrences']}")
    print(f"Exceptions: {metadata['summary']['exceptions_count']}")
    print(f"Errors: {metadata['summary']['errors_count']}")
    print(f"Package names: {metadata['summary']['packages_count']}")
    print(f"Keywords: {metadata['summary']['keywords_count']}")
    
    print(f"\nFREQUENCY DISTRIBUTION:")
    for item, count in metadata['frequency'].items():
        print(f"  {item}: {count} occurrences")
    
    print(f"\nOCCURRENCE LOCATIONS:")
    for item, lines in metadata['line_occurrences'].items():
        print(f"  {item}: appears on lines {lines}")
    
    print(f"\nPACKAGE ANALYSIS:")
    package_summary = metadata['package_analysis']['summary']
    print(f"  Simple names: {package_summary['total_simple_names']}")
    print(f"  Package names: {package_summary['total_package_names']}")
    print(f"  Names with packages: {package_summary['names_with_packages']}")
    print(f"  Names with multiple packages: {package_summary['names_with_multiple_packages']}")

if __name__ == "__main__":
    demonstrate_metadata_extraction()
    
    # Additional test with more complex data
    complex_text = """
    com.mendix.MendixRuntimeException: Primary error
    MendixRuntimeException: Duplicate simple name
    org.company.CustomBusinessException: Custom error
    java.io.IOException: I/O error
    at com.mendix.internal.Processor.handle(MendixRuntimeException.java:123)
    Caused by: com.mendix.MendixRuntimeException: Root cause
    raise MendixRuntimeException("error")
    catch (IOException ex)
    """
    
    extractor = LogExtractor(
        include_package_names=True,
        prefer_simple_names=True,
        keywords=["error", "exception"]
    )
    
    metadata = extractor.extract_with_metadata(complex_text)
    print_detailed_report(metadata)