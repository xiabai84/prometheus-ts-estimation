from model.log_extractor import LogExtractor

def main():
    """Demonstrate the extended LogExtractor class with custom keywords."""
    # Sample text for testing
    sample_text = """
    2023-10-01 10:00:00 ERROR: FileNotFoundError: File not found
    2023-10-01 10:01:00 INFO: Process started successfully
    2023-10-01 10:02:00 WARNING: An exception occurred during processing
    2023-10-01 10:03:00 java.lang.NullPointerException: Cannot invoke method
        at com.example.MyClass.processData(MyClass.java:123)
    2023-10-01 10:04:00 Caused by: org.springframework.dao.DataAccessException
    2023-10-01 10:05:00 raise ValueError("Invalid value")
    2023-10-01 10:06:00 except RuntimeError as e:
    2023-10-01 10:07:00 Generic error without specific type
    2023-10-01 10:08:00 CRITICAL: System failure detected
    2023-10-01 10:09:00 WARNING: Memory leak suspected
    2023-10-01 10:10:00 FATAL: Application crash imminent
    2023-10-01 10:10:00 INFO: bai adds here new keywords "hello world"
    """
    
    # Create extractor with custom keywords
    custom_keywords = ["failure", "crash", "fatal", "critical", "leak", "timeout"]
    
    extractor = LogExtractor(
        include_package_names=True,
        include_keywords=True,
        custom_keywords=custom_keywords
    )
    
    print("Extended LogExtractor with Custom Keywords Demo")
    print("=" * 60)
    
    # Show current keywords
    print(f"\n1. Current custom keywords: {extractor.get_keywords()}")
    
    # Basic extraction
    findings = extractor.extract_from_text(sample_text)
    print(f"\n2. Basic extraction found {len(findings)} items:")
    for finding in findings:
        print(f"   - {finding}")
    
    # Extraction with metadata
    print(f"\n3. Extraction with metadata:")
    metadata = extractor.extract_with_metadata(sample_text)
    print(f"   Total findings: {metadata['summary']['total_findings']}")
    print(f"   Exceptions: {metadata['summary']['exceptions_count']}")
    print(f"   Errors: {metadata['summary']['errors_count']}")
    print(f"   Custom Keywords: {metadata['summary']['custom_keywords_count']}")
    
    # Add a new keyword dynamically
    extractor.add_keyword("deadlock")
    print(f"\n4. After adding 'deadlock': {extractor.get_keywords()}")
    
    # Remove a keyword
    extractor.remove_keyword("timeout")
    print(f"\n5. After removing 'timeout': {extractor.get_keywords()}")
    
    # Update all keywords will overwrite the settings before
    new_keywords = ["failure", "deadlock", "corruption", "integrity"]
    extractor.update_keywords(new_keywords)
    print(f"\n6. After updating keywords: {extractor.get_keywords()}")
    
    # add to new keyword after update
    extractor.add_keyword("hello")
    extractor.add_keyword("world")

    # Full report
    print(f"\n7. Full report with custom keywords:")
    extractor.print_report(sample_text, "Sample Log Analysis with Custom Keywords")


if __name__ == "__main__":
    main()