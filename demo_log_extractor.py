from model.log_extractor import LogExtractor

# Example usage and demonstration
def main():
    """Demonstrate the refactored LogExtractor class with unified keywords."""
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
    2023-10-01 10:08:00 CRITICAL: System failure detected - multiple failures
    2023-10-01 10:09:00 WARNING: Memory leak suspected, possible leak in module
    2023-10-01 10:10:00 FATAL: Application crash imminent, system will crash
    2023-10-01 10:11:00 TIMEOUT: Request timeout after 30 seconds
    2023-10-01 10:12:00 DEADLOCK: Thread deadlock detected in pool, hello world hello world hello world hello world hello world
    """
    
    # Create extractor with unified keywords (includes both exception keywords and custom keywords)
    keywords = ["exception", "error", "failure", "crash", "fatal", "critical", "leak", "timeout", "deadlock", "hello"]
    
    extractor = LogExtractor(
        include_package_names=True,
        keywords=keywords
    )
    
    print("Refactored LogExtractor with Unified Keywords")
    print("=" * 60)
    
    # Show current keywords
    print(f"\n1. Current keywords: {extractor.get_keywords()}")
    print(f"   Has keywords: {extractor.has_keywords()}")
    
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
    print(f"   Keywords: {metadata['summary']['keywords_count']}")
    
    # Frequency analysis
    print(f"\n4. Frequency analysis:")
    frequency = extractor.analyze_frequency(sample_text)
    for item, count in list(frequency.items())[:10]:  # Show top 10
        print(f"   - {item}: {count}")
    
    # Add a new keyword dynamically
    extractor.add_keyword("corruption")
    print(f"\n5. After adding 'corruption': {extractor.get_keywords()}")
    
    # Full report
    print(f"\n6. Full report:")
    extractor.print_report(sample_text, "Unified Keywords Analysis")


if __name__ == "__main__":
    main()
