from model.log_extractor import LogExtractor
def main():
    """Demonstrate the optimized ExceptionExtractor."""
    
    # Test data with various exception types
    test_text = """
    2024-01-15 10:30:15 ERROR: FileNotFoundError: File not found
    2024-01-15 10:30:16 INFO: Process started
    2024-01-15 10:30:20 WARNING: MendixRuntimeException: Business logic error
    2024-01-15 10:30:25 ERROR: java.lang.NullPointerException: Null reference
        at com.example.Processor.handle(Processor.java:123)
    2024-01-15 10:30:30 Caused by: org.springframework.dao.DataAccessException
    2024-01-15 10:30:35 raise CustomBusinessException("validation failed")
    2024-01-15 10:30:40 catch (IOException ex)
    2024-01-15 10:30:45 Multiple occurrences: MendixRuntimeException MendixRuntimeException
    2024-01-15 10:30:50 AnotherMendixRuntimeException: Another issue
    """
    
    # Create optimized extractor
    extractor = LogExtractor(
        include_package_names=True,
        keywords=["error", "exception", "failure"]
    )
    
    print("Optimized ExceptionExtractor Demonstration")
    print("=" * 60)
    print("Test Text:")
    print(test_text)
    print("\n" + "=" * 60)
    
    # Basic extraction
    print("1. Basic Extraction:")
    findings = extractor.extract_from_text(test_text)
    for i, finding in enumerate(findings, 1):
        print(f"   {i:2d}. {finding}")
    
    # Frequency analysis
    print("\n2. Frequency Analysis:")
    frequency = extractor.analyze_frequency(test_text)
    for item, count in frequency.items():
        print(f"   {item}: {count}")
    
    # Advanced frequency analysis
    print("\n3. Advanced Frequency Analysis (grouped):")
    advanced_freq = extractor.analyze_frequency_advanced(test_text, group_similar=True)
    for item, count in advanced_freq.items():
        print(f"   {item}: {count}")
    
    # Line occurrences
    print("\n4. Line Occurrences:")
    line_occurrences = extractor.analyze_frequency_by_line(test_text)
    for item, lines in line_occurrences.items():
        print(f"   {item}: lines {lines}")
    
    # Full metadata
    print("\n5. Full Metadata:")
    metadata = extractor.extract_with_metadata(test_text)
    print(f"   Total findings: {metadata['summary']['total_findings']}")
    print(f"   Total occurrences: {metadata['summary']['total_occurrences']}")
    print(f"   Exceptions: {metadata['summary']['exceptions_count']}")
    print(f"   Errors: {metadata['summary']['errors_count']}")

if __name__ == "__main__":
    main()
