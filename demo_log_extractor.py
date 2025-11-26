import re
from model.log_extractor import LogExtractor

# Demonstration of the fix
def demonstrate_frequency_fix():
    """Demonstrate the fixed frequency counting."""
    
    test_text = """
    com.mendix.MendixRuntimeException: First occurrence
    com.mendix.MendixRuntimeException: Second occurrence  
    MendixRuntimeException: Third occurrence (simple name)
    java.lang.NullPointerException: Fourth occurrence
    at com.mendix.internal.Processor.handle(MendixRuntimeException.java:123)
    """
    
    print("Frequency Counting Fix Demonstration")
    print("=" * 60)
    print("Test Text:")
    print(test_text)
    print("\n" + "=" * 60)
    
    # Test with different configurations
    configs = [
        ("Prefer simple names", True),
        ("Prefer package names", False),
    ]
    
    for config_name, prefer_simple in configs:
        print(f"\n{config_name} (prefer_simple_names={prefer_simple}):")
        
        extractor = LogExtractor(
            include_package_names=True,
            prefer_simple_names=prefer_simple,
            keywords=["error", "exception"]
        )
        
        findings = extractor.extract_from_text(test_text)
        print(f"  Findings: {findings}")
        
        frequency = extractor.analyze_frequency(test_text)
        print(f"  Frequency: {frequency}")
        
        # Manual verification
        manual_simple = len(re.findall(r'\bMendixRuntimeException\b', test_text))
        manual_package = len(re.findall(r'\bcom\.mendix\.MendixRuntimeException\b', test_text))
        manual_nullpointer = len(re.findall(r'\bjava\.lang\.NullPointerException\b', test_text))
        
        print(f"  Manual counts:")
        print(f"    MendixRuntimeException: {manual_simple}")
        print(f"    com.mendix.MendixRuntimeException: {manual_package}")
        print(f"    java.lang.NullPointerException: {manual_nullpointer}")
        
        # Check if counts are correct
        expected_total = manual_simple + manual_package + manual_nullpointer
        actual_total = sum(frequency.values())
        status = "✓ CORRECT" if actual_total == expected_total else f"✗ WRONG (expected {expected_total}, got {actual_total})"
        print(f"  Total count: {actual_total} {status}")

def test_double_counting_scenario():
    """Test the specific double-counting scenario."""
    
    # This text would cause double-counting in the old version
    problem_text = "com.mendix.MendixRuntimeException com.mendix.MendixRuntimeException"
    
    print("\n" + "=" * 60)
    print("Double-Counting Scenario Test")
    print("=" * 60)
    print(f"Problem text: '{problem_text}'")
    
    extractor = LogExtractor(
        include_package_names=True,
        prefer_simple_names=True,
        keywords=[]
    )
    
    # Old buggy behavior (simulated)
    old_frequency_buggy = {
        'MendixRuntimeException': 4,  # Wrong! Should be 2
        'com.mendix.MendixRuntimeException': 2  # Also wrong due to double-counting
    }
    
    # New fixed behavior
    new_frequency_fixed = extractor.analyze_frequency(problem_text)
    
    print(f"Old (buggy) frequency: {old_frequency_buggy}")
    print(f"New (fixed) frequency: {new_frequency_fixed}")
    
    manual_count = len(re.findall(r'\bcom\.mendix\.MendixRuntimeException\b', problem_text))
    print(f"Manual count of occurrences: {manual_count}")
    
    if new_frequency_fixed.get('MendixRuntimeException', 0) == manual_count:
        print("✓ FIXED: No more double-counting!")
    else:
        print("✗ STILL BROKEN: Double-counting persists")

if __name__ == "__main__":
    demonstrate_frequency_fix()
    test_double_counting_scenario()
    
    # Test with metadata extraction
    print("\n" + "=" * 60)
    print("Metadata Extraction with Fixed Frequency")
    print("=" * 60)
    
    test_text = "com.mendix.MendixRuntimeException com.mendix.MendixRuntimeException"
    extractor = LogExtractor(
        include_package_names=True,
        prefer_simple_names=True
    )
    
    metadata = extractor.extract_with_metadata(test_text)
    print(f"Total occurrences: {metadata['summary']['total_occurrences']}")
    print(f"Frequency: {metadata['frequency']}")
    print(f"Detailed frequency: {metadata['detailed_frequency']}")