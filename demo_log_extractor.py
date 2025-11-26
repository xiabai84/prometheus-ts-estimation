from model.log_extractor import LogExtractor
import re

def demonstrate_overcounting_fix():
    """Demonstrate how the fix prevents overcounting."""
    
    test_text = """
    com.mendix.MendixRuntimeException: Business logic error
    MendixRuntimeException: Same exception without package
    at com.mendix.internal.Processor.handle(MendixRuntimeException.java:123)
    raise MendixRuntimeException("error")
    catch (MendixRuntimeException ex)
    Another MendixRuntimeException in the same line MendixRuntimeException
    """
    
    print("Overcounting Fix Demonstration")
    print("=" * 60)
    print("Test Text:")
    print(test_text)
    print("\n" + "=" * 60)
    
    extractor = LogExtractor(
        include_package_names=True,
        prefer_simple_names=True,
        keywords=["error", "exception"]
    )
    
    # Show what gets extracted
    findings = extractor.extract_from_text(test_text)
    print(f"Extracted findings: {findings}")
    
    # Show frequency with different methods
    print("\nFrequency Analysis Methods:")
    
    freq_basic = extractor.analyze_frequency(test_text)
    print(f"Basic frequency: {freq_basic}")
    
    freq_exact = extractor.analyze_frequency_exact(test_text)
    print(f"Exact frequency: {freq_exact}")
    
    # Manual verification
    print("\nManual Verification:")
    manual_count_simple = len(re.findall(r'\bMendixRuntimeException\b', test_text))
    manual_count_package = len(re.findall(r'\bcom\.mendix\.MendixRuntimeException\b', test_text))
    print(f"Manual count - MendixRuntimeException: {manual_count_simple}")
    print(f"Manual count - com.mendix.MendixRuntimeException: {manual_count_package}")
    
    # Show pattern matches
    print("\nPattern Analysis:")
    for pattern, pattern_type, priority in extractor._compiled_patterns:
        matches = list(pattern.finditer(test_text))
        if matches:
            print(f"Pattern '{pattern_type}' (priority {priority}): {len(matches)} matches")
            for match in matches[:2]:  # Show first 2 matches
                print(f"  - '{match.group()}' at position {match.span()}")

def test_multiple_pattern_scenarios():
    """Test scenarios where multiple patterns could match the same text."""
    
    test_cases = [
        {
            'name': 'Package name in stack trace',
            'text': 'at com.mendix.MendixRuntimeException.handle(File.java:123)',
            'expected': ['MendixRuntimeException']
        },
        {
            'name': 'Simple name in raise statement',
            'text': 'raise MendixRuntimeException("error")',
            'expected': ['MendixRuntimeException']
        },
        {
            'name': 'Package name with colon',
            'text': 'com.mendix.MendixRuntimeException: error message',
            'expected': ['MendixRuntimeException']
        },
        {
            'name': 'Simple name with colon', 
            'text': 'MendixRuntimeException: error message',
            'expected': ['MendixRuntimeException']
        },
        {
            'name': 'Mixed occurrences',
            'text': 'com.mendix.MendixRuntimeException and MendixRuntimeException',
            'expected': ['MendixRuntimeException']
        }
    ]
    
    extractor = LogExtractor(
        include_package_names=False,  # Only want simple names for clarity
        prefer_simple_names=True,
        keywords=[]
    )
    
    print("\n" + "=" * 60)
    print("Multiple Pattern Scenario Tests")
    print("=" * 60)
    
    for case in test_cases:
        print(f"\nTest: {case['name']}")
        print(f"Text: '{case['text']}'")
        
        findings = extractor.extract_from_text(case['text'])
        frequency = extractor.analyze_frequency_exact(case['text'])
        
        print(f"Expected: {case['expected']}")
        print(f"Actual findings: {findings}")
        print(f"Frequency: {frequency}")
        
        status = "✓ PASS" if findings == case['expected'] else "✗ FAIL"
        print(f"Status: {status}")

if __name__ == "__main__":
    demonstrate_overcounting_fix()
    test_multiple_pattern_scenarios()