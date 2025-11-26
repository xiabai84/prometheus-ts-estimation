import re
from collections import Counter

def extract_exceptions_and_errors_enhanced(text):
    """
    Enhanced extraction for both exceptions and error-related information
    """
    patterns = [
        # Exception patterns
        (r'(\b\w*?Error\b):', 'error_suffix'),
        (r'(\b\w*?Exception\b):', 'exception_suffix'),
        (r'(\bException\b):', 'generic_exception'),
        (r'at\s+.*?(\b\w*?Error\b)', 'stack_error'),
        (r'at\s+.*?(\b\w*?Exception\b)', 'stack_exception'),
        (r'at\s+.*?(\bException\b)', 'stack_generic'),
        (r'raise\s+(\w+)', 'raise_stmt'),
        (r'catch\s*\((\w+)\s+\w+\)', 'catch_stmt'),
        (r'except\s+(\w+)', 'except_stmt'),
        (r'throws\s+(\w+)', 'throws_decl'),
        (r'throw new\s+(\w+)', 'throw_new'),
        (r'class\s+(\w+.*Exception)', 'class_def'),
        
        # Error-specific patterns (NEW)
        (r'(\b\w*?Error\b)\s', 'error_standalone'),
        (r'ERROR:\s*(\w*?Error)', 'error_prefix'),
        (r'error:\s*(\w*?Error)', 'error_lower_prefix'),
        (r'(\bError\b):', 'generic_error'),
        (r'(\bRuntimeError\b)', 'runtime_error'),
        (r'(\bAssertionError\b)', 'assertion_error'),
    ]
    
    exceptions_and_errors = set()
    
    for pattern, pattern_type in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            name = match.strip()
            
            if is_valid_exception_or_error_name(name):
                exceptions_and_errors.add(name)
    
    # Additional processing for error messages and contexts
    error_contexts = extract_error_contexts(text)
    exceptions_and_errors.update(error_contexts)
    
    return sorted(list(exceptions_and_errors))

def is_valid_exception_or_error_name(name):
    """
    Check if the extracted name is likely a valid exception or error name
    """
    if not name or len(name) < 3:
        return False
    
    # Common non-exception words to exclude
    exclude_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'with', 'from', 'by', 'as', 'is', 'are', 'was', 'were'
    }
    if name.lower() in exclude_words:
        return False
    
    # Must contain Error, Exception, or be a known generic exception/error
    valid_keywords = ['error', 'exception', 'throw', 'catch', 'raise', 'fail']
    if any(keyword in name.lower() for keyword in valid_keywords):
        return True
    
    # Common generic exception and error names
    generic_names = {
        'Exception', 'Error', 'Throwable', 'RuntimeException',
        'RuntimeError', 'AssertionError', 'SystemError'
    }
    if name in generic_names:
        return True
    
    # Pattern-based validation
    if (name.endswith('Error') or name.endswith('Exception') or 
        name == 'Exception' or name == 'Error' or
        name.startswith('Err') or 'Fail' in name):
        return True
    
    # CamelCase validation for custom exceptions/errors
    if re.match(r'^[A-Z][a-zA-Z]*(Error|Exception|Fail)', name):
        return True
    
    return False

def extract_error_contexts(text):
    """
    Extract additional error-related information from context
    """
    error_contexts = set()
    lines = text.split('\n')
    
    for line in lines:
        # Look for error severity levels
        if re.search(r'\b(ERROR|Error|error)\b', line):
            # Extract potential error types from error lines
            error_matches = re.findall(r'(\b\w*?Error\b|\b\w*?Exception\b)', line)
            for match in error_matches:
                if is_valid_exception_or_error_name(match):
                    error_contexts.add(match)
        
        # Look for error codes or specific error patterns
        error_code_match = re.search(r'error[:\s]*([A-Z][a-zA-Z]*)', line, re.IGNORECASE)
        if error_code_match:
            potential_error = error_code_match.group(1)
            if is_valid_exception_or_error_name(potential_error):
                error_contexts.add(potential_error)
    
    return error_contexts

def categorize_findings(findings):
    """
    Categorize the extracted findings into exceptions and errors
    """
    exceptions = []
    errors = []
    others = []
    
    for finding in findings:
        finding_lower = finding.lower()
        if 'exception' in finding_lower:
            exceptions.append(finding)
        elif 'error' in finding_lower:
            errors.append(finding)
        else:
            others.append(finding)
    
    return {
        'exceptions': sorted(exceptions),
        'errors': sorted(errors),
        'others': sorted(others)
    }

# Test with comprehensive text containing both exceptions and errors
test_text = """
2023-10-01 10:00:00 ERROR: FileNotFoundError: File not found
2023-10-01 10:01:00 INFO: Process started successfully
2023-10-01 10:02:00 ERROR: Exception: Generic exception occurred
2023-10-01 10:03:00 at java.lang.Exception: Some stack trace
2023-10-01 10:04:00 raise ValueError("Invalid value")
2023-10-01 10:05:00 catch (IOException e)
2023-10-01 10:06:00 except RuntimeError:
2023-10-01 10:07:00 CustomException: Custom error message
2023-10-01 10:08:00 throws SQLException
2023-10-01 10:09:00 throw new IllegalArgumentException("Invalid argument")
2023-10-01 10:10:00 ERROR: ConnectionError: Failed to connect
2023-10-01 10:11:00 error: ValidationError: Invalid input data
2023-10-01 10:12:00 RuntimeError: Unexpected runtime issue
2023-10-01 10:13:00 AssertionError: Assertion failed
2023-10-01 10:14:00 SystemError: System level error
2023-10-01 10:15:00 Some random text without errors
2023-10-01 10:16:00 Error: Generic error without specific type
2023-10-01 10:17:00 at com.example.NetworkError: Network operation failed
"""

# Extract all findings
findings = extract_exceptions_and_errors_enhanced(test_text)

# Categorize the findings
categorized = categorize_findings(findings)

# Print results
print("ALL FINDINGS:")
for finding in findings:
    print(f"  - {finding}")

print("\nCATEGORIZED RESULTS:")
print("Exceptions:")
for exc in categorized['exceptions']:
    print(f"  - {exc}")

print("\nErrors:")
for err in categorized['errors']:
    print(f"  - {err}")

print("\nOthers:")
for other in categorized['others']:
    print(f"  - {other}")

# Additional analysis
print(f"\nSUMMARY:")
print(f"Total findings: {len(findings)}")
print(f"Exceptions: {len(categorized['exceptions'])}")
print(f"Errors: {len(categorized['errors'])}")
print(f"Others: {len(categorized['others'])}")