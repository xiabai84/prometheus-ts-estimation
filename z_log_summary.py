import re
from collections import Counter

def extract_exceptions_and_errors_comprehensive(text):
    """
    Comprehensive extraction including package names, stack traces, and keyword exceptions
    """
    patterns = [
        # Exception patterns with package names
        (r'(\b\w*?Error\b(?:\.\w+)*)', 'error_with_package'),
        (r'(\b\w*?Exception\b(?:\.\w+)*)', 'exception_with_package'),
        (r'(\bException\b(?:\.\w+)*)', 'generic_exception_with_package'),
        
        # Stack trace patterns
        (r'at\s+([a-zA-Z0-9_.]+\.[a-zA-Z0-9_]+(?:Error|Exception))', 'stack_trace_full'),
        (r'at\s+([a-zA-Z0-9_.]+(?:Error|Exception))', 'stack_trace_simple'),
        (r'Caused by:\s*([a-zA-Z0-9_.]+\.[a-zA-Z0-9_]+(?:Error|Exception))', 'caused_by'),
        
        # Exception context patterns
        (r'Exception:\s*([^\n]+)', 'exception_description'),
        (r'Error:\s*([^\n]+)', 'error_description'),
        
        # Basic patterns
        (r'raise\s+([a-zA-Z0-9_.]+)', 'raise_stmt'),
        (r'catch\s*\(([a-zA-Z0-9_.]+)\s+\w+\)', 'catch_stmt'),
        (r'except\s+([a-zA-Z0-9_.]+)', 'except_stmt'),
        (r'throws\s+([a-zA-Z0-9_.]+)', 'throws_decl'),
        (r'throw new\s+([a-zA-Z0-9_.]+)', 'throw_new'),
        (r'class\s+([a-zA-Z0-9_.]*Exception)', 'class_def'),
        
        # Keyword patterns (NEW - for generic "exception" mentions)
        (r'\b(exception)\b', 'keyword_exception'),
        (r'\b(error)\b', 'keyword_error'),
    ]
    
    findings = set()
    
    for pattern, pattern_type in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if pattern_type in ['keyword_exception', 'keyword_error']:
                # For keyword matches, add the keyword itself
                findings.add(match.capitalize())
            elif pattern_type in ['exception_description', 'error_description']:
                # For descriptions, extract potential exception names from the description
                extracted = extract_from_description(match)
                findings.update(extracted)
            else:
                # For other patterns, add the matched name directly
                name = match.strip()
                if is_valid_exception_or_error_name(name):
                    findings.add(name)
    
    # Additional processing for lines containing exception/error keywords
    keyword_findings = extract_from_keyword_context(text)
    findings.update(keyword_findings)
    
    return sorted(list(findings))

def extract_from_description(description):
    """
    Extract potential exception names from exception/error descriptions
    """
    findings = set()
    
    # Look for exception-like patterns in descriptions
    exception_matches = re.findall(r'([A-Z][a-zA-Z]*(?:Error|Exception))', description)
    for match in exception_matches:
        if is_valid_exception_or_error_name(match):
            findings.add(match)
    
    return findings

def extract_from_keyword_context(text):
    """
    Extract findings from lines that contain exception/error keywords
    """
    findings = set()
    lines = text.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        
        # If line contains exception/error keywords but no specific exception was caught
        if ('exception' in line_lower or 'error' in line_lower) and not re.search(r'[A-Z][a-zA-Z]*(?:Error|Exception)', line):
            # Check if it's a generic mention
            if 'exception' in line_lower and not any('exception' in finding.lower() for finding in findings):
                findings.add('Exception')
            if 'error' in line_lower and not any('error' in finding.lower() for finding in findings):
                findings.add('Error')
        
        # Extract full package paths from stack traces
        package_matches = re.findall(r'([a-zA-Z0-9_.]+\.[A-Z][a-zA-Z]*(?:Error|Exception))', line)
        for match in package_matches:
            if is_valid_exception_or_error_name(match):
                findings.add(match)
    
    return findings

def is_valid_exception_or_error_name(name):
    """
    Check if the extracted name is likely a valid exception or error name
    """
    if not name or len(name) < 3:
        return False
    
    # Common non-exception words to exclude
    exclude_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'with', 'from', 'by', 'as', 'is', 'are', 'was', 'were',
        'this', 'that', 'these', 'those', 'what', 'when', 'where'
    }
    if name.lower() in exclude_words:
        return False
    
    # Allow package names and generic keywords
    if name in ['Exception', 'Error']:
        return True
    
    # Must contain Error, Exception, or be a known generic exception/error
    valid_keywords = ['error', 'exception', 'throw', 'catch', 'raise', 'fail']
    if any(keyword in name.lower() for keyword in valid_keywords):
        return True
    
    # Pattern-based validation (more permissive for package names)
    if ('.' in name or  # Allow package names
        name.endswith('Error') or 
        name.endswith('Exception') or 
        name == 'Exception' or 
        name == 'Error' or
        name.startswith('Err') or 
        'Fail' in name):
        return True
    
    # CamelCase validation for custom exceptions/errors
    if re.match(r'^[A-Z][a-zA-Z]*(Error|Exception|Fail)', name):
        return True
    
    return False

def categorize_findings_detailed(findings):
    """
    Categorize findings with detailed analysis
    """
    exceptions = []
    errors = []
    packages = []
    keywords = []
    
    for finding in findings:
        finding_lower = finding.lower()
        
        if '.' in finding:
            packages.append(finding)
        elif finding in ['Exception', 'Error']:
            keywords.append(finding)
        elif 'exception' in finding_lower:
            exceptions.append(finding)
        elif 'error' in finding_lower:
            errors.append(finding)
        else:
            keywords.append(finding)
    
    return {
        'exceptions': sorted(exceptions),
        'errors': sorted(errors),
        'packages': sorted(packages),
        'keywords': sorted(keywords)
    }

# Test with comprehensive text including package names, stack traces, and keywords
test_text = """
2023-10-01 10:00:00 ERROR: FileNotFoundError: File not found
2023-10-01 10:01:00 INFO: Process started successfully
2023-10-01 10:02:00 WARNING: An exception occurred during processing
2023-10-01 10:03:00 ERROR: There was an error in the system
2023-10-01 10:04:00 java.lang.NullPointerException: Cannot invoke method on null object
    at com.example.MyClass.processData(MyClass.java:123)
    at com.example.OtherClass.execute(OtherClass.java:45)
2023-10-01 10:05:00 Caused by: org.springframework.dao.DataAccessException: Database error
    at org.springframework.jdbc.core.JdbcTemplate.execute(JdbcTemplate.java:234)
2023-10-01 10:06:00 javax.persistence.PersistenceException: Entity not found
2023-10-01 10:07:00 com.company.custom.CustomBusinessException: Business rule violation
2023-10-01 10:08:00 python NameError: name 'undefined_var' is not defined
2023-10-01 10:09:00 raise ValueError("Invalid value provided")
2023-10-01 10:10:00 catch (java.io.IOException e)
2023-10-01 10:11:00 except RuntimeError as e:
2023-10-01 10:12:00 This line has no exception or error
2023-10-01 10:13:00 Some generic exception message without specific type
2023-10-01 10:14:00 Generic error without specific error class
2023-10-01 10:15:00 org.hibernate.HibernateException: Session is closed
    at org.hibernate.internal.SessionImpl.checkOpen(SessionImpl.java:567)
"""

# Extract all findings
findings = extract_exceptions_and_errors_comprehensive(test_text)

# Categorize the findings
categorized = categorize_findings_detailed(findings)

# Print results
print("ALL FINDINGS:")
for i, finding in enumerate(findings, 1):
    print(f"  {i:2d}. {finding}")

print("\nDETAILED CATEGORIZATION:")
print("Exceptions:")
for exc in categorized['exceptions']:
    print(f"  - {exc}")

print("\nErrors:")
for err in categorized['errors']:
    print(f"  - {err}")

print("\nPackages (Full class paths):")
for pkg in categorized['packages']:
    print(f"  - {pkg}")

print("\nKeywords:")
for kw in categorized['keywords']:
    print(f"  - {kw}")

# Summary statistics
print(f"\nSUMMARY:")
print(f"Total findings: {len(findings)}")
print(f"Exceptions: {len(categorized['exceptions'])}")
print(f"Errors: {len(categorized['errors'])}")
print(f"Packages: {len(categorized['packages'])}")
print(f"Keywords: {len(categorized['keywords'])}")