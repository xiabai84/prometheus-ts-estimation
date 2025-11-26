import re
from collections import Counter, defaultdict
from typing import List, Dict, Set, Any, Optional, Tuple

class LogExtractor:
    """
    Complete library for extracting exceptions and errors from text logs.
    Includes metadata extraction and prevents overcounting.
    """
    
    def __init__(self, 
                 include_package_names: bool = True, 
                 keywords: Optional[List[str]] = None,
                 prefer_simple_names: bool = True):
        """
        Initialize the LogExtractor.
        
        Args:
            include_package_names: Whether to include full package paths
            keywords: List of keywords to search for
            prefer_simple_names: Whether to prefer simple class names over full package names
        """
        self.include_package_names = include_package_names
        self.keywords = keywords or []
        self.prefer_simple_names = prefer_simple_names
        self._compiled_patterns = self._compile_prioritized_patterns()
        self._cache = {}
    
    def _compile_prioritized_patterns(self) -> List[Tuple[re.Pattern, str, int]]:
        """
        Compile patterns with priorities to prevent overmatching.
        Higher priority patterns are processed first.
        """
        patterns = []
        
        # Define patterns with priorities (higher number = higher priority)
        base_patterns = [
            # Highest priority: Full package names in stack traces (most specific)
            (r'at\s+(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))', 'stack_trace_full', 100),
            (r'Caused by:\s*(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))', 'caused_by', 100),
            
            # High priority: Code statements
            (r'(?:raise|throw)\s+([a-zA-Z0-9_.]+(?:Error|Exception))', 'raise_throw', 90),
            (r'(?:throws|catch|except)\s+([a-zA-Z0-9_.]+(?:Error|Exception))', 'declaration', 90),
            
            # Medium priority: Full package names in messages
            (r'\b(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))(?:\s*:|)', 'package_with_colon', 80),
            (r'\b(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))\b', 'package_general', 70),
            
            # Lower priority: Simple exception names
            (r'\b([A-Z][a-zA-Z0-9_]*(?:Error|Exception))(?:\s*:|)', 'simple_with_colon', 60),
            (r'\b([A-Z][a-zA-Z0-9_]*(?:Error|Exception))\b', 'simple_general', 50),
            
            # Context patterns (lowest priority)
            (r'(?:Exception|Error):\s*[^\n]+', 'exception_description', 40),
        ]
        
        # Compile base patterns
        for pattern, pattern_type, priority in base_patterns:
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                patterns.append((compiled, pattern_type, priority))
            except re.error as e:
                print(f"Warning: Failed to compile pattern '{pattern}': {e}")
                continue
        
        # Add keyword patterns with medium priority
        for keyword in self.keywords:
            try:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                compiled = re.compile(pattern, re.IGNORECASE)
                patterns.append((compiled, f'keyword_{keyword}', 75))
            except re.error:
                continue
        
        # Sort by priority (highest first)
        patterns.sort(key=lambda x: x[2], reverse=True)
        return patterns
    
    def extract_from_text(self, text: str, use_cache: bool = True) -> List[str]:
        """
        Extract exceptions without duplicates using pattern prioritization.
        
        Args:
            text: Input text to analyze
            use_cache: Whether to use caching
            
        Returns:
            List of unique exception/error names found
        """
        if use_cache and text in self._cache:
            return self._cache[text]
        
        findings = set()
        processed_positions = set()
        
        for pattern, pattern_type, priority in self._compiled_patterns:
            for match in pattern.finditer(text):
                start_pos, end_pos = match.span()
                
                if self._is_position_processed(start_pos, end_pos, processed_positions):
                    continue
                
                processed_positions.add((start_pos, end_pos))
                
                if pattern_type.startswith('keyword_'):
                    findings.add(match.group().capitalize())
                elif pattern_type == 'exception_description':
                    extracted = self._extract_from_description_safe(match.group())
                    findings.update(extracted)
                else:
                    names = self._extract_names_from_match(match, pattern_type)
                    for name in names:
                        if name and self._is_valid_exception_or_error_name(name):
                            findings.add(name)
        
        result = sorted(list(findings))
        
        if use_cache:
            if len(self._cache) > 100:
                self._cache.clear()
            self._cache[text] = result
        
        return result
    
    def extract_from_file(self, file_path: str, encoding: str = 'utf-8') -> List[str]:
        """
        Extract exceptions and errors from a file.
        
        Args:
            file_path: Path to the file to analyze
            encoding: File encoding (default: utf-8)
            
        Returns:
            List of unique exception/error names found
        """
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                content = file.read()
        
        return self.extract_from_text(content, use_cache=False)
    
    def extract_with_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract exceptions with comprehensive metadata.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing findings and metadata
        """
        findings = self.extract_from_text(text)
        frequency = self.analyze_frequency_exact(text)
        line_occurrences = self.analyze_frequency_by_line(text)
        categorized = self._categorize_findings(findings)
        package_analysis = self.get_package_analysis(text)
        
        return {
            'findings': findings,
            'frequency': frequency,
            'line_occurrences': line_occurrences,
            'categorized': categorized,
            'package_analysis': package_analysis,
            'summary': {
                'total_findings': len(findings),
                'total_occurrences': sum(frequency.values()),
                'exceptions_count': len(categorized['exceptions']),
                'errors_count': len(categorized['errors']),
                'packages_count': len(categorized['package_names']),
                'keywords_count': len(categorized['keywords']),
                'simple_names_count': len(categorized['simple_names'])
            }
        }
    
    def analyze_frequency(self, text: str) -> Dict[str, int]:
        """
        Accurate frequency analysis without overcounting.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with accurate frequency counts
        """
        frequency_counter = Counter()
        processed_positions = set()
        
        for pattern, pattern_type, priority in self._compiled_patterns:
            for match in pattern.finditer(text):
                start_pos, end_pos = match.span()
                
                if self._is_position_processed(start_pos, end_pos, processed_positions):
                    continue
                
                processed_positions.add((start_pos, end_pos))
                
                if pattern_type.startswith('keyword_'):
                    frequency_counter[match.group().capitalize()] += 1
                elif pattern_type == 'exception_description':
                    extracted = self._extract_from_description_safe(match.group())
                    for name in extracted:
                        frequency_counter[name] += 1
                else:
                    names = self._extract_names_from_match(match, pattern_type)
                    for name in names:
                        if name and self._is_valid_exception_or_error_name(name):
                            if self.prefer_simple_names and '.' in name:
                                simple_name = name.split('.')[-1]
                                frequency_counter[simple_name] += 1
                                if self.include_package_names:
                                    frequency_counter[name] += 1
                            else:
                                frequency_counter[name] += 1
        
        return dict(frequency_counter.most_common())
    
    def analyze_frequency_exact(self, text: str) -> Dict[str, int]:
        """
        Exact frequency analysis using extraction-based counting.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with exact frequency counts
        """
        findings_with_context = self._extract_findings_with_context(text)
        frequency_counter = Counter()
        
        for finding, context in findings_with_context:
            if self.prefer_simple_names and '.' in finding:
                simple_name = finding.split('.')[-1]
                frequency_counter[simple_name] += 1
                if self.include_package_names:
                    frequency_counter[finding] += 1
            else:
                frequency_counter[finding] += 1
        
        return dict(frequency_counter.most_common())
    
    def analyze_frequency_by_line(self, text: str) -> Dict[str, List[int]]:
        """
        Analyze frequency with line number information.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with findings and their line numbers
        """
        line_occurrences = defaultdict(list)
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            findings = self.extract_from_text(line, use_cache=False)
            for finding in findings:
                line_occurrences[finding].append(line_num)
        
        return dict(line_occurrences)
    
    def analyze_frequency_detailed(self, text: str) -> Dict[str, Dict[str, Any]]:
        """
        Detailed frequency analysis with additional information.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with detailed frequency information
        """
        basic_frequency = self.analyze_frequency_exact(text)
        package_mappings = self._extract_package_mappings(text)
        findings = self.extract_from_text(text)
        categorized = self._categorize_findings(findings)
        
        detailed_frequency = {}
        total_occurrences = sum(basic_frequency.values())
        
        for item, count in basic_frequency.items():
            is_package_name = '.' in item
            simple_name = item.split('.')[-1] if is_package_name else item
            
            related_names = []
            if is_package_name:
                related_names.append(simple_name)
            elif simple_name in package_mappings:
                related_names.extend(package_mappings[simple_name])
            
            detailed_frequency[item] = {
                'count': count,
                'is_package_name': is_package_name,
                'simple_name': simple_name,
                'related_names': related_names,
                'category': self._categorize_single_item(item, categorized),
                'percentage': round((count / total_occurrences * 100), 2) if total_occurrences > 0 else 0,
                'full_package_names': package_mappings.get(simple_name, [])
            }
        
        return detailed_frequency
    
    def get_package_analysis(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive package name analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with package analysis information
        """
        findings = self.extract_from_text(text)
        package_mappings = self._extract_package_mappings(text)
        
        simple_names = set()
        package_names = set()
        
        for finding in findings:
            if '.' in finding:
                package_names.add(finding)
                simple_names.add(finding.split('.')[-1])
            else:
                simple_names.add(finding)
        
        multi_package_names = {}
        for simple_name, packages in package_mappings.items():
            if len(packages) > 1:
                multi_package_names[simple_name] = packages
        
        return {
            'simple_names': sorted(list(simple_names)),
            'package_names': sorted(list(package_names)),
            'package_mappings': package_mappings,
            'multi_package_names': multi_package_names,
            'summary': {
                'total_simple_names': len(simple_names),
                'total_package_names': len(package_names),
                'names_with_packages': len(package_mappings),
                'names_with_multiple_packages': len(multi_package_names)
            }
        }
    
    def _is_position_processed(self, start: int, end: int, processed_positions: Set[tuple]) -> bool:
        """Check if a text position has already been processed."""
        for proc_start, proc_end in processed_positions:
            if start >= proc_start and end <= proc_end:
                return True
            if start < proc_end and end > proc_start:
                return True
        return False
    
    def _extract_findings_with_context(self, text: str) -> List[Tuple[str, str]]:
        """Extract findings with their context for accurate counting."""
        findings_with_context = []
        processed_positions = set()
        
        for pattern, pattern_type, priority in self._compiled_patterns:
            for match in pattern.finditer(text):
                start_pos, end_pos = match.span()
                
                if self._is_position_processed(start_pos, end_pos, processed_positions):
                    continue
                
                processed_positions.add((start_pos, end_pos))
                
                if pattern_type.startswith('keyword_'):
                    findings_with_context.append((match.group().capitalize(), pattern_type))
                elif pattern_type == 'exception_description':
                    extracted = self._extract_from_description_safe(match.group())
                    for name in extracted:
                        findings_with_context.append((name, pattern_type))
                else:
                    names = self._extract_names_from_match(match, pattern_type)
                    for name in names:
                        if name and self._is_valid_exception_or_error_name(name):
                            findings_with_context.append((name, pattern_type))
        
        return findings_with_context
    
    def _extract_names_from_match(self, match: re.Match, pattern_type: str) -> List[str]:
        """Extract names from match based on pattern type."""
        matched_text = match.group()
        groups = match.groups()
        
        if pattern_type in ['stack_trace_full', 'caused_by']:
            full_name = groups[0] if groups else matched_text
            return self._get_name_variants(full_name)
        
        elif pattern_type in ['raise_throw', 'declaration']:
            exception_name = groups[0] if groups else matched_text.split()[-1]
            return self._get_name_variants(exception_name)
        
        elif pattern_type in ['package_with_colon', 'package_general']:
            full_name = groups[0] if groups else matched_text
            return self._get_name_variants(full_name)
        
        elif pattern_type in ['simple_with_colon', 'simple_general']:
            simple_name = groups[0] if groups else matched_text
            return [simple_name]
        
        return [matched_text]
    
    def _get_name_variants(self, name: str) -> List[str]:
        """Get both full and simple name variants."""
        if '.' in name:
            full_name = name
            simple_name = name.split('.')[-1]
            if self.prefer_simple_names:
                return [simple_name, full_name] if self.include_package_names else [simple_name]
            else:
                return [full_name, simple_name]
        else:
            return [name]
    
    def _extract_from_description_safe(self, description: str) -> Set[str]:
        """Safe extraction from descriptions without overcounting."""
        findings = set()
        
        exception_pattern = re.compile(r'\b([A-Z][a-zA-Z0-9_]*(?:Error|Exception))\b')
        package_pattern = re.compile(r'\b(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))\b')
        
        for match in exception_pattern.finditer(description):
            name = match.group(1)
            if self._is_valid_exception_or_error_name(name):
                findings.add(name)
        
        for match in package_pattern.finditer(description):
            name = match.group(1)
            if self._is_valid_exception_or_error_name(name):
                if self.prefer_simple_names:
                    findings.add(name.split('.')[-1])
                else:
                    findings.add(name)
        
        return findings
    
    def _extract_package_mappings(self, text: str) -> Dict[str, List[str]]:
        """Extract mappings between simple names and their full package names."""
        package_mappings = defaultdict(list)
        package_pattern = re.compile(r'\b(([a-zA-Z0-9_]+\.)+[A-Z][a-zA-Z0-9_]*(?:Error|Exception))\b')
        
        for match in package_pattern.finditer(text):
            full_name = match.group(1)
            simple_name = full_name.split('.')[-1]
            
            if full_name not in package_mappings[simple_name]:
                package_mappings[simple_name].append(full_name)
        
        return dict(package_mappings)
    
    def _categorize_findings(self, findings: List[str]) -> Dict[str, List[str]]:
        """Categorize findings into different types."""
        categories = {
            'exceptions': [],
            'errors': [],
            'package_names': [],
            'simple_names': [],
            'keywords': []
        }
        
        for finding in findings:
            finding_lower = finding.lower()
            
            if '.' in finding:
                categories['package_names'].append(finding)
                simple_name = finding.split('.')[-1]
                if 'exception' in simple_name.lower():
                    categories['exceptions'].append(simple_name)
                elif 'error' in simple_name.lower():
                    categories['errors'].append(simple_name)
                else:
                    categories['simple_names'].append(simple_name)
            elif finding.lower() in [kw.lower() for kw in self.keywords]:
                categories['keywords'].append(finding)
            elif 'exception' in finding_lower:
                categories['exceptions'].append(finding)
            elif 'error' in finding_lower:
                categories['errors'].append(finding)
            else:
                categories['simple_names'].append(finding)
        
        for category in categories:
            categories[category] = sorted(list(set(categories[category])))
        
        return categories
    
    def _categorize_single_item(self, item: str, categorized: Dict[str, List[str]]) -> str:
        """Categorize a single item."""
        if item in categorized['package_names']:
            return 'package_name'
        elif item in categorized['exceptions']:
            return 'exception'
        elif item in categorized['errors']:
            return 'error'
        elif item in categorized['keywords']:
            return 'keyword'
        else:
            return 'simple_name'
    
    def _is_valid_exception_or_error_name(self, name: str) -> bool:
        """Validation that handles both package and simple names."""
        if not name or len(name) < 3:
            return False
        
        simple_name = name.split('.')[-1] if '.' in name else name
        
        exclude_words = {'the', 'and', 'or', 'but', 'for', 'with', 'from', 'this', 'that'}
        if simple_name.lower() in exclude_words:
            return False
        
        if simple_name.lower() in [kw.lower() for kw in self.keywords]:
            return True
        
        if (simple_name.endswith(('Error', 'Exception')) or
            simple_name in ('Exception', 'Error') or
            any(keyword in simple_name.lower() for keyword in ['error', 'exception', 'fail'])):
            return True
        
        if re.match(r'^[A-Z][a-zA-Z0-9]*(Error|Exception|Fail)', simple_name):
            return True
        
        return False
    
    # Utility methods
    def update_keywords(self, new_keywords: List[str]) -> None:
        self.keywords = new_keywords
        self._compiled_patterns = self._compile_prioritized_patterns()
        self._cache.clear()
    
    def add_keyword(self, keyword: str) -> None:
        if keyword not in self.keywords:
            self.keywords.append(keyword)
            self._compiled_patterns = self._compile_prioritized_patterns()
            self._cache.clear()
    
    def remove_keyword(self, keyword: str) -> None:
        if keyword in self.keywords:
            self.keywords.remove(keyword)
            self._compiled_patterns = self._compile_prioritized_patterns()
            self._cache.clear()
    
    def get_keywords(self) -> List[str]:
        return self.keywords.copy()
    
    def clear_cache(self) -> None:
        self._cache.clear()

