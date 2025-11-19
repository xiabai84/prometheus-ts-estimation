from opensearchpy import OpenSearch
from datetime import datetime
from typing import List, Dict, Any, Optional, Union


class OpenSearchLogger:
    def __init__(self, host: str = 'localhost', port: int = 9200, 
                 username: str = None, password: str = None, 
                 use_ssl: bool = False, verify_certs: bool = False):
        """
        Initialize OpenSearch client
        
        Args:
            host: OpenSearch host address
            port: Port number
            username: Username
            password: Password
            use_ssl: Whether to use SSL
            verify_certs: Whether to verify certificates
        """
        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_compress=True,
            http_auth=(username, password) if username and password else None,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_show_warn=False
        )
        
    def search_match_phrase(self, index: str, field: str, phrase: str, 
                          size: int = 100, sort_asc: bool = True,
                          timestamp_from: str = None, timestamp_to: str = None,
                          source_fields: List[str] = None) -> Dict[str, Any]:
        """
        Search using match_phrase for exact phrase matching
        
        Args:
            index: Index name
            field: Field name
            phrase: Search phrase
            size: Number of results to return
            sort_asc: Whether to sort by timestamp in ascending order
            timestamp_from: Start time (format: '2024-01-01T00:00:00' or 'now-1h')
            timestamp_to: End time
            source_fields: List of fields to return
            
        Returns:
            Search results
        """
        query = {
            "bool": {
                "must": [
                    {
                        "match_phrase": {
                            field: phrase
                        }
                    }
                ]
            }
        }
        
        # Add time range filter
        if timestamp_from or timestamp_to:
            time_range = {}
            if timestamp_from:
                time_range["gte"] = timestamp_from
            if timestamp_to:
                time_range["lte"] = timestamp_to
                
            if "filter" not in query["bool"]:
                query["bool"]["filter"] = []
            query["bool"]["filter"].append({
                "range": {
                    "@timestamp": time_range
                }
            })
        
        search_body = {
            "query": query,
            "size": size,
            "sort": [
                {
                    "@timestamp": {
                        "order": "asc" if sort_asc else "desc"
                    }
                }
            ]
        }
        
        if source_fields:
            search_body["_source"] = source_fields
            
        return self.client.search(
            index=index,
            body=search_body
        )
    
    def search_match(self, index: str, field: str, query_text: str,
                    operator: str = "or", size: int = 100,
                    timestamp_from: str = None, timestamp_to: str = None,
                    fuzziness: str = None, min_should_match: str = None) -> Dict[str, Any]:
        """
        Search using match for full-text search
        
        Args:
            index: Index name
            field: Field name
            query_text: Search text
            operator: Operator ('and' or 'or')
            size: Number of results to return
            timestamp_from: Start time
            timestamp_to: End time
            fuzziness: Fuzziness level
            min_should_match: Minimum should match level
            
        Returns:
            Search results
        """
        match_query = {
            "query": query_text
        }
        
        if operator:
            match_query["operator"] = operator
        if fuzziness:
            match_query["fuzziness"] = fuzziness
        if min_should_match:
            match_query["minimum_should_match"] = min_should_match
        
        query = {
            "bool": {
                "must": [
                    {
                        "match": {
                            field: match_query
                        }
                    }
                ]
            }
        }
        
        # Add time range filter
        if timestamp_from or timestamp_to:
            time_range = {}
            if timestamp_from:
                time_range["gte"] = timestamp_from
            if timestamp_to:
                time_range["lte"] = timestamp_to
                
            if "filter" not in query["bool"]:
                query["bool"]["filter"] = []
            query["bool"]["filter"].append({
                "range": {
                    "@timestamp": time_range
                }
            })
        
        search_body = {
            "query": query,
            "size": size,
            "sort": [
                {
                    "@timestamp": {
                        "order": "desc"
                    }
                }
            ]
        }
            
        return self.client.search(
            index=index,
            body=search_body
        )
    
    def search_multi_match(self, index: str, fields: List[str], query_text: str,
                          size: int = 100, timestamp_from: str = None, 
                          timestamp_to: str = None) -> Dict[str, Any]:
        """
        Search across multiple fields
        
        Args:
            index: Index name
            fields: List of fields
            query_text: Search text
            size: Number of results to return
            timestamp_from: Start time
            timestamp_to: End time
            
        Returns:
            Search results
        """
        query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": fields
                        }
                    }
                ]
            }
        }
        
        # Add time range filter
        if timestamp_from or timestamp_to:
            time_range = {}
            if timestamp_from:
                time_range["gte"] = timestamp_from
            if timestamp_to:
                time_range["lte"] = timestamp_to
                
            if "filter" not in query["bool"]:
                query["bool"]["filter"] = []
            query["bool"]["filter"].append({
                "range": {
                    "@timestamp": time_range
                }
            })
        
        search_body = {
            "query": query,
            "size": size,
            "sort": [
                {
                    "@timestamp": {
                        "order": "desc"
                    }
                }
            ]
        }
            
        return self.client.search(
            index=index,
            body=search_body
        )
    
    def search_recent_logs(self, index: str, size: int = 100, 
                          time_range: str = "now-1h",
                          sort_asc: bool = False) -> Dict[str, Any]:
        """
        Search for recent logs
        
        Args:
            index: Index name
            size: Number of results to return
            time_range: Time range (e.g., 'now-1h', 'now-24h')
            sort_asc: Whether to sort by timestamp in ascending order
            
        Returns:
            Search results
        """
        search_body = {
            "query": {
                "range": {
                    "@timestamp": {
                        "gte": time_range
                    }
                }
            },
            "size": size,
            "sort": [
                {
                    "@timestamp": {
                        "order": "asc" if sort_asc else "desc"
                    }
                }
            ]
        }
            
        return self.client.search(
            index=index,
            body=search_body
        )
    
    def search_with_filters(self, index: str, 
                           must_conditions: List[Dict] = None,
                           should_conditions: List[Dict] = None,
                           filter_conditions: List[Dict] = None,
                           size: int = 100,
                           timestamp_from: str = None,
                           timestamp_to: str = None,
                           source_fields: List[str] = None) -> Dict[str, Any]:
        """
        Search with complex conditions
        
        Args:
            index: Index name
            must_conditions: Conditions that must be satisfied
            should_conditions: Conditions that should be satisfied
            filter_conditions: Filter conditions
            size: Number of results to return
            timestamp_from: Start time
            timestamp_to: End time
            source_fields: List of fields to return
            
        Returns:
            Search results
        """
        query = {
            "bool": {}
        }
        
        if must_conditions:
            query["bool"]["must"] = must_conditions
            
        if should_conditions:
            query["bool"]["should"] = should_conditions
            query["bool"]["minimum_should_match"] = 1
            
        if filter_conditions:
            query["bool"]["filter"] = filter_conditions
        
        # Add time range filter
        if timestamp_from or timestamp_to:
            time_range = {}
            if timestamp_from:
                time_range["gte"] = timestamp_from
            if timestamp_to:
                time_range["lte"] = timestamp_to
                
            if "filter" not in query["bool"]:
                query["bool"]["filter"] = []
            query["bool"]["filter"].append({
                "range": {
                    "@timestamp": time_range
                }
            })
        
        search_body = {
            "query": query,
            "size": size,
            "sort": [
                {
                    "@timestamp": {
                        "order": "desc"
                    }
                }
            ]
        }
        
        if source_fields:
            search_body["_source"] = source_fields
            
        return self.client.search(
            index=index,
            body=search_body
        )
    
    def scroll_search(self, index: str, query: Dict, size: int = 1000, 
                     scroll_time: str = "2m") -> List[Dict]:
        """
        Search large datasets using scroll API
        
        Args:
            index: Index name
            query: Query conditions
            size: Batch size
            scroll_time: Scroll keep-alive time
            
        Returns:
            List of all matching documents
        """
        all_results = []
        
        # Initial search
        response = self.client.search(
            index=index,
            body=query,
            scroll=scroll_time,
            size=size
        )
        
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']
        all_results.extend(hits)
        
        # Continue fetching until no more results
        while len(hits) > 0:
            response = self.client.scroll(
                scroll_id=scroll_id,
                scroll=scroll_time
            )
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']
            all_results.extend(hits)
        
        # Clean up scroll
        self.client.clear_scroll(scroll_id=scroll_id)
        
        return all_results
    
    def get_indices(self) -> List[str]:
        """
        Get list of all indices
        
        Returns:
            List of index names
        """
        return list(self.client.indices.get_alias("*").keys())


# Usage example
if __name__ == "__main__":
    # Initialize client
    logger = OpenSearchLogger(
        host="localhost",
        port=9200,
        username="admin",  # If authentication is required
        password="admin"   # If authentication is required
    )
    
    # Example 1: Search using match_phrase for exact phrase
    result1 = logger.search_match_phrase(
        index="logs-*",
        field="message",
        phrase="connection timeout",
        size=50,
        timestamp_from="now-1h",
        timestamp_to="now",
        source_fields=["@timestamp", "message", "level", "host"]
    )
    
    # Example 2: Search using match
    result2 = logger.search_match(
        index="app-logs",
        field="error_message",
        query_text="database connection failed",
        operator="and",
        size=100,
        timestamp_from="2024-01-01T00:00:00",
        timestamp_to="2024-01-15T23:59:59"
    )
    
    # Example 3: Multi-field search
    result3 = logger.search_multi_match(
        index="logs-*",
        fields=["message", "description", "content"],
        query_text="critical error",
        size=200,
        timestamp_from="now-24h"
    )
    
    # Example 4: Recent logs
    result4 = logger.search_recent_logs(
        index="logs-*",
        size=100,
        time_range="now-30m",
        sort_asc=False
    )
    
    # Example 5: Complex condition search
    result5 = logger.search_with_filters(
        index="app-logs",
        must_conditions=[
            {
                "term": {
                    "level": "ERROR"
                }
            }
        ],
        filter_conditions=[
            {
                "exists": {
                    "field": "stack_trace"
                }
            }
        ],
        size=50,
        timestamp_from="now-6h",
        source_fields=["@timestamp", "message", "level", "application"]
    )
    
    # Print results example
    def print_results(results, title):
        print(f"\n=== {title} ===")
        print(f"Total hits: {results['hits']['total']['value']}")
        for hit in results['hits']['hits']:
            source = hit['_source']
            print(f"Time: {source.get('@timestamp', 'N/A')}")
            print(f"Message: {source.get('message', 'N/A')}")
            print("-" * 50)
    
    print_results(result1, "Match Phrase Search Results")
    print_results(result4, "Recent Logs Search Results")
    