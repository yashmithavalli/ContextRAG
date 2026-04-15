import hashlib
import json

class QueryCache:
    """An in-memory dictionary-based cache for storing query responses."""
    def __init__(self):
        self._cache = {}

    def _generate_key(self, query: str, history: list[dict]) -> str:
        """Cache key must now hash (question + last 2 turns)"""
        # Feature 1 requirement: hash question + last 2 turns
        # 2 turns = 4 messages max
        recent_history = history[-4:] if history else []
        
        # Remove volatile fields like sources that might shift or be too huge
        # We only care about role and content for semantic uniqueness of the turn
        cleaned_history = [{"role": msg.get("role"), "content": msg.get("content")} for msg in recent_history]
        
        history_str = json.dumps(cleaned_history, sort_keys=True)
        key_input = f"{query}_{history_str}"
        return hashlib.md5(key_input.encode('utf-8')).hexdigest()

    def get(self, query: str, history: list[dict] = None):
        """Retrieve a cached response for the query + history if it exists, else return None."""
        history = history or []
        key = self._generate_key(query, history)
        return self._cache.get(key)

    def set(self, query: str, history: list[dict], response: dict):
        """Cache the response dict for the given query + history."""
        history = history or []
        key = self._generate_key(query, history)
        self._cache[key] = response

    def clear(self):
        """Clear the entire cache."""
        self._cache.clear()

# Global singleton instance to be imported by main routines
query_cache = QueryCache()
