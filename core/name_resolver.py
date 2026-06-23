# =============================================================================
# LLM4MOF Autonomous System - Name Resolver (Shared Utility)
# =============================================================================
# Function & Purpose:
# This module translates raw Building Block IDs (e.g., "N164", "E70") from the 
# database into human-readable chemical names (e.g., "Zr6 Cluster", "BDC") for 
# Agent 1's feedback tables. Without this, the AI would receive meaningless
# alphanumeric codes and fail to learn which chemistries work or fail.
# 
# Why it is a separate file (Singleton Pattern):
# The `pormake_bb_dictionary_v3.2.json` is a large file. By using a Singleton 
# pattern (`get_name_resolver()`), this script loads the JSON into memory exactly 
# once at startup. If the Matchmaker, SensitivityAnalyzer, and FeedbackGenerator 
# all loaded the JSON individually from the hard drive, the experiment would slow 
# down dramatically. This guarantees it is initialized only once per run.
# =============================================================================

import os
import sys
import json

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BB_DICTIONARY_V3_PATH


class NameResolver:
    """
    Singleton-style resolver that maps Building Block IDs to readable names.
    
    All modules that need to translate IDs (e.g., "N164" -> "Zr6 Cluster")
    should use this shared instance instead of loading the dictionary themselves.
    
    Usage:
        from core.name_resolver import get_name_resolver
        resolver = get_name_resolver()
        name = resolver.resolve("N164")  # -> "Zr6 Cluster"
    """
    
    _instance = None
    
    def __init__(self):
        """
        Load the BB dictionary and build:
          1. ID -> readable_name map (self._name_map)
          2. Full Data List (self.bb_data)
          3. ID -> Data Dict Lookup (self.bb_lookup)
        """
        self._name_map = {}
        self.bb_data = []
        self.bb_lookup = {}
        
        try:
            if os.path.exists(BB_DICTIONARY_V3_PATH):
                with open(BB_DICTIONARY_V3_PATH, 'r', encoding='utf-8') as f:
                    self.bb_data = json.load(f)
                    
                    for item in self.bb_data:
                        # 1. Populate Name Map
                        if 'ID' in item:
                            self._name_map[item['ID']] = item.get('readable_name', item['ID'])
                            # 2. Populate Lookup
                            self.bb_lookup[item['ID']] = item
                            
                print(f"[NameResolver] Loaded {len(self._name_map)} items from BB Dictionary.")
            else:
                print(f"[NameResolver] WARNING: BB Dictionary not found at {BB_DICTIONARY_V3_PATH}. Using raw IDs.")
        except Exception as e:
            print(f"[NameResolver] ERROR loading name map: {e}")
    
    def resolve(self, item_id: str, fallback: str = None) -> str:
        """
        Resolve a building block ID to its human-readable name.
        
        Args:
            item_id: The BB ID (e.g., "N164", "E70")
            fallback: Optional fallback value. Defaults to item_id itself.
            
        Returns:
            Human-readable name or the fallback/ID if not found.
        """
        return self._name_map.get(item_id, fallback if fallback is not None else item_id)
    
    def translate_mof_filename(self, filename_str: str) -> str:
        """
        Translates a MOF filename code (e.g., "ukd+N164+E70") to 
        human-readable component names.
        
        For QMOF IDs (e.g., "qmof-0070c13"), returns the ID as-is.
        
        Returns:
            String like "Node: Zr6 Cluster | Linker: BDC" or QMOF ID
        """
        try:
            # QMOF IDs don't use '+' separator
            if '+' not in filename_str:
                return filename_str
            parts = filename_str.split('+')
            if len(parts) < 3:
                return filename_str
            node_name = self.resolve(parts[1])
            edge_name = self.resolve(parts[2])
            return f"Node: {node_name} | Linker: {edge_name}".replace("X-terminated", "").strip()
        except (IndexError, KeyError, TypeError):
            return filename_str
    
    @property
    def name_map(self) -> dict:
        """Direct access to the full ID -> name dictionary (read-only copy)."""
        return dict(self._name_map)



# Module-level singleton accessor
_resolver_instance = None

def get_name_resolver() -> NameResolver:
    """
    Get the shared NameResolver instance (lazy singleton).
    First call loads the dictionary; subsequent calls reuse the same instance.
    """
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = NameResolver()
    return _resolver_instance
