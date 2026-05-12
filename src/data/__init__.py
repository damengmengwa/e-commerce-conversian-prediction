"""Data loading and validation module.

Functions:
- load_events: Load raw e-commerce events from CSV and clean
- validate_features: Analyze feature quality, correlation, and multicollinearity
"""
from .loader import load_events
from .validator import validate_features

__all__ = ["load_events", "validate_features"]
