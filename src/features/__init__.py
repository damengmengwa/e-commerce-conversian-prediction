"""Feature engineering module.

Functions:
- engineer_features: Generate 29 behavioral and temporal features from raw events
"""
from .engineer import engineer_features

__all__ = ["engineer_features"]
