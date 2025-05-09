"""
SickleGraph: An AI-powered knowledge graph for gene therapy innovation in Africa.

This package provides tools for integrating diverse biomedical data sources,
querying the knowledge graph using natural language, and making predictions
to accelerate research and discovery.
"""

import logging

from sicklegraph.graph import GraphDatabaseFactory, create_initial_schema # type: ignore
from sicklegraph.eliza import ElizaAssistant # type: ignore
from sicklegraph.inference import TargetPredictor, TrialMatcher # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

__version__ = "0.1.0"
__author__ = "SickleGraph Team"
__email__ = "info@sicklegraph.org"
__license__ = "MIT"
__description__ = "An AI-powered knowledge graph for gene therapy innovation in Africa."

__all__ = [
    "GraphDatabaseFactory",
    "create_initial_schema",
    "ElizaAssistant",
    "TargetPredictor",
    "TrialMatcher"
]