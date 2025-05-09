"""
Data pipeline module for SickleGraph.

This module provides the data pipeline functionality for SickleGraph.
"""

from sicklegraph.pipeline.base import DataSource, SickleGraphPipeline # type: ignore
from sicklegraph.pipeline.pubmed import PubMedSource # type: ignore
from sicklegraph.pipeline.clinical_trials import ClinicalTrialsSource # type: ignore

__all__ = [
    "DataSource",
    "SickleGraphPipeline",
    "PubMedSource",
    "ClinicalTrialsSource"
]