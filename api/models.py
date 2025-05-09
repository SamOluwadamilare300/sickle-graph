"""
API models for SickleGraph.

This module defines the API models for SickleGraph.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field # type: ignore


class GeneModel(BaseModel):
    """Gene model."""
    id: str
    symbol: str
    name: str
    chromosome: Optional[str] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    strand: Optional[str] = None
    sequence: Optional[str] = None
    description: Optional[str] = None


class MutationModel(BaseModel):
    """Mutation model."""
    id: str
    name: str
    type: str
    reference_allele: Optional[str] = None
    alternative_allele: Optional[str] = None
    position: Optional[int] = None
    consequence: Optional[str] = None
    population_frequency: Optional[float] = None
    african_frequency: Optional[float] = None
    nigerian_frequency: Optional[float] = None
    description: Optional[str] = None


class TreatmentModel(BaseModel):
    """Treatment model."""
    id: str
    name: str
    type: str
    mechanism: Optional[str] = None
    development_stage: Optional[str] = None
    approval_status: Optional[str] = None
    description: Optional[str] = None


class PaperModel(BaseModel):
    """Paper model."""
    id: str
    title: str
    abstract: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    citation_count: Optional[int] = None
    url: Optional[str] = None


class ClinicalTrialModel(BaseModel):
    """Clinical trial model."""
    id: str
    title: str
    description: Optional[str] = None
    status: str
    phase: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    enrollment: Optional[int] = None
    url: Optional[str] = None


class ResearcherModel(BaseModel):
    """Researcher model."""
    id: str
    name: str
    email: Optional[str] = None
    orcid: Optional[str] = None
    h_index: Optional[int] = None
    publication_count: Optional[int] = None


class InstitutionModel(BaseModel):
    """Institution model."""
    id: str
    name: str
    type: Optional[str] = None
    url: Optional[str] = None


class CountryModel(BaseModel):
    """Country model."""
    id: str
    name: str
    iso_code: Optional[str] = None
    population: Optional[int] = None
    scd_prevalence: Optional[float] = None


class ElizaQueryRequest(BaseModel):
    """ELIZA query request model."""
    query: str
    language: Optional[str] = "en"


class ElizaQueryResponse(BaseModel):
    """ELIZA query response model."""
    response: str
    language: str


class TargetPredictionRequest(BaseModel):
    """Target prediction request model."""
    mutation: str = "HbSS"
    population: Optional[str] = None
    top_n: int = 5


class TargetPredictionResponse(BaseModel):
    """Target prediction response model."""
    targets: List[Dict[str, Any]]


class OffTargetPredictionRequest(BaseModel):
    """Off-target prediction request model."""
    gene: str
    population: Optional[str] = None
    top_n: int = 5


class OffTargetPredictionResponse(BaseModel):
    """Off-target prediction response model."""
    off_targets: List[Dict[str, Any]]


class ClinicalTrialMatchRequest(BaseModel):
    """Clinical trial match request model."""
    mutation: str = "HbSS"
    population: Optional[str] = None
    top_n: int = 5


class ClinicalTrialMatchResponse(BaseModel):
    """Clinical trial match response model."""
    trials: List[Dict[str, Any]]


class TreatmentOutcomeRequest(BaseModel):
    """Treatment outcome prediction request model."""
    treatment_id: str
    mutation: str = "HbSS"
    population: Optional[str] = None


class TreatmentOutcomeResponse(BaseModel):
    """Treatment outcome prediction response model."""
    outcome: Dict[str, Any]