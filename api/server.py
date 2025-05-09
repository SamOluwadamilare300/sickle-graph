"""
API server for SickleGraph.

This module implements the API server for SickleGraph.
"""

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore

from sicklegraph.api.models import ( # type: ignore
    GeneModel,
    MutationModel,
    TreatmentModel,
    PaperModel,
    ClinicalTrialModel,
    ElizaQueryRequest,
    ElizaQueryResponse,
    TargetPredictionRequest,
    TargetPredictionResponse,
    OffTargetPredictionRequest,
    OffTargetPredictionResponse,
    ClinicalTrialMatchRequest,
    ClinicalTrialMatchResponse,
    TreatmentOutcomeRequest,
    TreatmentOutcomeResponse
)
from sicklegraph.graph import GraphDatabaseFactory, NodeLabel # type: ignore
from sicklegraph.eliza import ElizaAssistant # type: ignore
from sicklegraph.inference import TargetPredictor, TrialMatcher # type: ignore

logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(
    title="SickleGraph API",
    description="API for SickleGraph, an AI-powered knowledge graph for gene therapy innovation in Africa.",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the graph database
graph = GraphDatabaseFactory.create_from_env()
if not graph:
    raise RuntimeError("Failed to create graph database")

# Connect to the graph database
if not graph.connect():
    raise RuntimeError("Failed to connect to graph database")

# Create the ELIZA assistant
eliza = ElizaAssistant(graph)

# Create the inference engines
target_predictor = TargetPredictor(graph)
trial_matcher = TrialMatcher(graph)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the SickleGraph API"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/genes", response_model=List[GeneModel])
async def get_genes(limit: int = 10, offset: int = 0):
    """
    Get genes.
    
    Args:
        limit (int): The maximum number of genes to return.
        offset (int): The offset to start from.
        
    Returns:
        List[GeneModel]: The genes.
    """
    try:
        query = f"""
        MATCH (g:{NodeLabel.GENE})
        RETURN g
        ORDER BY g.symbol
        SKIP {offset}
        LIMIT {limit}
        """
        
        results = graph.execute_query(query)
        
        genes = []
        for result in results:
            gene = result.get("g", {})
            genes.append(GeneModel(**gene))
        
        return genes
    
    except Exception as e:
        logger.error(f"Failed to get genes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/genes/{gene_id}", response_model=GeneModel)
async def get_gene(gene_id: str):
    """
    Get a gene by ID.
    
    Args:
        gene_id (str): The gene ID.
        
    Returns:
        GeneModel: The gene.
    """
    try:
        gene = graph.get_node(gene_id, NodeLabel.GENE)
        
        if not gene:
            raise HTTPException(status_code=404, detail=f"Gene not found: {gene_id}")
        
        return GeneModel(**gene)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to get gene: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/treatments", response_model=List[TreatmentModel])
async def get_treatments(limit: int = 10, offset: int = 0):
    """
    Get treatments.
    
    Args:
        limit (int): The maximum number of treatments to return.
        offset (int): The offset to start from.
        
    Returns:
        List[TreatmentModel]: The treatments.
    """
    try:
        query = f"""
        MATCH (t:{NodeLabel.TREATMENT})
        RETURN t
        ORDER BY t.name
        SKIP {offset}
        LIMIT {limit}
        """
        
        results = graph.execute_query(query)
        
        treatments = []
        for result in results:
            treatment = result.get("t", {})
            treatments.append(TreatmentModel(**treatment))
        
        return treatments
    
    except Exception as e:
        logger.error(f"Failed to get treatments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/treatments/{treatment_id}", response_model=TreatmentModel)
async def get_treatment(treatment_id: str):
    """
    Get a treatment by ID.
    
    Args:
        treatment_id (str): The treatment ID.
        
    Returns:
        TreatmentModel: The treatment.
    """
    try:
        treatment = graph.get_node(treatment_id, NodeLabel.TREATMENT)
        
        if not treatment:
            raise HTTPException(status_code=404, detail=f"Treatment not found: {treatment_id}")
        
        return TreatmentModel(**treatment)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to get treatment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clinical-trials", response_model=List[ClinicalTrialModel])
async def get_clinical_trials(limit: int = 10, offset: int = 0):
    """
    Get clinical trials.
    
    Args:
        limit (int): The maximum number of clinical trials to return.
        offset (int): The offset to start from.
        
    Returns:
        List[ClinicalTrialModel]: The clinical trials.
    """
    try:
        query = f"""
        MATCH (t:{NodeLabel.CLINICAL_TRIAL})
        RETURN t
        ORDER BY t.start_date DESC
        SKIP {offset}
        LIMIT {limit}
        """
        
        results = graph.execute_query(query)
        
        trials = []
        for result in results:
            trial = result.get("t", {})
            trials.append(ClinicalTrialModel(**trial))
        
        return trials
    
    except Exception as e:
        logger.error(f"Failed to get clinical trials: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/clinical-trials/{trial_id}", response_model=ClinicalTrialModel)
async def get_clinical_trial(trial_id: str):
    """
    Get a clinical trial by ID.
    
    Args:
        trial_id (str): The clinical trial ID.
        
    Returns:
        ClinicalTrialModel: The clinical trial.
    """
    try:
        trial = graph.get_node(trial_id, NodeLabel.CLINICAL_TRIAL)
        
        if not trial:
            raise HTTPException(status_code=404, detail=f"Clinical trial not found: {trial_id}")
        
        return ClinicalTrialModel(**trial)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to get clinical trial: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers", response_model=List[PaperModel])
async def get_papers(limit: int = 10, offset: int = 0):
    """
    Get papers.
    
    Args:
        limit (int): The maximum number of papers to return.
        offset (int): The offset to start from.
        
    Returns:
        List[PaperModel]: The papers.
    """
    try:
        query = f"""
        MATCH (p:{NodeLabel.PAPER})
        RETURN p
        ORDER BY p.publication_date DESC
        SKIP {offset}
        LIMIT {limit}
        """
        
        results = graph.execute_query(query)
        
        papers = []
        for result in results:
            paper = result.get("p", {})
            papers.append(PaperModel(**paper))
        
        return papers
    
    except Exception as e:
        logger.error(f"Failed to get papers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/papers/{paper_id}", response_model=PaperModel)
async def get_paper(paper_id: str):
    """
    Get a paper by ID.
    
    Args:
        paper_id (str): The paper ID.
        
    Returns:
        PaperModel: The paper.
    """
    try:
        paper = graph.get_node(paper_id, NodeLabel.PAPER)
        
        if not paper:
            raise HTTPException(status_code=404, detail=f"Paper not found: {paper_id}")
        
        return PaperModel(**paper)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to get paper: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eliza/query", response_model=ElizaQueryResponse)
async def eliza_query(request: ElizaQueryRequest):
    """
    Query the ELIZA AI Research Assistant.
    
    Args:
        request (ElizaQueryRequest): The query request.
        
    Returns:
        ElizaQueryResponse: The query response.
    """
    try:
        response = eliza.query(request.query, request.language)
        
        return ElizaQueryResponse(
            response=response,
            language=eliza.language
        )
    
    except Exception as e:
        logger.error(f"Failed to query ELIZA: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/predict-targets", response_model=TargetPredictionResponse)
async def predict_targets(request: TargetPredictionRequest):
    """
    Predict gene therapy targets.
    
    Args:
        request (TargetPredictionRequest): The prediction request.
        
    Returns:
        TargetPredictionResponse: The prediction response.
    """
    try:
        targets = target_predictor.predict_targets(
            request.mutation,
            request.population,
            request.top_n
        )
        
        return TargetPredictionResponse(targets=targets)
    
    except Exception as e:
        logger.error(f"Failed to predict targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/predict-off-targets", response_model=OffTargetPredictionResponse)
async def predict_off_targets(request: OffTargetPredictionRequest):
    """
    Predict off-target effects.
    
    Args:
        request (OffTargetPredictionRequest): The prediction request.
        
    Returns:
        OffTargetPredictionResponse: The prediction response.
    """
    try:
        off_targets = target_predictor.predict_off_targets(
            request.gene,
            request.population,
            request.top_n
        )
        
        return OffTargetPredictionResponse(off_targets=off_targets)
    
    except Exception as e:
        logger.error(f"Failed to predict off-targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/match-trials", response_model=ClinicalTrialMatchResponse)
async def match_trials(request: ClinicalTrialMatchRequest):
    """
    Match clinical trials.
    
    Args:
        request (ClinicalTrialMatchRequest): The match request.
        
    Returns:
        ClinicalTrialMatchResponse: The match response.
    """
    try:
        trials = trial_matcher.match_trials(
            request.mutation,
            request.population,
            request.top_n
        )
        
        return ClinicalTrialMatchResponse(trials=trials)
    
    except Exception as e:
        logger.error(f"Failed to match trials: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/predict-treatment-outcome", response_model=TreatmentOutcomeResponse)
async def predict_treatment_outcome(request: TreatmentOutcomeRequest):
    """
    Predict treatment outcome.
    
    Args:
        request (TreatmentOutcomeRequest): The prediction request.
        
    Returns:
        TreatmentOutcomeResponse: The prediction response.
    """
    try:
        outcome = target_predictor.engine.predict_treatment_outcomes(
            request.treatment_id,
            request.mutation,
            request.population
        )
        
        return TreatmentOutcomeResponse(outcome=outcome)
    
    except Exception as e:
        logger.error(f"Failed to predict treatment outcome: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn # type: ignore
    
    # Get the port from the environment or use the default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run("sicklegraph.api.server:app", host="0.0.0.0", port=port, reload=True)