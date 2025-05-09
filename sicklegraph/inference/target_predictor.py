"""
Target predictor for SickleGraph.

This module implements a target predictor for gene therapy targets.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np # type: ignore

from sicklegraph.graph import GraphDatabase, NodeLabel, RelationshipType # type: ignore
from sicklegraph.inference.base import InferenceEngine # type: ignore

logger = logging.getLogger(__name__)


class TargetPredictor:
    """
    Target predictor for SickleGraph.
    
    This class implements a target predictor for gene therapy targets.
    """
    
    def __init__(self, graph: GraphDatabase):
        """
        Initialize the target predictor.
        
        Args:
            graph (GraphDatabase): The graph database to query.
        """
        self.graph = graph
        self.engine = InferenceEngine(graph)
    
    def predict_targets(
        self, 
        mutation: str = "HbSS",
        population: Optional[str] = None,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Predict gene therapy targets for a mutation and population.
        
        Args:
            mutation (str): The mutation type (e.g., "HbSS").
            population (Optional[str]): The population (e.g., "nigerian").
            top_n (int): The number of top targets to return.
            
        Returns:
            List[Dict[str, Any]]: The predicted targets.
        """
        return self.engine.predict_gene_targets(mutation, population, top_n)
    
    def predict_off_targets(
        self, 
        gene: str,
        population: Optional[str] = None,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Predict off-target effects for a gene and population.
        
        Args:
            gene (str): The gene symbol (e.g., "HBB").
            population (Optional[str]): The population (e.g., "nigerian").
            top_n (int): The number of top off-targets to return.
            
        Returns:
            List[Dict[str, Any]]: The predicted off-targets.
        """
        return self.engine.predict_off_targets(gene, population, top_n)
    
    def get_target_details(self, gene_symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a gene target.
        
        Args:
            gene_symbol (str): The gene symbol (e.g., "HBB").
            
        Returns:
            Dict[str, Any]: The gene details.
        """
        try:
            # Get the gene
            query = f"""
            MATCH (g:{NodeLabel.GENE})
            WHERE g.symbol = '{gene_symbol}'
            RETURN g
            """
            results = self.graph.execute_query(query)
            
            if not results:
                logger.warning(f"Gene not found: {gene_symbol}")
                return {}
            
            gene = results[0]["g"]
            
            # Get treatments targeting this gene
            treatment_query = f"""
            MATCH (t:{NodeLabel.TREATMENT})-[r:{RelationshipType.TARGETS}]->(g:{NodeLabel.GENE})
            WHERE g.symbol = '{gene_symbol}'
            RETURN t, r
            """
            treatment_results = self.graph.execute_query(treatment_query)
            
            treatments = []
            for result in treatment_results:
                treatment = result.get("t", {})
                relationship = result.get("r", {})
                
                treatments.append({
                    "id": treatment.get("id"),
                    "name": treatment.get("name"),
                    "type": treatment.get("type"),
                    "mechanism": relationship.get("mechanism"),
                    "efficacy": relationship.get("efficacy")
                })
            
            # Get papers mentioning this gene
            paper_query = f"""
            MATCH (p:{NodeLabel.PAPER})-[:{RelationshipType.TARGETS}]->(g:{NodeLabel.GENE})
            WHERE g.symbol = '{gene_symbol}'
            RETURN p
            ORDER BY p.publication_date DESC
            LIMIT 5
            """
            paper_results = self.graph.execute_query(paper_query)
            
            papers = []
            for result in paper_results:
                paper = result.get("p", {})
                
                papers.append({
                    "id": paper.get("id"),
                    "title": paper.get("title"),
                    "publication_date": paper.get("publication_date"),
                    "journal": paper.get("journal"),
                    "url": paper.get("url")
                })
            
            # Format the result
            return {
                "gene_id": gene.get("id"),
                "gene_symbol": gene.get("symbol"),
                "gene_name": gene.get("name"),
                "chromosome": gene.get("chromosome"),
                "start_position": gene.get("start_position"),
                "end_position": gene.get("end_position"),
                "description": gene.get("description"),
                "treatments": treatments,
                "papers": papers
            }
        
        except Exception as e:
            logger.error(f"Failed to get target details: {e}")
            return {}
    
    def compare_targets(self, gene_symbols: List[str]) -> Dict[str, Any]:
        """
        Compare multiple gene targets.
        
        Args:
            gene_symbols (List[str]): The gene symbols to compare.
            
        Returns:
            Dict[str, Any]: The comparison results.
        """
        try:
            if not gene_symbols:
                logger.warning("No gene symbols provided")
                return {}
            
            # Get details for each gene
            gene_details = []
            
            for gene_symbol in gene_symbols:
                details = self.get_target_details(gene_symbol)
                if details:
                    gene_details.append(details)
            
            if not gene_details:
                logger.warning("No gene details found")
                return {}
            
            # Compare the genes
            comparison = {
                "genes": gene_details,
                "comparison": {
                    "treatment_count": {},
                    "paper_count": {},
                    "efficacy": {}
                }
            }
            
            for details in gene_details:
                gene_symbol = details.get("gene_symbol")
                
                if not gene_symbol:
                    continue
                
                # Treatment count
                treatment_count = len(details.get("treatments", []))
                comparison["comparison"]["treatment_count"][gene_symbol] = treatment_count
                
                # Paper count
                paper_count = len(details.get("papers", []))
                comparison["comparison"]["paper_count"][gene_symbol] = paper_count
                
                # Average efficacy
                efficacies = [t.get("efficacy", 0.0) for t in details.get("treatments", [])]
                avg_efficacy = np.mean(efficacies) if efficacies else 0.0
                comparison["comparison"]["efficacy"][gene_symbol] = avg_efficacy
            
            return comparison
        
        except Exception as e:
            logger.error(f"Failed to compare targets: {e}")
            return {}