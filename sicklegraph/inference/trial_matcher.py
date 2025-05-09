"""
Clinical trial matcher for SickleGraph.

This module implements a clinical trial matcher.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from sicklegraph.graph import GraphDatabase, NodeLabel, RelationshipType # type: ignore
from sicklegraph.inference.base import InferenceEngine # type: ignore

logger = logging.getLogger(__name__)


class TrialMatcher:
    """
    Clinical trial matcher for SickleGraph.
    
    This class implements a clinical trial matcher.
    """
    
    def __init__(self, graph: GraphDatabase):
        """
        Initialize the trial matcher.
        
        Args:
            graph (GraphDatabase): The graph database to query.
        """
        self.graph = graph
        self.engine = InferenceEngine(graph)
    
    def match_trials(
        self, 
        mutation: str = "HbSS",
        population: Optional[str] = None,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Match clinical trials for a mutation and population.
        
        Args:
            mutation (str): The mutation type (e.g., "HbSS").
            population (Optional[str]): The population (e.g., "nigerian").
            top_n (int): The number of top trials to return.
            
        Returns:
            List[Dict[str, Any]]: The matched trials.
        """
        return self.engine.match_clinical_trials(mutation, population, top_n)
    
    def get_trial_details(self, trial_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a clinical trial.
        
        Args:
            trial_id (str): The trial ID.
            
        Returns:
            Dict[str, Any]: The trial details.
        """
        try:
            # Get the trial
            query = f"""
            MATCH (t:{NodeLabel.CLINICAL_TRIAL})
            WHERE t.id = '{trial_id}'
            RETURN t
            """
            results = self.graph.execute_query(query)
            
            if not results:
                logger.warning(f"Trial not found: {trial_id}")
                return {}
            
            trial = results[0]["t"]
        except Exception as e:
            logger.error(f"An error occurred while fetching trial details: {e}")
            return {}
            
            # Get countries where the trial is conducted
            country_query = f"""
            MATCH (t:{NodeLabel.CLINICAL_TRIAL})-[r:{RelationshipType.CONDUCTED_IN}]->(c:{NodeLabel.COUNTRY})
            WHERE t.id = '{trial_id}'
            RETURN c, r
            """
            country_results = self.graph.execute_query(country_query)
            
            countries = []
            for result in country_results:
                country = result.get("c", {})
                relationship = result.get("r", {})
                
                countries.append({
                    "id": country.get("id"),
                    "name": country.get("name"),
                    "site_count": relationship.get("site_count"),
                    "enrollment": relationship.get("enrollment")
                })
            
            # Get institutions conducting the trial
            institution_query = f"""
            MATCH (t:{NodeLabel.CLINICAL_TRIAL})-[r:{RelationshipType.CONDUCTED_BY}]->(i:{NodeLabel.INSTITUTION})
            WHERE t.id = '{trial_id}'
            RETURN i, r
            """
            institution_results = self.graph.execute_query(institution_query)
            
            institutions = []
            for result in institution_results:
                institution = result.get("i", {})
                relationship = result.get("r", {})
                
                institutions.append({
                    "id": institution.get("id"),
                    "name": institution.get("name"),
                    "type": institution.get("type"),
                    "role": relationship.get("role"),
                    "principal_investigator": relationship.get("principal_investigator")
                })
            
            # Get treatments tested in the trial
            treatment_query = f"""
            MATCH (t:{NodeLabel.CLINICAL_TRIAL})-[r:{RelationshipType.TARGETS}]->(tr:{NodeLabel.TREATMENT})
            WHERE t.id = '{trial_id}'
            RETURN tr, r
            """
            treatment_results = self.graph.execute_query(treatment_query)
            
            treatments = []
            for result in treatment_results:
                treatment = result.get("tr", {})
                relationship = result.get("r", {})
                
                treatments.append({
                    "id": treatment.get("id"),
                    "name": treatment.get("name"),
                    "type": treatment.get("type"),
                    "mechanism": relationship.get("mechanism")
                })
            
            # Format the result
            return {
                "trial_id": trial.get("id"),
                "title": trial.get("title"),
                "description": trial.get("description"),
                "status": trial.get("status"),
                "phase": trial.get("phase"),
                "start_date": trial.get("start_date"),
                "end_date": trial.get("end_date"),
                "enrollment": trial.get("enrollment"),
                "countries": countries,
                "institutions": institutions,
                "treatments": treatments
            }