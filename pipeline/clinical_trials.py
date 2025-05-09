"""
ClinicalTrials.gov data source for SickleGraph.

This module implements the ClinicalTrials.gov data source.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import xml.etree.ElementTree as ET

try:
    import requests # type: ignore
except ImportError:
    requests = None

from sicklegraph.graph import GraphDatabase, NodeLabel, RelationshipType # type: ignore
from sicklegraph.pipeline.base import DataSource # type: ignore

logger = logging.getLogger(__name__)


class ClinicalTrialsSource(DataSource):
    """
    ClinicalTrials.gov data source for SickleGraph.
    
    This class implements the ClinicalTrials.gov data source.
    """
    
    def __init__(self):
        """
        Initialize the ClinicalTrials.gov data source.
        """
        self.base_url = "https://clinicaltrials.gov/api/query/full_studies"
    
    @property
    def name(self) -> str:
        """
        Get the name of the data source.
        
        Returns:
            str: The name of the data source.
        """
        return "clinical_trials"
    
    def extract(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Extract data from ClinicalTrials.gov.
        
        Args:
            query (str): The query to use for extraction.
            max_results (int): The maximum number of results to extract.
            
        Returns:
            List[Dict[str, Any]]: The extracted data.
        """
        try:
            if requests is None:
                logger.error("Requests is not installed. Please install it with 'pip install requests'.")
                return []
            
            # Search for clinical trials
            params = {
                "expr": query,
                "min_rnk": 1,
                "max_rnk": max_results,
                "fmt": "json"
            }
            
            logger.info(f"Searching ClinicalTrials.gov for: {query}")
            response = requests.get(self.base_url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to search ClinicalTrials.gov: {response.status_code} {response.text}")
                return []
            
            data = response.json()
            
            if "FullStudiesResponse" not in data:
                logger.error(f"Invalid response from ClinicalTrials.gov: {data}")
                return []
            
            studies = data["FullStudiesResponse"].get("FullStudies", [])
            
            if not studies:
                logger.warning("No clinical trials found")
                return []
            
            logger.info(f"Found {len(studies)} clinical trials")
            
            # Extract study data
            extracted_studies = []
            
            for study in studies:
                try:
                    study_data = study.get("Study", {})
                    protocol_section = study_data.get("ProtocolSection", {})
                    
                    # Extract basic information
                    identification_module = protocol_section.get("IdentificationModule", {})
                    nct_id = identification_module.get("NCTId")
                    
                    if not nct_id:
                        continue
                    
                    title = identification_module.get("BriefTitle")
                    
                    # Extract status information
                    status_module = protocol_section.get("StatusModule", {})
                    status = status_module.get("OverallStatus")
                    start_date = status_module.get("StartDate")
                    completion_date = status_module.get("CompletionDate")
                    
                    # Extract design information
                    design_module = protocol_section.get("DesignModule", {})
                    phase_list = design_module.get("PhaseList", {}).get("Phase", [])
                    
                    if isinstance(phase_list, str):
                        phase = phase_list
                    else:
                        phase = "/".join(phase_list) if phase_list else None
                    
                    # Extract description
                    description_module = protocol_section.get("DescriptionModule", {})
                    description = description_module.get("BriefSummary")
                    
                    # Extract eligibility information
                    eligibility_module = protocol_section.get("EligibilityModule", {})
                    eligibility_criteria = eligibility_module.get("EligibilityCriteria")
                    
                    # Extract location information
                    contacts_locations_module = protocol_section.get("ContactsLocationsModule", {})
                    location_list = contacts_locations_module.get("LocationList", {}).get("Location", [])
                    
                    if not isinstance(location_list, list):
                        location_list = [location_list]
                    
                    locations = []
                    
                    for location in location_list:
                        facility = location.get("Facility", {})
                        facility_name = facility.get("Name")
                        
                        address = location.get("Address", {})
                        city = address.get("City")
                        country = address.get("Country")
                        
                        if facility_name and country:
                            locations.append({
                                "facility": facility_name,
                                "city": city,
                                "country": country
                            })
                    
                    # Extract intervention information
                    arms_interventions_module = protocol_section.get("ArmsInterventionsModule", {})
                    intervention_list = arms_interventions_module.get("InterventionList", {}).get("Intervention", [])
                    
                    if not isinstance(intervention_list, list):
                        intervention_list = [intervention_list]
                    
                    interventions = []
                    
                    for intervention in intervention_list:
                        intervention_type = intervention.get("InterventionType")
                        intervention_name = intervention.get("InterventionName")
                        
                        if intervention_type and intervention_name:
                            interventions.append({
                                "type": intervention_type,
                                "name": intervention_name
                            })
                    
                    # Create a study record
                    study_record = {
                        "id": f"nct_{nct_id}",
                        "nct_id": nct_id,
                        "title": title,
                        "status": status,
                        "phase": phase,
                        "start_date": start_date,
                        "completion_date": completion_date,
                        "description": description,
                        "eligibility_criteria": eligibility_criteria,
                        "locations": locations,
                        "interventions": interventions,
                        "url": f"https://clinicaltrials.gov/study/{nct_id}"
                    }
                    
                    extracted_studies.append(study_record)
                
                except Exception as e:
                    logger.error(f"Failed to extract study data: {e}")
            
            logger.info(f"Extracted {len(extracted_studies)} clinical trials")
            return extracted_studies
        
        except Exception as e:
            logger.error(f"Failed to extract data from ClinicalTrials.gov: {e}")
            return []
    
    def transform(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform the extracted data.
        
        Args:
            data (List[Dict[str, Any]]): The extracted data.
            
        Returns:
            List[Dict[str, Any]]: The transformed data.
        """
        try:
            transformed_data = []
            
            for study in data:
                # Extract gene and treatment mentions from the title and description
                title = study.get("title", "")
                description = study.get("description", "")
                
                text = f"{title} {description}"
                
                # Extract gene mentions
                gene_mentions = self._extract_gene_mentions(text)
                
                # Extract treatment mentions
                treatment_mentions = self._extract_treatment_mentions(text)
                
                # Extract mutation mentions
                mutation_mentions = self._extract_mutation_mentions(text)
                
                # Extract additional treatment mentions from interventions
                interventions = study.get("interventions", [])
                
                for intervention in interventions:
                    intervention_name = intervention.get("name", "")
                    intervention_type = intervention.get("type", "")
                    
                    if intervention_name:
                        treatment_mentions.extend(self._extract_treatment_mentions(intervention_name))
                
                # Create a transformed study
                transformed_study = {
                    "study": study,
                    "gene_mentions": gene_mentions,
                    "treatment_mentions": treatment_mentions,
                    "mutation_mentions": mutation_mentions
                }
                
                transformed_data.append(transformed_study)
            
            logger.info(f"Transformed {len(transformed_data)} clinical trials")
            return transformed_data
        
        except Exception as e:
            logger.error(f"Failed to transform data: {e}")
            return []
    
    def load(self, data: List[Dict[str, Any]], graph: GraphDatabase) -> bool:
        """
        Load the transformed data into the graph database.
        
        Args:
            data (List[Dict[str, Any]]): The transformed data.
            graph (GraphDatabase): The graph database to load into.
            
        Returns:
            bool: True if the data was loaded successfully, False otherwise.
        """
        try:
            # Create a set to track created nodes
            created_genes = set()
            created_treatments = set()
            created_mutations = set()
            created_trials = set()
            created_countries = set()
            created_institutions = set()
            
            for item in data:
                study = item["study"]
                gene_mentions = item["gene_mentions"]
                treatment_mentions = item["treatment_mentions"]
                mutation_mentions = item["mutation_mentions"]
                
                # Create the clinical trial node
                trial_id = study["id"]
                
                if trial_id not in created_trials:
                    trial_properties = {
                        "id": trial_id,
                        "title": study["title"],
                        "description": study["description"],
                        "status": study["status"],
                        "phase": study["phase"],
                        "start_date": study["start_date"],
                        "end_date": study["completion_date"],
                        "url": study["url"]
                    }
                    
                    graph.create_node(NodeLabel.CLINICAL_TRIAL, trial_properties)
                    created_trials.add(trial_id)
                
                # Create country and institution nodes and relationships
                for location in study.get("locations", []):
                    country_name = location.get("country")
                    facility_name = location.get("facility")
                    
                    if country_name:
                        country_id = f"country_{country_name.lower().replace(' ', '_')}"
                        
                        if country_id not in created_countries:
                            country_properties = {
                                "id": country_id,
                                "name": country_name
                            }
                            
                            graph.create_node(NodeLabel.COUNTRY, country_properties)
                            created_countries.add(country_id)
                        
                        # Create the relationship between the trial and the country
                        graph.create_relationship(
                            trial_id, NodeLabel.CLINICAL_TRIAL,
                            country_id, NodeLabel.COUNTRY,
                            RelationshipType.CONDUCTED_IN,
                            {"confidence": 1.0}
                        )
                    
                    if facility_name:
                        institution_id = f"institution_{facility_name.lower().replace(' ', '_')}"
                        
                        if institution_id not in created_institutions:
                            institution_properties = {
                                "id": institution_id,
                                "name": facility_name,
                                "type": "Clinical"
                            }
                            
                            graph.create_node(NodeLabel.INSTITUTION, institution_properties)
                            created_institutions.add(institution_id)
                        
                        # Create the relationship between the trial and the institution
                        graph.create_relationship(
                            trial_id, NodeLabel.CLINICAL_TRIAL,
                            institution_id, NodeLabel.INSTITUTION,
                            RelationshipType.CONDUCTED_BY,
                            {"confidence": 1.0}
                        )
                        
                        # Create the relationship between the institution and the country
                        if country_name:
                            country_id = f"country_{country_name.lower().replace(' ', '_')}"
                            
                            graph.create_relationship(
                                institution_id, NodeLabel.INSTITUTION,
                                country_id, NodeLabel.COUNTRY,
                                RelationshipType.LOCATED_IN,
                                {"confidence": 1.0}
                            )
                
                # Create gene nodes and relationships
                for gene_mention in gene_mentions:
                    gene_id = f"gene_{gene_mention.lower()}"
                    
                    if gene_id not in created_genes:
                        gene_properties = {
                            "id": gene_id,
                            "symbol": gene_mention,
                            "name": self._get_gene_name(gene_mention)
                        }
                        
                        graph.create_node(NodeLabel.GENE, gene_properties)
                        created_genes.add(gene_id)
                    
                    # Create the relationship between the trial and the gene
                    graph.create_relationship(
                        trial_id, NodeLabel.CLINICAL_TRIAL,
                        gene_id, NodeLabel.GENE,
                        RelationshipType.TARGETS,
                        {"confidence": 0.8}
                    )
                
                # Create treatment nodes and relationships
                for treatment_mention in treatment_mentions:
                    treatment_id = f"treatment_{treatment_mention.lower().replace(' ', '_')}"
                    
                    if treatment_id not in created_treatments:
                        treatment_properties = {
                            "id": treatment_id,
                            "name": treatment_mention,
                            "type": self._get_treatment_type(treatment_mention)
                        }
                        
                        graph.create_node(NodeLabel.TREATMENT, treatment_properties)
                        created_treatments.add(treatment_id)
                    
                    # Create the relationship between the trial and the treatment
                    graph.create_relationship(
                        trial_id, NodeLabel.CLINICAL_TRIAL,
                        treatment_id, NodeLabel.TREATMENT,
                        RelationshipType.TARGETS,
                        {"confidence": 0.8}
                    )
                    
                    # Create relationships between treatments and genes
                    for gene_mention in gene_mentions:
                        gene_id = f"gene_{gene_mention.lower()}"
                        
                        graph.create_relationship(
                            treatment_id, NodeLabel.TREATMENT,
                            gene_id, NodeLabel.GENE,
                            RelationshipType.TARGETS,
                            {
                                "confidence": 0.7,
                                "mechanism": self._get_treatment_mechanism(treatment_mention)
                            }
                        )
                
                # Create mutation nodes and relationships
                for mutation_mention in mutation_mentions:
                    mutation_id = f"mutation_{mutation_mention.lower().replace(' ', '_')}"
                    
                    if mutation_id not in created_mutations:
                        mutation_properties = {
                            "id": mutation_id,
                            "name": mutation_mention,
                            "type": self._get_mutation_type(mutation_mention)
                        }
                        
                        graph.create_node(NodeLabel.MUTATION, mutation_properties)
                        created_mutations.add(mutation_id)
                    
                    # Create the relationship between the trial and the mutation
                    graph.create_relationship(
                        trial_id, NodeLabel.CLINICAL_TRIAL,
                        mutation_id, NodeLabel.MUTATION,
                        RelationshipType.TARGETS,
                        {"confidence": 0.8}
                    )
                    
                    # Create relationships between mutations and genes
                    for gene_mention in gene_mentions:
                        gene_id = f"gene_{gene_mention.lower()}"
                        
                        graph.create_relationship(
                            gene_id, NodeLabel.GENE,
                            mutation_id, NodeLabel.MUTATION,
                            RelationshipType.HAS_MUTATION,
                            {"confidence": 0.7}
                        )
            
            logger.info(f"Loaded {len(created_trials)} clinical trials, {len(created_genes)} genes, {len(created_treatments)} treatments, {len(created_mutations)} mutations, {len(created_countries)} countries, and {len(created_institutions)} institutions")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def _extract_gene_mentions(self, text: str) -> List[str]:
        """
        Extract gene mentions from text.
        
        Args:
            text (str): The text to extract from.
            
        Returns:
            List[str]: The extracted gene mentions.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a more sophisticated approach,
        # such as named entity recognition
        
        gene_keywords = [
            "HBB", "HBA1", "HBA2", "BCL11A", "HMOX1", "NOS3", "VCAM1",
            "hemoglobin", "globin", "beta-globin", "alpha-globin"
        ]
        
        mentions = []
        
        for keyword in gene_keywords:
            if keyword.lower() in text.lower():
                mentions.append(keyword)
        
        return mentions
    
    def _extract_treatment_mentions(self, text: str) -> List[str]:
        """
        Extract treatment mentions from text.
        
        Args:
            text (str): The text to extract from.
            
        Returns:
            List[str]: The extracted treatment mentions.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a more sophisticated approach,
        # such as named entity recognition
        
        treatment_keywords = [
            "CRISPR", "Cas9", "gene therapy", "gene editing", "lentiviral",
            "AAV", "hydroxyurea", "stem cell transplantation", "bone marrow transplantation"
        ]
        
        mentions = []
        
        for keyword in treatment_keywords:
            if keyword.lower() in text.lower():
                mentions.append(keyword)
        
        return mentions
    
    def _extract_mutation_mentions(self, text: str) -> List[str]:
        """
        Extract mutation mentions from text.
        
        Args:
            text (str): The text to extract from.
            
        Returns:
            List[str]: The extracted mutation mentions.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a more sophisticated approach,
        # such as named entity recognition
        
        mutation_keywords = [
            "HbSS", "HbAS", "HbAA", "sickle cell", "hemoglobin S",
            "hemoglobin A", "beta S", "beta A", "point mutation"
        ]
        
        mentions = []
        
        for keyword in mutation_keywords:
            if keyword.lower() in text.lower():
                mentions.append(keyword)
        
        return mentions
    
    def _get_gene_name(self, gene_symbol: str) -> str:
        """
        Get the full name of a gene.
        
        Args:
            gene_symbol (str): The gene symbol.
            
        Returns:
            str: The full name of the gene.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a gene database
        
        gene_names = {
            "HBB": "Hemoglobin Subunit Beta",
            "HBA1": "Hemoglobin Subunit Alpha 1",
            "HBA2": "Hemoglobin Subunit Alpha 2",
            "BCL11A": "BAF Chromatin Remodeling Complex Subunit",
            "HMOX1": "Heme Oxygenase 1",
            "NOS3": "Nitric Oxide Synthase 3",
            "VCAM1": "Vascular Cell Adhesion Molecule 1"
        }
        
        return gene_names.get(gene_symbol, f"Unknown gene {gene_symbol}")
    
    def _get_treatment_type(self, treatment_name: str) -> str:
        """
        Get the type of a treatment.
        
        Args:
            treatment_name (str): The treatment name.
            
        Returns:
            str: The type of the treatment.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a treatment database
        
        treatment_types = {
            "CRISPR": "Gene Therapy",
            "Cas9": "Gene Therapy",
            "gene therapy": "Gene Therapy",
            "gene editing": "Gene Therapy",
            "lentiviral": "Gene Therapy",
            "AAV": "Gene Therapy",
            "hydroxyurea": "Small Molecule",
            "stem cell transplantation": "Cell Therapy",
            "bone marrow transplantation": "Cell Therapy"
        }
        
        return treatment_types.get(treatment_name, "Unknown")
    
    def _get_treatment_mechanism(self, treatment_name: str) -> str:
        """
        Get the mechanism of a treatment.
        
        Args:
            treatment_name (str): The treatment name.
            
        Returns:
            str: The mechanism of the treatment.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a treatment database
        
        treatment_mechanisms = {
            "CRISPR": "Gene Editing",
            "Cas9": "Gene Editing",
            "gene therapy": "Gene Addition",
            "gene editing": "Gene Editing",
            "lentiviral": "Gene Addition",
            "AAV": "Gene Addition",
            "hydroxyurea": "HbF Induction",
            "stem cell transplantation": "Cell Replacement",
            "bone marrow transplantation": "Cell Replacement"
        }
        
        return treatment_mechanisms.get(treatment_name, "Unknown")
    
    def _get_mutation_type(self, mutation_name: str) -> str:
        """
        Get the type of a mutation.
        
        Args:
            mutation_name (str): The mutation name.
            
        Returns:
            str: The type of the mutation.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a mutation database
        
        mutation_types = {
            "HbSS": "Homozygous",
            "HbAS": "Heterozygous",
            "HbAA": "Normal",
            "sickle cell": "Homozygous",
            "hemoglobin S": "Variant",
            "hemoglobin A": "Normal",
            "beta S": "Variant",
            "beta A": "Normal",
            "point mutation": "SNP"
        }
        
        return mutation_types.get(mutation_name, "Unknown")