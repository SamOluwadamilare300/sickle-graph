"""
Base ELIZA AI Research Assistant for SickleGraph.

This module defines the base ELIZA AI Research Assistant.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import json

from sicklegraph.graph import GraphDatabase, NodeLabel, RelationshipType # type: ignore

logger = logging.getLogger(__name__)


class ElizaAssistant:
    """
    Base ELIZA AI Research Assistant for SickleGraph.
    
    This class implements the base ELIZA AI Research Assistant.
    """
    
    def __init__(self, graph: GraphDatabase, model_name: str = "gpt-4"):
        """
        Initialize the ELIZA AI Research Assistant.
        
        Args:
            graph (GraphDatabase): The graph database to query.
            model_name (str): The name of the language model to use.
        """
        self.graph = graph
        self.model_name = model_name
        self.conversation_history = []
        self.language = "en"  # Default language
        
        # Load language resources
        self.language_resources = self._load_language_resources()
    
    def query(self, query_text: str, language: Optional[str] = None) -> str:
        """
        Query the ELIZA AI Research Assistant.
        
        Args:
            query_text (str): The query text.
            language (Optional[str]): The language of the query (en, yo, ha, ig).
            
        Returns:
            str: The response text.
        """
        try:
            # Set the language if provided
            if language:
                self.language = language
            
            # Detect language if not explicitly set
            if not language and self.language == "en":
                detected_language = self._detect_language(query_text)
                if detected_language:
                    self.language = detected_language
            
            # Translate query to English if necessary
            if self.language != "en":
                english_query = self._translate_to_english(query_text, self.language)
                logger.info(f"Translated query from {self.language} to English: {english_query}")
            else:
                english_query = query_text
            
            # Add the query to the conversation history
            self.conversation_history.append({"role": "user", "content": english_query})
            
            # Process the query
            query_type = self._classify_query(english_query)
            
            # Generate the response based on the query type
            if query_type == "graph_query":
                response = self._handle_graph_query(english_query)
            elif query_type == "literature_query":
                response = self._handle_literature_query(english_query)
            elif query_type == "clinical_trial_query":
                response = self._handle_clinical_trial_query(english_query)
            elif query_type == "gene_therapy_query":
                response = self._handle_gene_therapy_query(english_query)
            elif query_type == "african_context_query":
                response = self._handle_african_context_query(english_query)
            else:
                response = self._handle_general_query(english_query)
            
            # Translate response if necessary
            if self.language != "en":
                translated_response = self._translate_from_english(response, self.language)
                logger.info(f"Translated response from English to {self.language}")
                response = translated_response
            
            # Add the response to the conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            
            # Return an error message in the appropriate language
            error_message = self._get_localized_message("error_message")
            return error_message
    
    def _classify_query(self, query_text: str) -> str:
        """
        Classify the query type.
        
        Args:
            query_text (str): The query text.
            
        Returns:
            str: The query type.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a more sophisticated approach,
        # such as a trained classifier
        
        query_lower = query_text.lower()
        
        if any(term in query_lower for term in ["graph", "network", "relationship", "connected", "link"]):
            return "graph_query"
        elif any(term in query_lower for term in ["paper", "publication", "research", "article", "journal"]):
            return "literature_query"
        elif any(term in query_lower for term in ["trial", "clinical", "study", "phase", "patient"]):
            return "clinical_trial_query"
        elif any(term in query_lower for term in ["gene", "therapy", "crispr", "edit", "mutation", "hbb"]):
            return "gene_therapy_query"
        elif any(term in query_lower for term in ["africa", "nigeria", "african", "population", "diversity"]):
            return "african_context_query"
        else:
            return "general_query"
    
    def _handle_graph_query(self, query_text: str) -> str:
        """
        Handle a graph query.
        
        Args:
            query_text (str): The query text.
            
        Returns:
            str: The response text.
        """
        # Extract entities from the query
        entities = self._extract_entities(query_text)
        
        if not entities:
            return self._get_localized_message("no_entities_found")
        
        # Construct and execute a graph query
        results = []
        
        for entity_type, entity_name in entities:
            if entity_type == "gene":
                # Query gene information
                gene_query = f"""
                MATCH (g:{NodeLabel.GENE})
                WHERE g.symbol = '{entity_name}' OR g.name = '{entity_name}'
                RETURN g
                """
                gene_results = self.graph.execute_query(gene_query)
                
                if gene_results:
                    results.append(f"Found gene {entity_name}:")
                    for result in gene_results:
                        gene = result.get("g", {})
                        results.append(f"- Symbol: {gene.get('symbol')}")
                        results.append(f"- Name: {gene.get('name')}")
                        results.append(f"- Description: {gene.get('description')}")
                
                # Query treatments targeting the gene
                treatment_query = f"""
                MATCH (t:{NodeLabel.TREATMENT})-[r:{RelationshipType.TARGETS}]->(g:{NodeLabel.GENE})
                WHERE g.symbol = '{entity_name}' OR g.name = '{entity_name}'
                RETURN t, r
                """
                treatment_results = self.graph.execute_query(treatment_query)
                
                if treatment_results:
                    results.append(f"\nTreatments targeting {entity_name}:")
                    for result in treatment_results:
                        treatment = result.get("t", {})
                        relationship = result.get("r", {})
                        results.append(f"- {treatment.get('name')} ({treatment.get('type')})")
                        results.append(f"  Mechanism: {relationship.get('mechanism')}")
                        results.append(f"  Efficacy: {relationship.get('efficacy')}")
            
            elif entity_type == "treatment":
                # Query treatment information
                treatment_query = f"""
                MATCH (t:{NodeLabel.TREATMENT})
                WHERE t.name CONTAINS '{entity_name}'
                RETURN t
                """
                treatment_results = self.graph.execute_query(treatment_query)
                
                if treatment_results:
                    results.append(f"Found treatment {entity_name}:")
                    for result in treatment_results:
                        treatment = result.get("t", {})
                        results.append(f"- Name: {treatment.get('name')}")
                        results.append(f"- Type: {treatment.get('type')}")
                        results.append(f"- Mechanism: {treatment.get('mechanism')}")
                        results.append(f"- Description: {treatment.get('description')}")
                
                # Query genes targeted by the treatment
                gene_query = f"""
                MATCH (t:{NodeLabel.TREATMENT})-[r:{RelationshipType.TARGETS}]->(g:{NodeLabel.GENE})
                WHERE t.name CONTAINS '{entity_name}'
                RETURN g, r
                """
                gene_results = self.graph.execute_query(gene_query)
                
                if gene_results:
                    results.append(f"\nGenes targeted by {entity_name}:")
                    for result in gene_results:
                        gene = result.get("g", {})
                        relationship = result.get("r", {})
                        results.append(f"- {gene.get('symbol')} ({gene.get('name')})")
                        results.append(f"  Mechanism: {relationship.get('mechanism')}")
                        results.append(f"  Efficacy: {relationship.get('efficacy')}")
        
        if not results:
            return self._get_localized_message("no_results_found")
        
        return "\n".join(results)
    
    def _handle_literature_query(self, query_text: str) -> str:
        """
        Handle a literature query.
        
        Args:
            query_text (str): The query text.
            
        Returns:
            str: The response text.
        """
        # Extract keywords from the query
        keywords = self._extract_keywords(query_text)
        
        if not keywords:
            return self._get_localized_message("no_keywords_found")
        
        # Construct and execute a literature query
        keyword_conditions = []
        for keyword in keywords:
            keyword_conditions.append(f"p.title CONTAINS '{keyword}' OR p.abstract CONTAINS '{keyword}'")
        
        keyword_condition = " OR ".join(keyword_conditions)
        
        query = f"""
        MATCH (p:{NodeLabel.PAPER})
        WHERE {keyword_condition}
        RETURN p
        ORDER BY p.publication_date DESC
        LIMIT 5
        """
        
        results = self.graph.execute_query(query)
        
        if not results:
            return self._get_localized_message("no_papers_found")
        
        # Format the results
        response = [f"Found {len(results)} papers related to {', '.join(keywords)}:"]
        
        for i, result in enumerate(results):
            paper = result.get("p", {})
            response.append(f"\n{i+1}. {paper.get('title')}")
            response.append(f"   Published: {paper.get('publication_date')}")
            response.append(f"   Journal: {paper.get('journal')}")
            
            if paper.get("abstract"):
                abstract = paper.get("abstract")
                if len(abstract) > 200:
                    abstract = abstract[:200] + "..."
                response.append(f"   Abstract: {abstract}")
            
            response.append(f"   URL: {paper.get('url')}")
        
        return "\n".join(response)
    
    def _handle_clinical_trial_query(self, query_text: str) -> str:
        """
        Handle a clinical trial query.
        
        Args:
            query_text (str): The query text.
            
        Returns:
            str: The response text.
        """
        # Check if the query is about African trials
        is_african_query = any(term in query_text.lower() for term in ["africa", "african", "nigeria", "ghana", "kenya"])
        
        # Construct and execute a clinical trial query
        if is_african_query:
            query = f"""
            MATCH (t:{NodeLabel.CLINICAL_TRIAL})-[:{RelationshipType.CONDUCTED_IN}]->(c:{NodeLabel.COUNTRY})-[:{RelationshipType.PART_OF}]->(cont:{NodeLabel.CONTINENT})
            WHERE cont.name = 'Africa'
            RETURN t, c.name as country
            ORDER BY t.start_date DESC
            LIMIT 5
            """
        else:
            # Extract keywords from the query
            keywords = self._extract_keywords(query_text)
            
            if not keywords:
                return self._get_localized_message("no_keywords_found")
            
            # Construct the query
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append(f"t.title CONTAINS '{keyword}' OR t.description CONTAINS '{keyword}'")
            
            keyword_condition = " OR ".join(keyword_conditions)
            
            query = f"""
            MATCH (t:{NodeLabel.CLINICAL_TRIAL})
            WHERE {keyword_condition}
            RETURN t
            ORDER BY t.start_date DESC
            LIMIT 5
            """
        
        results = self.graph.execute_query(query)
        
        if not results:
            if is_african_query:
                return self._get_localized_message("no_african_trials_found")
            else:
                return self._get_localized_message("no_trials_found")
        
        # Format the results
        if is_african_query:
            response = [f"Found {len(results)} clinical trials in Africa:"]
        else:
            response = [f"Found {len(results)} clinical trials:"]
        
        for i, result in enumerate(results):
            trial = result.get("t", {})
            country = result.get("country", "")
            
            response.append(f"\n{i+1}. {trial.get('title')}")
            response.append(f"   Status: {trial.get('status')}")
            response.append(f"   Phase: {trial.get('phase')}")
            
            if country:
                response.append(f"   Location: {country}")
            
            response.append(f"   Start Date: {trial.get('start_date')}")
            
            if trial.get("description"):
                description = trial.get("description")
                if len(description) > 200:
                    description = description[:200] + "..."
                response.append(f"   Description: {description}")
            
            response.append(f"   URL: {trial.get('url')}")
        
        return "\n".join(response)
    
    def _handle_gene_therapy_query(self, query_text: str) -> str:
        """
        Handle a gene therapy query.
        
        Args:
            query_text (str): The query text.
            
        Returns:
            str: The response text.
        """
        # Extract gene names from the query
        gene_names = self._extract_gene_names(query_text)
        
        if not gene_names:
            # If no specific genes mentioned, default to HBB for SCD
            gene_names = ["HBB"]
        
        # Construct and execute a gene therapy query
        gene_conditions = []
        for gene in gene_names:
            gene_conditions.append(f"g.symbol = '{gene}' OR g.name = '{gene}'")
        
        gene_condition = " OR ".join(gene_conditions)
        
        query = f"""
        MATCH (t:{NodeLabel.TREATMENT})-[r:{RelationshipType.TARGETS}]->(g:{NodeLabel.GENE})
        WHERE ({gene_condition}) AND t.type IN ['Genetic', 'Biological', 'Gene Therapy', 'CRISPR']
        RETURN t, g, r
        """
        
        results = self.graph.execute_query(query)
        
        if not results:
            return self._get_localized_message("no_gene_therapies_found")
        
        # Format the results
        response = [f"Found {len(results)} gene therapies targeting {', '.join(gene_names)}:"]
        
        for i, result in enumerate(results):
            treatment = result.get("t", {})
            gene = result.get("g", {})
            relationship = result.get("r", {})
            
            response.append(f"\n{i+1}. {treatment.get('name')} ({treatment.get('type')})")
            response.append(f"   Target Gene: {gene.get('symbol')} ({gene.get('name')})")
            response.append(f"   Mechanism: {relationship.get('mechanism')}")
            
            if treatment.get("description"):
                description = treatment.get("description")
                if len(description) > 200:
                    description = description[:200] + "..."
                response.append(f"   Description: {description}")
            
            # Find clinical trials for this treatment
            trial_query = f"""
            MATCH (t:{NodeLabel.TREATMENT})<-[:{RelationshipType.TARGETS}]-(c:{NodeLabel.CLINICAL_TRIAL})
            WHERE t.id = '{treatment.get('id')}'
            RETURN c
            LIMIT 3
            """
            
            trial_results = self.graph.execute_query(trial_query)
            
            if trial_results:
                response.append(f"   Clinical Trials:")
                for j, trial_result in enumerate(trial_results):
                    trial = trial_result.get("c", {})
                    response.append(f"     {j+1}. {trial.get('title')} (Phase: {trial.get('phase')}, Status: {trial.get('status')})")
        
        return "\n".join(response)
    
    def _handle_african_context_query(self, query_text: str) -> str:
        """
        Handle an African context query.
        
        Args:
            query_text (str): The query text.
            
        Returns:
            str: The response text.
        """
        # Extract country names from the query
        country_names = self._extract_country_names(query_text)
        
        # If no specific countries mentioned, default to Nigeria
        if not country_names:
            country_names = ["Nigeria"]
        
        # Construct and execute a query for clinical trials in the specified countries
        country_conditions = []
        for country in country_names:
            country_conditions.append(f"c.name = '{country}'")
        
        country_condition = " OR ".join(country_conditions)
        
        trial_query = f"""
        MATCH (t:{NodeLabel.CLINICAL_TRIAL})-[:{RelationshipType.CONDUCTED_IN}]->(c:{NodeLabel.COUNTRY})
        WHERE {country_condition}
        RETURN t, c.name as country
        ORDER BY t.start_date DESC
        LIMIT 3
        """
        
        trial_results = self.graph.execute_query(trial_query)
        
        # Construct and execute a query for research papers mentioning the specified countries
        paper_query = f"""
        MATCH (p:{NodeLabel.PAPER})
        WHERE {" OR ".join([f"p.title CONTAINS '{country}' OR p.abstract CONTAINS '{country}'" for country in country_names])}
        RETURN p
        ORDER BY p.publication_date DESC
        LIMIT 3
        """
        
        paper_results = self.graph.execute_query(paper_query)
        
        # Format the results
        response = [f"Information about SCD gene therapy research in {', '.join(country_names)}:"]
        
        if trial_results:
            response.append(f"\nClinical Trials ({len(trial_results)}):")
            for i, result in enumerate(trial_results):
                trial = result.get("t", {})
                country = result.get("country", "")
                
                response.append(f"\n{i+1}. {trial.get('title')}")
                response.append(f"   Status: {trial.get('status')}")
                response.append(f"   Phase: {trial.get('phase')}")
                response.append(f"   Location: {country}")
                response.append(f"   Start Date: {trial.get('start_date')}")
                response.append(f"   URL: {trial.get('url')}")
        else:
            response.append(f"\nNo clinical trials found in {', '.join(country_names)}.")
        
        if paper_results:
            response.append(f"\nResearch Papers ({len(paper_results)}):")
            for i, result in enumerate(paper_results):
                paper = result.get("p", {})
                
                response.append(f"\n{i+1}. {paper.get('title')}")
                response.append(f"   Published: {paper.get('publication_date')}")
                response.append(f"   Journal: {paper.get('journal')}")
                
                if paper.get("abstract"):
                    abstract = paper.get("abstract")
                    if len(abstract) > 200:
                        abstract = abstract[:200] + "..."
                    response.append(f"   Abstract: {abstract}")
                
                response.append(f"   URL: {paper.get('url')}")
        else:
            response.append(f"\nNo research papers found mentioning {', '.join(country_names)}.")
        
        # Add some general information about SCD in Africa
        response.append(f"\nGeneral Information:")
        response.append(f"Sickle cell disease (SCD) has a high prevalence in sub-Saharan Africa, with Nigeria having one of the highest burdens globally. Gene therapy approaches are being developed that could potentially provide curative options for patients in Africa, but challenges include cost, infrastructure, and genetic diversity considerations.")
        
        return "\n".join(response)
    
    def _handle_general_query(self, query_text: str) -> str:
        """
        Handle a general query.
        
        Args:
            query_text (str): The query text.
            
        Returns:
            str: The response text.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a more sophisticated approach,
        # such as a language model
        
        # Extract keywords from the query
        keywords = self._extract_keywords(query_text)
        
        if not keywords:
            return self._get_localized_message("general_help")
        
        # Check if the query is about SCD
        is_scd_query = any(term in query_text.lower() for term in ["sickle", "scd", "anemia", "hemoglobin"])
        
        if is_scd_query:
            return """
Sickle Cell Disease (SCD) is a group of inherited red blood cell disorders. In SCD, the red blood cells become hard and sticky and look like a C-shaped farm tool called a "sickle".

Key facts about SCD:
1. It's caused by a mutation in the HBB gene that leads to abnormal hemoglobin (HbS).
2. When red blood cells contain HbS, they can become sickle-shaped, rigid, and sticky.
3. These sickle cells can block blood flow, causing pain and organ damage.
4. SCD is particularly common in sub-Saharan Africa, especially Nigeria.

Gene therapy approaches for SCD:
1. Gene addition: Adding a functional copy of the HBB gene
2. Gene editing: Using CRISPR-Cas9 to correct the mutation
3. Fetal hemoglobin induction: Reactivating fetal hemoglobin (HbF) production

Current challenges for gene therapy in Africa:
1. High cost of treatment
2. Limited healthcare infrastructure
3. Need for genetic diversity in research
4. Regulatory and ethical considerations

SickleGraph aims to accelerate gene therapy innovation for SCD in Africa by integrating diverse biomedical data and providing tools for researchers.
"""
        else:
            return self._get_localized_message("general_help")
    
    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from text.
        
        Args:
            text (str): The text to extract from.
            
        Returns:
            List[Tuple[str, str]]: The extracted entities as (type, name) tuples.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a more sophisticated approach,
        # such as named entity recognition
        
        entities = []
        
        # Extract gene names
        gene_pattern = r'\b(HBB|HBA1|HBA2|BCL11A|HMOX1|NOS3|VCAM1)\b'
        gene_matches = re.findall(gene_pattern, text)
        
        for gene in gene_matches:
            entities.append(("gene", gene))
        
        # Extract treatment names
        treatment_keywords = ["CRISPR", "gene therapy", "gene editing", "lentiviral", "AAV"]
        
        for keyword in treatment_keywords:
            if keyword.lower() in text.lower():
                entities.append(("treatment", keyword))
        
        return entities
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text (str): The text to extract from.
            
        Returns:
            List[str]: The extracted keywords.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a more sophisticated approach,
        # such as keyword extraction algorithms
        
        # Remove common words and punctuation
        stop_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with", "about", "is", "are", "was", "were"]
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Add specific SCD-related terms if they appear in the text
        scd_terms = ["sickle", "scd", "hemoglobin", "anemia", "hbs", "hbf", "crispr", "gene", "therapy", "africa", "nigeria"]
        
        for term in scd_terms:
            if term in text.lower() and term not in keywords:
                keywords.append(term)
        
        return keywords
    
    def _extract_gene_names(self, text: str) -> List[str]:
        """
        Extract gene names from text.
        
        Args:
            text (str): The text to extract from.
            
        Returns:
            List[str]: The extracted gene names.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a more sophisticated approach,
        # such as named entity recognition
        
        gene_pattern = r'\b(HBB|HBA1|HBA2|BCL11A|HMOX1|NOS3|VCAM1)\b'
        gene_matches = re.findall(gene_pattern, text)
        
        return gene_matches
    
    def _extract_country_names(self, text: str) -> List[str]:
        """
        Extract country names from text.
        
        Args:
            text (str): The text to extract from.
            
        Returns:
            List[str]: The extracted country names.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a more sophisticated approach,
        # such as named entity recognition
        
        african_countries = [
            "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi",
            "Cabo Verde", "Cameroon", "Central African Republic", "Chad", "Comoros",
            "Congo", "Côte d'Ivoire", "Djibouti", "Egypt", "Equatorial Guinea",
            "Eritrea", "Eswatini", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
            "Guinea-Bissau", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar",
            "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique",
            "Namibia", "Niger", "Nigeria", "Rwanda", "Sao Tome and Principe",
            "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa",
            "South Sudan", "Sudan", "Tanzania", "Togo", "Tunisia", "Uganda",
            "Zambia", "Zimbabwe"
        ]
        
        found_countries = []
        
        for country in african_countries:
            if country.lower() in text.lower():
                found_countries.append(country)
        
        return found_countries
    
    def _detect_language(self, text: str) -> Optional[str]:
        """
        Detect the language of the text.
        
        Args:
            text (str): The text to detect.
            
        Returns:
            Optional[str]: The detected language code, or None if detection failed.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a more sophisticated approach,
        # such as a language detection library
        
        # Check for Yoruba-specific characters and words
        yoruba_words = ["ẹ", "ọ", "ṣ", "jọwọ", "pẹlẹ", "ṣeun", "bawo"]
        if any(word in text.lower() for word in yoruba_words):
            return "yo"
        
        # Check for Hausa-specific words
        hausa_words = ["sannu", "yaya", "kaka", "nagode", "madalla", "lafiya"]
        if any(word in text.lower() for word in hausa_words):
            return "ha"
        
        # Check for Igbo-specific words
        igbo_words = ["kedu", "biko", "daalu", "ndewo", "nnọọ", "maka"]
        if any(word in text.lower() for word in igbo_words):
            return "ig"
        
        # Default to English
        return None
    
    def _translate_to_english(self, text: str, source_language: str) -> str:
        """
        Translate text from the source language to English.
        
        Args:
            text (str): The text to translate.
            source_language (str): The source language code.
            
        Returns:
            str: The translated text.
        """
        # This is a placeholder implementation
        # In a real-world scenario, you would use a translation service
        
        # For demonstration purposes, we'll just return the original text
        # with a note that it would be translated
        return f"[Translated from {source_language} to English]: {text}"
    
    def _translate_from_english(self, text: str, target_language: str) -> str:
        """
        Translate text from English to the target language.
        
        Args:
            text (str): The text to translate.
            target_language (str): The target language code.
            
        Returns:
            str: The translated text.
        """
        # This is a placeholder implementation
        # In a real-world scenario, you would use a translation service
        
        # For demonstration purposes, we'll just return the original text
        # with a note that it would be translated
        return f"[Translated from English to {target_language}]: {text}"
    
    def _load_language_resources(self) -> Dict[str, Dict[str, str]]:
        """
        Load language resources for localization.
        
        Returns:
            Dict[str, Dict[str, str]]: The language resources.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would load these from files
        
        return {
            "en": {
                "error_message": "I'm sorry, I encountered an error while processing your query. Please try again.",
                "no_entities_found": "I couldn't identify any specific entities in your query. Please try again with more specific terms.",
                "no_results_found": "I couldn't find any results matching your query. Please try again with different terms.",
                "no_keywords_found": "I couldn't identify any keywords in your query. Please try again with more specific terms.",
                "no_papers_found": "I couldn't find any research papers matching your query. Please try again with different terms.",
                "no_trials_found": "I couldn't find any clinical trials matching your query. Please try again with different terms.",
                "no_african_trials_found": "I couldn't find any clinical trials in Africa matching your query. Please try again with different terms.",
                "no_gene_therapies_found": "I couldn't find any gene therapies matching your query. Please try again with different terms.",
                "general_help": """
I'm ELIZA, an AI research assistant for SickleGraph. I can help you with:

1. Information about sickle cell disease (SCD) and gene therapy
2. Finding research papers on SCD and gene therapy
3. Information about clinical trials for SCD gene therapy
4. Details about specific genes and treatments
5. Context about SCD gene therapy research in Africa

How can I assist you today?
"""
            },
            "yo": {
                "error_message": "Mo kanu, mo ni iṣoro nigba ti mo n ṣe ibeere rẹ. Jọwọ gbiyanju lẹẹkansi.",
                "general_help": "Mo je ELIZA, oluranlọwọ iwadi AI fun SickleGraph. Mo le ran ọ lọwọ pẹlu awọn ibeere nipa arun sickle cell ati iwosan gene."
            },
            "ha": {
                "error_message": "Yi hauka, na sami matsala yayin da nake aiki da tambayarka. Da fatan za a sake gwadawa.",
                "general_help": "Ni ne ELIZA, mataimakiyar bincike ta AI don SickleGraph. Zan iya taimaka maka da tambayoyi game da cutar sickle cell da magani na gene."
            },
            "ig": {
                "error_message": "Ndo, enwere nsogbu mgbe m na-arụ ọrụ na ajụjụ gị. Biko nwaa ọzọ.",
                "general_help": "Abụ m ELIZA, onye nyemaka nyocha AI maka SickleGraph. Enwere m ike inyere gị aka na ajụjụ gbasara ọrịa sickle cell na ọgwụgwọ gene."
            }
        }
    
    def _get_localized_message(self, message_key: str) -> str:
        """
        Get a localized message.
        
        Args:
            message_key (str): The message key.
            
        Returns:
            str: The localized message.
        """
        # Get the message for the current language, or fall back to English
        language_resources = self.language_resources.get(self.language, {})
        message = language_resources.get(message_key)
        
        if not message:
            # Fall back to English
            message = self.language_resources.get("en", {}).get(message_key, "")
        
        return message