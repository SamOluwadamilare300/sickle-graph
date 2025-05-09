"""
Unit tests for the ELIZA AI Research Assistant.
"""

import unittest
from unittest.mock import MagicMock, patch

from sicklegraph.eliza import ElizaAssistant # type: ignore
from sicklegraph.graph import NodeLabel, RelationshipType # type: ignore


class TestElizaAssistant(unittest.TestCase):
    """Tests for the ElizaAssistant class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a mock graph database
        self.graph = MagicMock()
        
        # Create the ELIZA assistant
        self.eliza = ElizaAssistant(self.graph)
    
    def test_init(self):
        """Test initializing the ELIZA assistant."""
        self.assertEqual(self.eliza.graph, self.graph)
        self.assertEqual(self.eliza.model_name, "gpt-4")
        self.assertEqual(self.eliza.conversation_history, [])
        self.assertEqual(self.eliza.language, "en")
    
    def test_query_general(self):
        """Test querying the ELIZA assistant with a general query."""
        # Mock the _handle_general_query method
        self.eliza._handle_general_query = MagicMock(return_value="General response")
        
        # Query the assistant
        response = self.eliza.query("Hello, how can you help me?")
        
        # Check the response
        self.assertEqual(response, "General response")
        self.eliza._handle_general_query.assert_called_once_with("Hello, how can you help me?")
        
        # Check that the conversation history was updated
        self.assertEqual(len(self.eliza.conversation_history), 2)
        self.assertEqual(self.eliza.conversation_history[0]["role"], "user")
        self.assertEqual(self.eliza.conversation_history[0]["content"], "Hello, how can you help me?")
        self.assertEqual(self.eliza.conversation_history[1]["role"], "assistant")
        self.assertEqual(self.eliza.conversation_history[1]["content"], "General response")
    
    def test_query_graph(self):
        """Test querying the ELIZA assistant with a graph query."""
        # Mock the _classify_query method
        self.eliza._classify_query = MagicMock(return_value="graph_query")
        
        # Mock the _handle_graph_query method
        self.eliza._handle_graph_query = MagicMock(return_value="Graph response")
        
        # Query the assistant
        response = self.eliza.query("Show me the relationship between HBB and CRISPR")
        
        # Check the response
        self.assertEqual(response, "Graph response")
        self.eliza._classify_query.assert_called_once_with("Show me the relationship between HBB and CRISPR")
        self.eliza._handle_graph_query.assert_called_once_with("Show me the relationship between HBB and CRISPR")
    
    def test_query_literature(self):
        """Test querying the ELIZA assistant with a literature query."""
        # Mock the _classify_query method
        self.eliza._classify_query = MagicMock(return_value="literature_query")
        
        # Mock the _handle_literature_query method
        self.eliza._handle_literature_query = MagicMock(return_value="Literature response")
        
        # Query the assistant
        response = self.eliza.query("Find papers about sickle cell disease gene therapy")
        
        # Check the response
        self.assertEqual(response, "Literature response")
        self.eliza._classify_query.assert_called_once_with("Find papers about sickle cell disease gene therapy")
        self.eliza._handle_literature_query.assert_called_once_with("Find papers about sickle cell disease gene therapy")
    
    def test_query_clinical_trial(self):
        """Test querying the ELIZA assistant with a clinical trial query."""
        # Mock the _classify_query method
        self.eliza._classify_query = MagicMock(return_value="clinical_trial_query")
        
        # Mock the _handle_clinical_trial_query method
        self.eliza._handle_clinical_trial_query = MagicMock(return_value="Clinical trial response")
        
        # Query the assistant
        response = self.eliza.query("Find clinical trials for sickle cell disease")
        
        # Check the response
        self.assertEqual(response, "Clinical trial response")
        self.eliza._classify_query.assert_called_once_with("Find clinical trials for sickle cell disease")
        self.eliza._handle_clinical_trial_query.assert_called_once_with("Find clinical trials for sickle cell disease")
    
    def test_query_gene_therapy(self):
        """Test querying the ELIZA assistant with a gene therapy query."""
        # Mock the _classify_query method
        self.eliza._classify_query = MagicMock(return_value="gene_therapy_query")
        
        # Mock the _handle_gene_therapy_query method
        self.eliza._handle_gene_therapy_query = MagicMock(return_value="Gene therapy response")
        
        # Query the assistant
        response = self.eliza.query("Tell me about gene therapy for HBB")
        
        # Check the response
        self.assertEqual(response, "Gene therapy response")
        self.eliza._classify_query.assert_called_once_with("Tell me about gene therapy for HBB")
        self.eliza._handle_gene_therapy_query.assert_called_once_with("Tell me about gene therapy for HBB")
    
    def test_query_african_context(self):
        """Test querying the ELIZA assistant with an African context query."""
        # Mock the _classify_query method
        self.eliza._classify_query = MagicMock(return_value="african_context_query")
        
        # Mock the _handle_african_context_query method
        self.eliza._handle_african_context_query = MagicMock(return_value="African context response")
        
        # Query the assistant
        response = self.eliza.query("Tell me about sickle cell disease in Nigeria")
        
        # Check the response
        self.assertEqual(response, "African context response")
        self.eliza._classify_query.assert_called_once_with("Tell me about sickle cell disease in Nigeria")
        self.eliza._handle_african_context_query.assert_called_once_with("Tell me about sickle cell disease in Nigeria")
    
    def test_query_error(self):
        """Test querying the ELIZA assistant with an error."""
        # Mock the _classify_query method to raise an exception
        self.eliza._classify_query = MagicMock(side_effect=Exception("Test error"))
        
        # Mock the _get_localized_message method
        self.eliza._get_localized_message = MagicMock(return_value="Error message")
        
        # Query the assistant
        response = self.eliza.query("This will cause an error")
        
        # Check the response
        self.assertEqual(response, "Error message")
        self.eliza._classify_query.assert_called_once_with("This will cause an error")
        self.eliza._get_localized_message.assert_called_once_with("error_message")
    
    def test_query_language(self):
        """Test querying the ELIZA assistant with a different language."""
        # Mock the _handle_general_query method
        self.eliza._handle_general_query = MagicMock(return_value="General response")
        
        # Mock the _translate_from_english method
        self.eliza._translate_from_english = MagicMock(return_value="Translated response")
        
        # Query the assistant
        response = self.eliza.query("Hello", "yo")
        
        # Check the response
        self.assertEqual(response, "Translated response")
        self.assertEqual(self.eliza.language, "yo")
        self.eliza._handle_general_query.assert_called_once_with("Hello")
        self.eliza._translate_from_english.assert_called_once_with("General response", "yo")
    
    def test_classify_query(self):
        """Test classifying queries."""
        # Test graph query
        self.assertEqual(self.eliza._classify_query("Show me the relationship between HBB and CRISPR"), "graph_query")
        
        # Test literature query
        self.assertEqual(self.eliza._classify_query("Find papers about sickle cell disease"), "literature_query")
        
        # Test clinical trial query
        self.assertEqual(self.eliza._classify_query("Find clinical trials for sickle cell disease"), "clinical_trial_query")
        
        # Test gene therapy query
        self.assertEqual(self.eliza._classify_query("Tell me about gene therapy for HBB"), "gene_therapy_query")
        
        # Test African context query
        self.assertEqual(self.eliza._classify_query("Tell me about sickle cell disease in Nigeria"), "african_context_query")
        
        # Test general query
        self.assertEqual(self.eliza._classify_query("Hello, how can you help me?"), "general_query")
    
    def test_extract_entities(self):
        """Test extracting entities from text."""
        # Test extracting gene names
        entities = self.eliza._extract_entities("HBB is a gene involved in sickle cell disease")
        self.assertIn(("gene", "HBB"), entities)
        
        # Test extracting treatment names
        entities = self.eliza._extract_entities("CRISPR is a gene editing technology")
        self.assertIn(("treatment", "CRISPR"), entities)
    
    def test_extract_keywords(self):
        """Test extracting keywords from text."""
        # Test extracting keywords
        keywords = self.eliza._extract_keywords("Sickle cell disease is a genetic disorder")
        self.assertIn("sickle", keywords)
        self.assertIn("cell", keywords)
        self.assertIn("disease", keywords)
        self.assertIn("genetic", keywords)
        self.assertIn("disorder", keywords)
    
    def test_extract_gene_names(self):
        """Test extracting gene names from text."""
        # Test extracting gene names
        gene_names = self.eliza._extract_gene_names("HBB and HBA1 are genes involved in sickle cell disease")
        self.assertIn("HBB", gene_names)
        self.assertIn("HBA1", gene_names)
    
    def test_extract_country_names(self):
        """Test extracting country names from text."""
        # Test extracting country names
        country_names = self.eliza._extract_country_names("Nigeria and Ghana have high prevalence of sickle cell disease")
        self.assertIn("Nigeria", country_names)
        self.assertIn("Ghana", country_names)
    
    def test_detect_language(self):
        """Test detecting language."""
        # Test detecting Yoruba
        self.assertEqual(self.eliza._detect_language("Jọwọ, ṣe alaye nipa arun sickle cell"), "yo")
        
        # Test detecting Hausa
        self.assertEqual(self.eliza._detect_language("Sannu, ina wani bayani game da cutar sickle cell"), "ha")
        
        # Test detecting Igbo
        self.assertEqual(self.eliza._detect_language("Biko, gwam maka oria sickle cell"), "ig")
        
        # Test detecting English (default)
        self.assertIsNone(self.eliza._detect_language("Please tell me about sickle cell disease"))
    
    def test_get_localized_message(self):
        """Test getting localized messages."""
        # Test getting an English message
        self.eliza.language = "en"
        message = self.eliza._get_localized_message("error_message")
        self.assertEqual(message, "I'm sorry, I encountered an error while processing your query. Please try again.")
        
        # Test getting a Yoruba message
        self.eliza.language = "yo"
        message = self.eliza._get_localized_message("error_message")
        self.assertEqual(message, "Mo kanu, mo ni iṣoro nigba ti mo n ṣe ibeere rẹ. Jọwọ gbiyanju lẹẹkansi.")
        
        # Test falling back to English for a missing message
        self.eliza.language = "yo"
        message = self.eliza._get_localized_message("no_entities_found")
        self.assertEqual(message, "I couldn't identify any specific entities in your query. Please try again with more specific terms.")


if __name__ == "__main__":
    unittest.main()