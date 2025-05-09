"""
Unit tests for the inference engine.
"""

import unittest
from unittest.mock import MagicMock, patch

from sicklegraph.inference import InferenceEngine, TargetPredictor, TrialMatcher 
from sicklegraph.graph import NodeLabel, RelationshipType 


class TestInferenceEngine(unittest.TestCase):
    """Tests for the InferenceEngine class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a mock graph database
        self.graph = MagicMock()
        
        # Create the inference engine
        self.engine = InferenceEngine(self.graph)
    
    def test_init(self):
        """Test initializing the inference engine."""
        self.assertEqual(self.engine.graph, self.graph)
    
    def test_predict_gene_targets(self):
        """Test predicting gene targets."""
        # Mock the execute_query method
        self.graph.execute_query.side_effect = [
            # First call: get all genes
            [
                {"g": {"id": "gene_hbb", "symbol": "HBB", "name": "Hemoglobin Subunit Beta"}},
                {"g": {"id": "gene_hba1", "symbol": "HBA1", "name": "Hemoglobin Subunit Alpha 1"}}
            ],
            # Second call: get treatments targeting HBB
            [
                {
                    "t": {"id": "treatment_crispr", "name": "CRISPR-Cas9", "type": "Gene Therapy"},
                    "r": {"mechanism": "Gene Editing", "efficacy": 0.8}
                }
            ],
            # Third call: get papers mentioning HBB
            [{"paper_count": 10}],
            # Fourth call: get treatments targeting HBA1
            [],
            # Fifth call: get papers mentioning HBA1
            [{"paper_count": 5}]
        ]
        
        # Predict gene targets
        targets = self.engine.predict_gene_targets("HbSS", "nigerian", 2)
        
        # Check the targets
        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0]["gene_symbol"], "HBB")
        self.assertEqual(targets[1]["gene_symbol"], "HBA1")
    
    def test_predict_off_targets(self):
        """Test predicting off-target effects."""
        # Mock the execute_query method
        self.graph.execute_query.side_effect = [
            # First call: get the target gene
            [{"g": {"id": "gene_hbb", "symbol": "HBB", "name": "Hemoglobin Subunit Beta", "chromosome": "11"}}],
            # Second call: get all genes
            [
                {"g": {"id": "gene_hba1", "symbol": "HBA1", "name": "Hemoglobin Subunit Alpha 1", "chromosome": "16"}},
                {"g": {"id": "gene_hbd", "symbol": "HBD", "name": "Hemoglobin Subunit Delta", "chromosome": "11"}}
            ]
        ]
        
        # Predict off-targets
        off_targets = self.engine.predict_off_targets("HBB", "nigerian", 2)
        
        # Check the off-targets
        self.assertEqual(len(off_targets), 2)
    
    def test_match_clinical_trials(self):
        """Test matching clinical trials."""
        # Mock the execute_query method
        self.graph.execute_query.return_value = [
            {
                "t": {
                    "id": "nct_123",
                    "title": "Gene Therapy for Sickle Cell Disease",
                    "status": "Recruiting",
                    "phase": "Phase 1/2",
                    "start_date": "2022-01-01",
                    "url": "https://clinicaltrials.gov/study/NCT123"
                },
                "country": "Nigeria"
            },
            {
                "t": {
                    "id": "nct_456",
                    "title": "CRISPR-Cas9 for Sickle Cell Disease",
                    "status": "Active, not recruiting",
                    "phase": "Phase 1",
                    "start_date": "2021-06-01",
                    "url": "https://clinicaltrials.gov/study/NCT456"
                },
                "country": "Ghana"
            }
        ]
        
        # Match clinical trials
        trials = self.engine.match_clinical_trials("HbSS", "african", 2)
        
        # Check the trials
        self.assertEqual(len(trials), 2)
        self.assertEqual(trials[0]["trial_id"], "nct_123")
        self.assertEqual(trials[1]["trial_id"], "nct_456")
    
    def test_predict_treatment_outcomes(self):
        """Test predicting treatment outcomes."""
        # Mock the execute_query method
        self.graph.execute_query.return_value = [
            {
                "t": {
                    "id": "treatment_crispr",
                    "name": "CRISPR-Cas9",
                    "type": "Gene Therapy"
                }
            }
        ]
        
        # Predict treatment outcomes
        outcome = self.engine.predict_treatment_outcomes("treatment_crispr", "HbSS", "nigerian")
        
        # Check the outcome
        self.assertEqual(outcome["treatment_id"], "treatment_crispr")
        self.assertEqual(outcome["treatment_name"], "CRISPR-Cas9")
        self.assertEqual(outcome["treatment_type"], "Gene Therapy")
        self.assertEqual(outcome["mutation_type"], "HbSS")
        self.assertEqual(outcome["population"], "nigerian")
        self.assertIn("efficacy", outcome)
        self.assertIn("safety", outcome)
        self.assertIn("durability", outcome)
        self.assertIn("cost_effectiveness", outcome)
        self.assertIn("overall_score", outcome)
        self.assertIn("confidence", outcome)


class TestTargetPredictor(unittest.TestCase):
    """Tests for the TargetPredictor class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a mock graph database
        self.graph = MagicMock()
        
        # Create the target predictor
        self.predictor = TargetPredictor(self.graph)
    
    def test_init(self):
        """Test initializing the target predictor."""
        self.assertEqual(self.predictor.graph, self.graph)
        self.assertIsInstance(self.predictor.engine, InferenceEngine)
    
    @patch("sicklegraph.inference.target_predictor.InferenceEngine.predict_gene_targets")
    def test_predict_targets(self, mock_predict_gene_targets):
        """Test predicting gene therapy targets."""
        # Mock the predict_gene_targets method
        mock_predict_gene_targets.return_value = [
            {
                "rank": 1,
                "gene_id": "gene_hbb",
                "gene_symbol": "HBB",
                "gene_name": "Hemoglobin Subunit Beta",
                "score": 0.9,
                "treatments": [
                    {"id": "treatment_crispr", "name": "CRISPR-Cas9", "type": "Gene Therapy"}
                ]
            }
        ]
        
        # Predict targets
        targets = self.predictor.predict_targets("HbSS", "nigerian", 1)
        
        # Check the targets
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["gene_symbol"], "HBB")
        mock_predict_gene_targets.assert_called_once_with("HbSS", "nigerian", 1)
    
    @patch("sicklegraph.inference.target_predictor.InferenceEngine.predict_off_targets")
    def test_predict_off_targets(self, mock_predict_off_targets):
        """Test predicting off-target effects."""
        # Mock the predict_off_targets method
        mock_predict_off_targets.return_value = [
            {
                "rank": 1,
                "gene_id": "gene_hbd",
                "gene_symbol": "HBD",
                "gene_name": "Hemoglobin Subunit Delta",
                "score": 0.7,
                "reason": "Potential off-target for HBB editing"
            }
        ]
        
        # Predict off-targets
        off_targets = self.predictor.predict_off_targets("HBB", "nigerian", 1)
        
        # Check the off-targets
        self.assertEqual(len(off_targets), 1)
        self.assertEqual(off_targets[0]["gene_symbol"], "HBD")
        mock_predict_off_targets.assert_called_once_with("HBB", "nigerian", 1)
    
    def test_get_target_details(self):
        """Test getting target details."""
        # Mock the execute_query method
        self.graph.execute_query.side_effect = [
            # First call: get the gene
            [
                {
                    "g": {
                        "id": "gene_hbb",
                        "symbol": "HBB",
                        "name": "Hemoglobin Subunit Beta",
                        "chromosome": "11",
                        "start_position": 5246694,
                        "end_position": 5248301,
                        "description": "Hemoglobin subunit beta is a protein that in humans is encoded by the HBB gene."
                    }
                }
            ],
            # Second call: get treatments targeting the gene
            [
                {
                    "t": {"id": "treatment_crispr", "name": "CRISPR-Cas9", "type": "Gene Therapy"},
                    "r": {"mechanism": "Gene Editing", "efficacy": 0.8}
                }
            ],
            # Third call: get papers mentioning the gene
            [
                {
                    "p": {
                        "id": "pubmed_123",
                        "title": "CRISPR-Cas9 gene editing for sickle cell disease",
                        "publication_date": "2022-01-01",
                        "journal": "Nature",
                        "url": "https://pubmed.ncbi.nlm.nih.gov/123/"
                    }
                }
            ]
        ]
        
        # Get target details
        details = self.predictor.get_target_details("HBB")
        
        # Check the details
        self.assertEqual(details["gene_id"], "gene_hbb")
        self.assertEqual(details["gene_symbol"], "HBB")
        self.assertEqual(details["gene_name"], "Hemoglobin Subunit Beta")
        self.assertEqual(details["chromosome"], "11")
        self.assertEqual(details["start_position"], 5246694)
        self.assertEqual(details["end_position"], 5248301)
        self.assertEqual(details["description"], "Hemoglobin subunit beta is a protein that in humans is encoded by the HBB gene.")
        
        self.assertEqual(len(details["treatments"]), 1)
        self.assertEqual(details["treatments"][0]["id"], "treatment_crispr")
        self.assertEqual(details["treatments"][0]["name"], "CRISPR-Cas9")
        self.assertEqual(details["treatments"][0]["type"], "Gene Therapy")
        self.assertEqual(details["treatments"][0]["mechanism"], "Gene Editing")
        self.assertEqual(details["treatments"][0]["efficacy"], 0.8)
        
        self.assertEqual(len(details["papers"]), 1)
        self.assertEqual(details["papers"][0]["id"], "pubmed_123")
        self.assertEqual(details["papers"][0]["title"], "CRISPR-Cas9 gene editing for sickle cell disease")
        self.assertEqual(details["papers"][0]["publication_date"], "2022-01-01")
        self.assertEqual(details["papers"][0]["journal"], "Nature")
        self.assertEqual(details["papers"][0]["url"], "https://pubmed.ncbi.nlm.nih.gov/123/")


class TestTrialMatcher(unittest.TestCase):
    """Tests for the TrialMatcher class."""
    
    def setUp(self):
        """Set up the test case."""
        # Create a mock graph database
        self.graph = MagicMock()
        
        # Create the trial matcher
        self.matcher = TrialMatcher(self.graph)
    
    def test_init(self):
        """Test initializing the trial matcher."""
        self.assertEqual(self.matcher.graph, self.graph)
        self.assertIsInstance(self.matcher.engine, InferenceEngine)
    
    @patch("sicklegraph.inference.trial_matcher.InferenceEngine.match_clinical_trials")
    def test_match_trials(self, mock_match_clinical_trials):
        """Test matching clinical trials."""
        # Mock the match_clinical_trials method
        mock_match_clinical_trials.return_value = [
            {
                "rank": 1,
                "trial_id": "nct_123",
                "title": "Gene Therapy for Sickle Cell Disease",
                "status": "Recruiting",
                "phase": "Phase 1/2",
                "country": "Nigeria",
                "start_date": "2022-01-01",
                "url": "https://clinicaltrials.gov/study/NCT123",
                "score": 0.95
            }
        ]
        
        # Match trials
        trials = self.matcher.match_trials("HbSS", "nigerian", 1)
        
        # Check the trials
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0]["trial_id"], "nct_123")
        mock_match_clinical_trials.assert_called_once_with("HbSS", "nigerian", 1)


if __name__ == "__main__":
    unittest.main()