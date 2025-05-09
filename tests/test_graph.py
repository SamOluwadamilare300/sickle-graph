"""
Unit tests for the graph module.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from sicklegraph.graph import (
    GraphDatabaseFactory,
    KnowledgeGraphSchema,
    create_initial_schema
)


class TestKnowledgeGraphSchema(unittest.TestCase):
    """Tests for the KnowledgeGraphSchema class."""
    
    def test_create_initial_schema(self):
        """Test creating the initial schema."""
        schema = create_initial_schema()
        self.assertIsInstance(schema, KnowledgeGraphSchema)
    
    def test_generate_kuzu_schema(self):
        """Test generating Kùzu schema statements."""
        schema = create_initial_schema()
        statements = schema.generate_kuzu_schema()
        self.assertIsInstance(statements, list)
        self.assertTrue(len(statements) > 0)
        
        # Check that the statements are valid
        for stmt in statements:
            self.assertIsInstance(stmt, str)
            self.assertTrue("CREATE" in stmt)
    
    def test_generate_neo4j_schema(self):
        """Test generating Neo4j schema statements."""
        schema = create_initial_schema()
        statements = schema.generate_neo4j_schema()
        self.assertIsInstance(statements, list)
        self.assertTrue(len(statements) > 0)
        
        # Check that the statements are valid
        for stmt in statements:
            self.assertIsInstance(stmt, str)
            self.assertTrue("CREATE" in stmt)
    
    def test_to_json(self):
        """Test converting the schema to JSON."""
        schema = create_initial_schema()
        json_str = schema.to_json()
        self.assertIsInstance(json_str, str)
        self.assertTrue("nodes" in json_str)
        self.assertTrue("relationships" in json_str)
    
    def test_from_json(self):
        """Test creating a schema from JSON."""
        schema = create_initial_schema()
        json_str = schema.to_json()
        
        new_schema = KnowledgeGraphSchema.from_json(json_str)
        self.assertIsInstance(new_schema, KnowledgeGraphSchema)
        self.assertEqual(len(schema.node_schemas), len(new_schema.node_schemas))
        self.assertEqual(len(schema.relationship_schemas), len(new_schema.relationship_schemas))


class TestGraphDatabaseFactory(unittest.TestCase):
    """Tests for the GraphDatabaseFactory class."""
    
    @patch("sicklegraph.graph.factory.KuzuAdapter")
    def test_create_kuzu_database(self, mock_kuzu_adapter):
        """Test creating a Kùzu database."""
        # Mock the KuzuAdapter
        mock_kuzu_adapter.return_value = MagicMock()
        
        # Create the database
        db = GraphDatabaseFactory.create_database("kuzu", {"db_path": "./data/kuzu"})
        
        # Check that the database was created
        self.assertIsNotNone(db)
        mock_kuzu_adapter.assert_called_once_with("./data/kuzu")
    
    @patch("sicklegraph.graph.factory.Neo4jAdapter")
    def test_create_neo4j_database(self, mock_neo4j_adapter):
        """Test creating a Neo4j database."""
        # Mock the Neo4jAdapter
        mock_neo4j_adapter.return_value = MagicMock()
        
        # Create the database
        db = GraphDatabaseFactory.create_database(
            "neo4j",
            {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password"
            }
        )
        
        # Check that the database was created
        self.assertIsNotNone(db)
        mock_neo4j_adapter.assert_called_once_with(
            "bolt://localhost:7687",
            "neo4j",
            "password",
            "neo4j"
        )
    
    def test_create_unsupported_database(self):
        """Test creating an unsupported database."""
        # Create the database
        db = GraphDatabaseFactory.create_database("unsupported", {})
        
        # Check that the database was not created
        self.assertIsNone(db)
    
    @patch("sicklegraph.graph.factory.os.environ")
    @patch("sicklegraph.graph.factory.GraphDatabaseFactory.create_database")
    def test_create_from_env_kuzu(self, mock_create_database, mock_environ):
        """Test creating a database from environment variables (Kùzu)."""
        # Mock the environment variables
        mock_environ.get.side_effect = lambda key, default=None: {
            "SICKLEGRAPH_DB_TYPE": "kuzu",
            "SICKLEGRAPH_KUZU_PATH": "./data/kuzu"
        }.get(key, default)
        
        # Mock the create_database method
        mock_create_database.return_value = MagicMock()
        
        # Create the database
        db = GraphDatabaseFactory.create_from_env()
        
        # Check that the database was created
        self.assertIsNotNone(db)
        mock_create_database.assert_called_once_with("kuzu", {"db_path": "./data/kuzu"})
    
    @patch("sicklegraph.graph.factory.os.environ")
    @patch("sicklegraph.graph.factory.GraphDatabaseFactory.create_database")
    def test_create_from_env_neo4j(self, mock_create_database, mock_environ):
        """Test creating a database from environment variables (Neo4j)."""
        # Mock the environment variables
        mock_environ.get.side_effect = lambda key, default=None: {
            "SICKLEGRAPH_DB_TYPE": "neo4j",
            "SICKLEGRAPH_NEO4J_URI": "bolt://localhost:7687",
            "SICKLEGRAPH_NEO4J_USERNAME": "neo4j",
            "SICKLEGRAPH_NEO4J_PASSWORD": "password",
            "SICKLEGRAPH_NEO4J_DATABASE": "neo4j"
        }.get(key, default)
        
        # Mock the create_database method
        mock_create_database.return_value = MagicMock()
        
        # Create the database
        db = GraphDatabaseFactory.create_from_env()
        
        # Check that the database was created
        self.assertIsNotNone(db)
        mock_create_database.assert_called_once_with(
            "neo4j",
            {
                "uri": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password",
                "database": "neo4j"
            }
        )


if __name__ == "__main__":
    unittest.main()