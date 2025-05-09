"""
Graph database factory for SickleGraph.

This module provides a factory for creating graph database instances.
"""

import os
import logging
from typing import Dict, Optional, Union

from sicklegraph.graph.base import GraphDatabase # type: ignore
from sicklegraph.graph.kuzu_adapter import KuzuAdapter # type: ignore
from sicklegraph.graph.neo4j_adapter import Neo4jAdapter # type: ignore

logger = logging.getLogger(__name__)


class GraphDatabaseFactory:
    """
    Factory for creating graph database instances.
    
    This class provides methods for creating instances of different graph database
    implementations.
    """
    
    @staticmethod
    def create_database(
        db_type: str,
        config: Dict[str, str]
    ) -> Optional[GraphDatabase]:
        """
        Create a graph database instance.
        
        Args:
            db_type (str): The type of database to create ("kuzu" or "neo4j").
            config (Dict[str, str]): The configuration for the database.
            
        Returns:
            Optional[GraphDatabase]: The graph database instance, or None if creation failed.
        """
        try:
            if db_type.lower() == "kuzu":
                if "db_path" not in config:
                    raise ValueError("db_path is required for Kùzu database")
                
                db_path = config["db_path"]
                db = KuzuAdapter(db_path)
                
                logger.info(f"Created Kùzu database adapter with path {db_path}")
                return db
            
            elif db_type.lower() == "neo4j":
                required_keys = ["uri", "username", "password"]
                for key in required_keys:
                    if key not in config:
                        raise ValueError(f"{key} is required for Neo4j database")
                
                uri = config["uri"]
                username = config["username"]
                password = config["password"]
                database = config.get("database", "neo4j")
                
                db = Neo4jAdapter(uri, username, password, database)
                
                logger.info(f"Created Neo4j database adapter with URI {uri}")
                return db
            
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
        
        except Exception as e:
            logger.error(f"Failed to create database: {e}")
            return None
    
    @staticmethod
    def create_from_env() -> Optional[GraphDatabase]:
        """
        Create a graph database instance from environment variables.
        
        Returns:
            Optional[GraphDatabase]: The graph database instance, or None if creation failed.
        """
        try:
            db_type = os.environ.get("SICKLEGRAPH_DB_TYPE", "kuzu").lower()
            
            if db_type == "kuzu":
                db_path = os.environ.get("SICKLEGRAPH_KUZU_PATH", "./data/kuzu")
                return GraphDatabaseFactory.create_database("kuzu", {"db_path": db_path})
            
            elif db_type == "neo4j":
                uri = os.environ.get("SICKLEGRAPH_NEO4J_URI")
                username = os.environ.get("SICKLEGRAPH_NEO4J_USERNAME")
                password = os.environ.get("SICKLEGRAPH_NEO4J_PASSWORD")
                database = os.environ.get("SICKLEGRAPH_NEO4J_DATABASE", "neo4j")
                
                if not uri or not username or not password:
                    raise ValueError("Neo4j environment variables not set")
                
                return GraphDatabaseFactory.create_database(
                    "neo4j", 
                    {
                        "uri": uri,
                        "username": username,
                        "password": password,
                        "database": database
                    }
                )
            
            else:
                raise ValueError(f"Unsupported database type: {db_type}")
        
        except Exception as e:
            logger.error(f"Failed to create database from environment: {e}")
            return None