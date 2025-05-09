"""
Graph module for SickleGraph.

This module provides the main graph interface for SickleGraph.
"""

from sicklegraph.graph.base import GraphDatabase # type: ignore
from sicklegraph.graph.kuzu_adapter import KuzuAdapter # type: ignore
from sicklegraph.graph.neo4j_adapter import Neo4jAdapter # type: ignore
from sicklegraph.graph.factory import GraphDatabaseFactory # type: ignore
from sicklegraph.graph.schema import ( # type: ignore
    KnowledgeGraphSchema,
    NodeLabel,
    RelationshipType,
    create_initial_schema
)

__all__ = [
    "GraphDatabase",
    "KuzuAdapter",
    "Neo4jAdapter",
    "GraphDatabaseFactory",
    "KnowledgeGraphSchema",
    "NodeLabel",
    "RelationshipType",
    "create_initial_schema"
]