"""
Base graph database interface for SickleGraph.

This module defines the abstract base class for graph database implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

from sicklegraph.graph.schema import KnowledgeGraphSchema, NodeLabel, RelationshipType # type: ignore

logger = logging.getLogger(__name__)


class GraphDatabase(ABC):
    """
    Abstract base class for graph database implementations.
    
    This class defines the interface that all graph database implementations
    must adhere to.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the graph database.
        
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the graph database.
        
        Returns:
            bool: True if disconnection was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def initialize_schema(self, schema: KnowledgeGraphSchema) -> bool:
        """
        Initialize the database schema.
        
        Args:
            schema (KnowledgeGraphSchema): The schema to initialize.
            
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def create_node(self, label: NodeLabel, properties: Dict[str, Any]) -> str:
        """
        Create a node in the graph.
        
        Args:
            label (NodeLabel): The label of the node.
            properties (Dict[str, Any]): The properties of the node.
            
        Returns:
            str: The ID of the created node.
        """
        pass
    
    @abstractmethod
    def create_relationship(
        self, 
        source_id: str, 
        source_label: NodeLabel,
        target_id: str, 
        target_label: NodeLabel,
        relationship_type: RelationshipType,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Create a relationship between two nodes.
        
        Args:
            source_id (str): The ID of the source node.
            source_label (NodeLabel): The label of the source node.
            target_id (str): The ID of the target node.
            target_label (NodeLabel): The label of the target node.
            relationship_type (RelationshipType): The type of the relationship.
            properties (Optional[Dict[str, Any]]): The properties of the relationship.
            
        Returns:
            bool: True if creation was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_node(self, node_id: str, label: NodeLabel) -> Optional[Dict[str, Any]]:
        """
        Get a node by ID and label.
        
        Args:
            node_id (str): The ID of the node.
            label (NodeLabel): The label of the node.
            
        Returns:
            Optional[Dict[str, Any]]: The node properties, or None if not found.
        """
        pass
    
    @abstractmethod
    def get_nodes_by_property(
        self, 
        label: NodeLabel, 
        property_name: str, 
        property_value: Any
    ) -> List[Dict[str, Any]]:
        """
        Get nodes by a property value.
        
        Args:
            label (NodeLabel): The label of the nodes.
            property_name (str): The name of the property.
            property_value (Any): The value of the property.
            
        Returns:
            List[Dict[str, Any]]: The matching nodes.
        """
        pass
    
    @abstractmethod
    def update_node(
        self, 
        node_id: str, 
        label: NodeLabel, 
        properties: Dict[str, Any]
    ) -> bool:
        """
        Update a node's properties.
        
        Args:
            node_id (str): The ID of the node.
            label (NodeLabel): The label of the node.
            properties (Dict[str, Any]): The properties to update.
            
        Returns:
            bool: True if update was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def delete_node(self, node_id: str, label: NodeLabel) -> bool:
        """
        Delete a node.
        
        Args:
            node_id (str): The ID of the node.
            label (NodeLabel): The label of the node.
            
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        pass
    
    @abstractmethod
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom query.
        
        Args:
            query (str): The query to execute.
            parameters (Optional[Dict[str, Any]]): The query parameters.
            
        Returns:
            List[Dict[str, Any]]: The query results.
        """
        pass
    
    @abstractmethod
    def get_neighbors(
        self, 
        node_id: str, 
        label: NodeLabel, 
        relationship_type: Optional[RelationshipType] = None,
        direction: str = "OUTGOING",
        neighbor_label: Optional[NodeLabel] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the neighbors of a node.
        
        Args:
            node_id (str): The ID of the node.
            label (NodeLabel): The label of the node.
            relationship_type (Optional[RelationshipType]): The type of relationship to follow.
            direction (str): The direction of the relationship ("OUTGOING", "INCOMING", or "BOTH").
            neighbor_label (Optional[NodeLabel]): The label of the neighbor nodes.
            
        Returns:
            List[Dict[str, Any]]: The neighboring nodes.
        """
        pass
    
    @abstractmethod
    def get_shortest_path(
        self, 
        source_id: str, 
        source_label: NodeLabel,
        target_id: str, 
        target_label: NodeLabel,
        max_depth: int = 5
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get the shortest path between two nodes.
        
        Args:
            source_id (str): The ID of the source node.
            source_label (NodeLabel): The label of the source node.
            target_id (str): The ID of the target node.
            target_label (NodeLabel): The label of the target node.
            max_depth (int): The maximum path depth to search.
            
        Returns:
            Optional[List[Dict[str, Any]]]: The path as a list of nodes and relationships, or None if no path exists.
        """
        pass
    
    @abstractmethod
    def bulk_import_nodes(
        self, 
        label: NodeLabel, 
        nodes: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Import multiple nodes in bulk.
        
        Args:
            label (NodeLabel): The label of the nodes.
            nodes (List[Dict[str, Any]]): The nodes to import.
            
        Returns:
            List[str]: The IDs of the created nodes.
        """
        pass
    
    @abstractmethod
    def bulk_import_relationships(
        self, 
        relationships: List[Tuple[str, NodeLabel, str, NodeLabel, RelationshipType, Dict[str, Any]]]
    ) -> int:
        """
        Import multiple relationships in bulk.
        
        Args:
            relationships (List[Tuple]): The relationships to import as tuples of
                (source_id, source_label, target_id, target_label, relationship_type, properties).
            
        Returns:
            int: The number of relationships created.
        """
        pass
    
    @abstractmethod
    def get_node_count(self, label: Optional[NodeLabel] = None) -> int:
        """
        Get the count of nodes in the graph.
        
        Args:
            label (Optional[NodeLabel]): The label to count, or None for all nodes.
            
        Returns:
            int: The node count.
        """
        pass
    
    @abstractmethod
    def get_relationship_count(self, relationship_type: Optional[RelationshipType] = None) -> int:
        """
        Get the count of relationships in the graph.
        
        Args:
            relationship_type (Optional[RelationshipType]): The relationship type to count, or None for all relationships.
            
        Returns:
            int: The relationship count.
        """
        pass