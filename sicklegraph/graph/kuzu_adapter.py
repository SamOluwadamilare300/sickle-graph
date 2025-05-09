"""
Kùzu database adapter for SickleGraph.

This module implements the GraphDatabase interface for the Kùzu graph database.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import kuzu # type: ignore
except ImportError:
    raise ImportError("Kùzu is not installed. Please install it with 'pip install kuzu'.")

from sicklegraph.graph.base import GraphDatabase # type: ignore
from sicklegraph.graph.schema import KnowledgeGraphSchema, NodeLabel, RelationshipType # type: ignore

logger = logging.getLogger(__name__)


class KuzuAdapter(GraphDatabase):
    """
    Kùzu database adapter for SickleGraph.
    
    This class implements the GraphDatabase interface for the Kùzu graph database.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the Kùzu adapter.
        
        Args:
            db_path (str): The path to the Kùzu database.
        """
        self.db_path = db_path
        self.db = None
        self.conn = None
    
    def connect(self) -> bool:
        """
        Connect to the Kùzu database.
        
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        try:
            # Create the database directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Connect to the database
            self.db = kuzu.Database(self.db_path)
            self.conn = kuzu.Connection(self.db)
            logger.info(f"Connected to Kùzu database at {self.db_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Kùzu database: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the Kùzu database.
        
        Returns:
            bool: True if disconnection was successful, False otherwise.
        """
        try:
            self.conn = None
            self.db = None
            logger.info("Disconnected from Kùzu database")
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from Kùzu database: {e}")
            return False
    
    def initialize_schema(self, schema: KnowledgeGraphSchema) -> bool:
        """
        Initialize the database schema.
        
        Args:
            schema (KnowledgeGraphSchema): The schema to initialize.
            
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Execute schema creation statements
            for stmt in schema.generate_kuzu_schema():
                logger.debug(f"Executing schema statement: {stmt}")
                self.conn.execute(stmt)
            
            logger.info("Schema initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            return False
    
    def create_node(self, label: NodeLabel, properties: Dict[str, Any]) -> str:
        """
        Create a node in the graph.
        
        Args:
            label (NodeLabel): The label of the node.
            properties (Dict[str, Any]): The properties of the node.
            
        Returns:
            str: The ID of the created node.
        """
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Prepare property names and values
            prop_names = ", ".join(properties.keys())
            prop_placeholders = ", ".join([f"${i}" for i in range(len(properties))])
            prop_values = list(properties.values())
            
            # Create the node
            query = f"CREATE ({label}:{{{prop_names}}}) VALUES ({prop_placeholders})"
            self.conn.execute(query, prop_values)
            
            # Return the ID
            node_id = properties.get("id")
            if not node_id:
                raise ValueError("Node must have an 'id' property")
            
            logger.debug(f"Created {label} node with ID {node_id}")
            return node_id
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            raise
    
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
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Prepare the query
            if properties:
                prop_names = ", ".join(properties.keys())
                prop_placeholders = ", ".join([f"${i+2}" for i in range(len(properties))])
                prop_values = list(properties.values())
                
                query = f"""
                MATCH (s:{source_label}), (t:{target_label})
                WHERE s.id = $0 AND t.id = $1
                CREATE (s)-[:{relationship_type} {{{prop_names}}}]->
                (t) VALUES ({prop_placeholders})
                """
                params = [source_id, target_id] + prop_values
            else:
                query = f"""
                MATCH (s:{source_label}), (t:{target_label})
                WHERE s.id = $0 AND t.id = $1
                CREATE (s)-[:{relationship_type}]->(t)
                """
                params = [source_id, target_id]
            
            # Execute the query
            self.conn.execute(query, params)
            
            logger.debug(f"Created {relationship_type} relationship from {source_label}:{source_id} to {target_label}:{target_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False
    
    def get_node(self, node_id: str, label: NodeLabel) -> Optional[Dict[str, Any]]:
        """
        Get a node by ID and label.
        
        Args:
            node_id (str): The ID of the node.
            label (NodeLabel): The label of the node.
            
        Returns:
            Optional[Dict[str, Any]]: The node properties, or None if not found.
        """
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Query the node
            query = f"MATCH (n:{label}) WHERE n.id = $0 RETURN n"
            result = self.conn.execute(query, [node_id])
            
            # Process the result
            if result.has_next():
                row = result.get_next()
                node = row[0]
                return dict(node)
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get node: {e}")
            return None
    
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
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Query the nodes
            query = f"MATCH (n:{label}) WHERE n.{property_name} = $0 RETURN n"
            result = self.conn.execute(query, [property_value])
            
            # Process the results
            nodes = []
            while result.has_next():
                row = result.get_next()
                node = row[0]
                nodes.append(dict(node))
            
            return nodes
        except Exception as e:
            logger.error(f"Failed to get nodes by property: {e}")
            return []
    
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
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Prepare the SET clause
            set_clauses = []
            params = [node_id]
            
            for i, (key, value) in enumerate(properties.items()):
                set_clauses.append(f"n.{key} = ${i+1}")
                params.append(value)
            
            set_clause = ", ".join(set_clauses)
            
            # Update the node
            query = f"MATCH (n:{label}) WHERE n.id = $0 SET {set_clause}"
            self.conn.execute(query, params)
            
            logger.debug(f"Updated {label} node with ID {node_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update node: {e}")
            return False
    
    def delete_node(self, node_id: str, label: NodeLabel) -> bool:
        """
        Delete a node.
        
        Args:
            node_id (str): The ID of the node.
            label (NodeLabel): The label of the node.
            
        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Delete the node
            query = f"MATCH (n:{label}) WHERE n.id = $0 DELETE n"
            self.conn.execute(query, [node_id])
            
            logger.debug(f"Deleted {label} node with ID {node_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete node: {e}")
            return False
    
    def execute_query(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom query.
        
        Args:
            query (str): The query to execute.
            parameters (Optional[Dict[str, Any]]): The query parameters.
            
        Returns:
            List[Dict[str, Any]]: The query results.
        """
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Execute the query
            params = list(parameters.values()) if parameters else []
            result = self.conn.execute(query, params)
            
            # Process the results
            results = []
            column_names = result.get_column_names()
            
            while result.has_next():
                row = result.get_next()
                row_dict = {}
                
                for i, col_name in enumerate(column_names):
                    value = row[i]
                    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
                        row_dict[col_name] = dict(value)
                    else:
                        row_dict[col_name] = value
                
                results.append(row_dict)
            
            return results
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            return []
    
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
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Prepare the query based on direction and relationship type
            if direction == "OUTGOING":
                if relationship_type:
                    if neighbor_label:
                        query = f"""
                        MATCH (n:{label})-[:{relationship_type}]->(m:{neighbor_label})
                        WHERE n.id = $0
                        RETURN m
                        """
                    else:
                        query = f"""
                        MATCH (n:{label})-[:{relationship_type}]->(m)
                        WHERE n.id = $0
                        RETURN m
                        """
                else:
                    if neighbor_label:
                        query = f"""
                        MATCH (n:{label})-[]->(m:{neighbor_label})
                        WHERE n.id = $0
                        RETURN m
                        """
                    else:
                        query = f"""
                        MATCH (n:{label})-[]->(m)
                        WHERE n.id = $0
                        RETURN m
                        """
            elif direction == "INCOMING":
                if relationship_type:
                    if neighbor_label:
                        query = f"""
                        MATCH (n:{label})<-[:{relationship_type}]-(m:{neighbor_label})
                        WHERE n.id = $0
                        RETURN m
                        """
                    else:
                        query = f"""
                        MATCH (n:{label})<-[:{relationship_type}]-(m)
                        WHERE n.id = $0
                        RETURN m
                        """
                else:
                    if neighbor_label:
                        query = f"""
                        MATCH (n:{label})<-[]-(m:{neighbor_label})
                        WHERE n.id = $0
                        RETURN m
                        """
                    else:
                        query = f"""
                        MATCH (n:{label})<-[]-(m)
                        WHERE n.id = $0
                        RETURN m
                        """
            else:  # BOTH
                if relationship_type:
                    if neighbor_label:
                        query = f"""
                        MATCH (n:{label})-[:{relationship_type}]-(m:{neighbor_label})
                        WHERE n.id = $0
                        RETURN m
                        """
                    else:
                        query = f"""
                        MATCH (n:{label})-[:{relationship_type}]-(m)
                        WHERE n.id = $0
                        RETURN m
                        """
                else:
                    if neighbor_label:
                        query = f"""
                        MATCH (n:{label})-[]-(m:{neighbor_label})
                        WHERE n.id = $0
                        RETURN m
                        """
                    else:
                        query = f"""
                        MATCH (n:{label})-[]-(m)
                        WHERE n.id = $0
                        RETURN m
                        """
            
            # Execute the query
            result = self.conn.execute(query, [node_id])
            
            # Process the results
            neighbors = []
            while result.has_next():
                row = result.get_next()
                neighbor = row[0]
                neighbors.append(dict(neighbor))
            
            return neighbors
        except Exception as e:
            logger.error(f"Failed to get neighbors: {e}")
            return []
    
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
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Kùzu doesn't have built-in shortest path algorithms yet, so we'll implement a BFS
            # This is a simplified implementation and may not be efficient for large graphs
            
            # Start with the source node
            visited = set([source_id])
            queue = [[(source_id, source_label, None, None)]]
            
            # BFS
            while queue and len(queue[0]) <= max_depth:
                path = queue.pop(0)
                node_id, node_label, _, _ = path[-1]
                
                # Check if we've reached the target
                if node_id == target_id:
                    # Convert path to the expected format
                    result = []
                    for i, (nid, nlabel, rtype, rprops) in enumerate(path):
                        node = self.get_node(nid, nlabel)
                        if node:
                            result.append({"type": "node", "data": node})
                        
                        if i < len(path) - 1 and rtype:
                            result.append({
                                "type": "relationship",
                                "data": {
                                    "type": rtype,
                                    "properties": rprops or {}
                                }
                            })
                    
                    return result
                
                # Get neighbors
                neighbors = self.get_neighbors(node_id, node_label, direction="OUTGOING")
                
                for neighbor in neighbors:
                    neighbor_id = neighbor["id"]
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        
                        # Get the relationship type and properties
                        rel_query = f"""
                        MATCH (n:{node_label})-[r]->(m)
                        WHERE n.id = $0 AND m.id = $1
                        RETURN type(r) as type, r as properties
                        """
                        rel_result = self.conn.execute(rel_query, [node_id, neighbor_id])
                        
                        rel_type = None
                        rel_props = None
                        
                        if rel_result.has_next():
                            rel_row = rel_result.get_next()
                            rel_type = rel_row[0]
                            rel_props = dict(rel_row[1])
                        
                        # Get the neighbor label
                        label_query = "MATCH (n) WHERE n.id = $0 RETURN labels(n)[0] as label"
                        label_result = self.conn.execute(label_query, [neighbor_id])
                        
                        neighbor_label = None
                        if label_result.has_next():
                            label_row = label_result.get_next()
                            neighbor_label = label_row[0]
                        
                        # Add to queue
                        new_path = path.copy()
                        new_path.append((neighbor_id, neighbor_label, rel_type, rel_props))
                        queue.append(new_path)
            
            # No path found
            return None
        except Exception as e:
            logger.error(f"Failed to get shortest path: {e}")
            return None
    
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
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Kùzu doesn't have a specific bulk import API yet, so we'll use a transaction
            # This is a simplified implementation and may not be efficient for very large imports
            
            node_ids = []
            
            # Start a transaction
            self.conn.execute("BEGIN TRANSACTION")
            
            try:
                for node in nodes:
                    # Ensure the node has an ID
                    if "id" not in node:
                        raise ValueError("Each node must have an 'id' property")
                    
                    # Create the node
                    node_id = self.create_node(label, node)
                    node_ids.append(node_id)
                
                # Commit the transaction
                self.conn.execute("COMMIT")
                
                logger.info(f"Bulk imported {len(node_ids)} {label} nodes")
                return node_ids
            except Exception as e:
                # Rollback on error
                self.conn.execute("ROLLBACK")
                raise e
        except Exception as e:
            logger.error(f"Failed to bulk import nodes: {e}")
            return []
    
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
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Kùzu doesn't have a specific bulk import API yet, so we'll use a transaction
            # This is a simplified implementation and may not be efficient for very large imports
            
            count = 0
            
            # Start a transaction
            self.conn.execute("BEGIN TRANSACTION")
            
            try:
                for source_id, source_label, target_id, target_label, rel_type, props in relationships:
                    # Create the relationship
                    success = self.create_relationship(
                        source_id, source_label, target_id, target_label, rel_type, props
                    )
                    
                    if success:
                        count += 1
                
                # Commit the transaction
                self.conn.execute("COMMIT")
                
                logger.info(f"Bulk imported {count} relationships")
                return count
            except Exception as e:
                # Rollback on error
                self.conn.execute("ROLLBACK")
                raise e
        except Exception as e:
            logger.error(f"Failed to bulk import relationships: {e}")
            return 0
    
    def get_node_count(self, label: Optional[NodeLabel] = None) -> int:
        """
        Get the count of nodes in the graph.
        
        Args:
            label (Optional[NodeLabel]): The label to count, or None for all nodes.
            
        Returns:
            int: The node count.
        """
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Query the count
            if label:
                query = f"MATCH (n:{label}) RETURN count(n) as count"
            else:
                query = "MATCH (n) RETURN count(n) as count"
            
            result = self.conn.execute(query)
            
            # Get the count
            if result.has_next():
                row = result.get_next()
                return row[0]
            else:
                return 0
        except Exception as e:
            logger.error(f"Failed to get node count: {e}")
            return 0
    
    def get_relationship_count(self, relationship_type: Optional[RelationshipType] = None) -> int:
        """
        Get the count of relationships in the graph.
        
        Args:
            relationship_type (Optional[RelationshipType]): The relationship type to count, or None for all relationships.
            
        Returns:
            int: The relationship count.
        """
        try:
            if not self.conn:
                raise ValueError("Not connected to database")
            
            # Query the count
            if relationship_type:
                query = f"MATCH ()-[r:{relationship_type}]->() RETURN count(r) as count"
            else:
                query = "MATCH ()-[r]->() RETURN count(r) as count"
            
            result = self.conn.execute(query)
            
            # Get the count
            if result.has_next():
                row = result.get_next()
                return row[0]
            else:
                return 0
        except Exception as e:
            logger.error(f"Failed to get relationship count: {e}")
            return 0