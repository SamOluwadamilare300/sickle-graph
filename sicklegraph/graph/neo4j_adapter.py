"""
Neo4j database adapter for SickleGraph.

This module implements the GraphDatabase interface for the Neo4j graph database.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from neo4j import GraphDatabase as Neo4jDriver # type: ignore
    from neo4j.exceptions import Neo4jError # type: ignore
except ImportError:
    raise ImportError("Neo4j is not installed. Please install it with 'pip install neo4j'.")

from sicklegraph.graph.base import GraphDatabase # type: ignore
from sicklegraph.graph.schema import KnowledgeGraphSchema, NodeLabel, RelationshipType # type: ignore

logger = logging.getLogger(__name__)


class Neo4jAdapter(GraphDatabase):
    """
    Neo4j database adapter for SickleGraph.
    
    This class implements the GraphDatabase interface for the Neo4j graph database.
    """
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize the Neo4j adapter.
        
        Args:
            uri (str): The URI of the Neo4j database.
            username (str): The username for authentication.
            password (str): The password for authentication.
            database (str): The name of the database to use.
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
    
    def connect(self) -> bool:
        """
        Connect to the Neo4j database.
        
        Returns:
            bool: True if connection was successful, False otherwise.
        """
        try:
            self.driver = Neo4jDriver(self.uri, auth=(self.username, self.password))
            # Test the connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j database at {self.uri}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            return False
    
    def disconnect(self) -> bool:
        """
        Disconnect from the Neo4j database.
        
        Returns:
            bool: True if disconnection was successful, False otherwise.
        """
        try:
            if self.driver:
                self.driver.close()
                self.driver = None
                logger.info("Disconnected from Neo4j database")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to disconnect from Neo4j database: {e}")
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Execute schema creation statements
                for stmt in schema.generate_neo4j_schema():
                    logger.debug(f"Executing schema statement: {stmt}")
                    session.run(stmt)
                
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Create the node
                query = f"CREATE (n:{label} $props) RETURN n.id as id"
                result = session.run(query, props=properties)
                record = result.single()
                
                if not record:
                    raise ValueError("Failed to create node")
                
                node_id = record["id"]
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Create the relationship
                query = f"""
                MATCH (s:{source_label}), (t:{target_label})
                WHERE s.id = $source_id AND t.id = $target_id
                CREATE (s)-[r:{relationship_type} $props]->(t)
                RETURN r
                """
                
                result = session.run(
                    query, 
                    source_id=source_id, 
                    target_id=target_id, 
                    props=properties or {}
                )
                
                record = result.single()
                
                if not record:
                    logger.warning(f"Failed to create relationship: nodes not found")
                    return False
                
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Query the node
                query = f"MATCH (n:{label}) WHERE n.id = $id RETURN n"
                result = session.run(query, id=node_id)
                record = result.single()
                
                if not record:
                    return None
                
                node = record["n"]
                return dict(node)
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Query the nodes
                query = f"MATCH (n:{label}) WHERE n.{property_name} = $value RETURN n"
                result = session.run(query, value=property_value)
                
                nodes = []
                for record in result:
                    node = record["n"]
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Update the node
                query = f"MATCH (n:{label}) WHERE n.id = $id SET n += $props RETURN n"
                result = session.run(query, id=node_id, props=properties)
                record = result.single()
                
                if not record:
                    logger.warning(f"Failed to update node: node not found")
                    return False
                
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Delete the node
                query = f"MATCH (n:{label}) WHERE n.id = $id DETACH DELETE n"
                result = session.run(query, id=node_id)
                
                # Check if the node was deleted
                summary = result.consume()
                if summary.counters.nodes_deleted > 0:
                    logger.debug(f"Deleted {label} node with ID {node_id}")
                    return True
                else:
                    logger.warning(f"Failed to delete node: node not found")
                    return False
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Execute the query
                result = session.run(query, parameters or {})
                
                # Process the results
                results = []
                for record in result:
                    row = {}
                    for key, value in record.items():
                        if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
                            row[key] = dict(value)
                        else:
                            row[key] = value
                    results.append(row)
                
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Prepare the query based on direction and relationship type
                if direction == "OUTGOING":
                    if relationship_type:
                        if neighbor_label:
                            query = f"""
                            MATCH (n:{label})-[:{relationship_type}]->(m:{neighbor_label})
                            WHERE n.id = $id
                            RETURN m
                            """
                        else:
                            query = f"""
                            MATCH (n:{label})-[:{relationship_type}]->(m)
                            WHERE n.id = $id
                            RETURN m
                            """
                    else:
                        if neighbor_label:
                            query = f"""
                            MATCH (n:{label})-[]->(m:{neighbor_label})
                            WHERE n.id = $id
                            RETURN m
                            """
                        else:
                            query = f"""
                            MATCH (n:{label})-[]->(m)
                            WHERE n.id = $id
                            RETURN m
                            """
                elif direction == "INCOMING":
                    if relationship_type:
                        if neighbor_label:
                            query = f"""
                            MATCH (n:{label})<-[:{relationship_type}]-(m:{neighbor_label})
                            WHERE n.id = $id
                            RETURN m
                            """
                        else:
                            query = f"""
                            MATCH (n:{label})<-[:{relationship_type}]-(m)
                            WHERE n.id = $id
                            RETURN m
                            """
                    else:
                        if neighbor_label:
                            query = f"""
                            MATCH (n:{label})<-[]-(m:{neighbor_label})
                            WHERE n.id = $id
                            RETURN m
                            """
                        else:
                            query = f"""
                            MATCH (n:{label})<-[]-(m)
                            WHERE n.id = $id
                            RETURN m
                            """
                else:  # BOTH
                    if relationship_type:
                        if neighbor_label:
                            query = f"""
                            MATCH (n:{label})-[:{relationship_type}]-(m:{neighbor_label})
                            WHERE n.id = $id
                            RETURN m
                            """
                        else:
                            query = f"""
                            MATCH (n:{label})-[:{relationship_type}]-(m)
                            WHERE n.id = $id
                            RETURN m
                            """
                    else:
                        if neighbor_label:
                            query = f"""
                            MATCH (n:{label})-[]-(m:{neighbor_label})
                            WHERE n.id = $id
                            RETURN m
                            """
                        else:
                            query = f"""
                            MATCH (n:{label})-[]-(m)
                            WHERE n.id = $id
                            RETURN m
                            """
                
                # Execute the query
                result = session.run(query, id=node_id)
                
                # Process the results
                neighbors = []
                for record in result:
                    neighbor = record["m"]
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Use Neo4j's built-in shortest path algorithm
                query = f"""
                MATCH (source:{source_label}), (target:{target_label})
                WHERE source.id = $source_id AND target.id = $target_id
                MATCH path = shortestPath((source)-[*1..{max_depth}]-(target))
                RETURN path
                """
                
                result = session.run(query, source_id=source_id, target_id=target_id)
                record = result.single()
                
                if not record:
                    return None
                
                # Extract the path
                path = record["path"]
                
                # Convert to the expected format
                result = []
                for i, entity in enumerate(path):
                    if i % 2 == 0:  # Node
                        result.append({
                            "type": "node",
                            "data": dict(entity)
                        })
                    else:  # Relationship
                        result.append({
                            "type": "relationship",
                            "data": {
                                "type": entity.type,
                                "properties": dict(entity)
                            }
                        })
                
                return result
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Use Neo4j's UNWIND for bulk import
                query = f"""
                UNWIND $nodes as node
                CREATE (n:{label})
                SET n = node
                RETURN n.id as id
                """
                
                result = session.run(query, nodes=nodes)
                
                # Collect the IDs
                node_ids = [record["id"] for record in result]
                
                logger.info(f"Bulk imported {len(node_ids)} {label} nodes")
                return node_ids
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Convert the relationships to a format suitable for Neo4j
                rel_data = []
                for source_id, source_label, target_id, target_label, rel_type, props in relationships:
                    rel_data.append({
                        "source_id": source_id,
                        "source_label": source_label,
                        "target_id": target_id,
                        "target_label": target_label,
                        "type": rel_type,
                        "props": props or {}
                    })
                
                # Use Neo4j's UNWIND for bulk import
                query = """
                UNWIND $relationships as rel
                MATCH (source) WHERE source.id = rel.source_id AND any(label IN labels(source) WHERE label = rel.source_label)
                MATCH (target) WHERE target.id = rel.target_id AND any(label IN labels(target) WHERE label = rel.target_label)
                CALL apoc.create.relationship(source, rel.type, rel.props, target) YIELD rel as created
                RETURN count(created) as count
                """
                
                result = session.run(query, relationships=rel_data)
                record = result.single()
                
                if not record:
                    return 0
                
                count = record["count"]
                logger.info(f"Bulk imported {count} relationships")
                return count
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Query the count
                if label:
                    query = f"MATCH (n:{label}) RETURN count(n) as count"
                else:
                    query = "MATCH (n) RETURN count(n) as count"
                
                result = session.run(query)
                record = result.single()
                
                if not record:
                    return 0
                
                return record["count"]
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
            if not self.driver:
                raise ValueError("Not connected to database")
            
            with self.driver.session(database=self.database) as session:
                # Query the count
                if relationship_type:
                    query = f"MATCH ()-[r:{relationship_type}]->() RETURN count(r) as count"
                else:
                    query = "MATCH ()-[r]->() RETURN count(r) as count"
                
                result = session.run(query)
                record = result.single()
                
                if not record:
                    return 0
                
                return record["count"]
        except Exception as e:
            logger.error(f"Failed to get relationship count: {e}")
            return 0