"""
Main data pipeline for SickleGraph.

This module implements the main data pipeline for SickleGraph.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from sicklegraph.pipeline.base import DataPipeline, DataSource
from sicklegraph.graph import GraphDatabase, NodeLabel, RelationshipType

logger = logging.getLogger(__name__)


class SickleGraphPipeline(DataPipeline):
    """
    Main data pipeline for SickleGraph.
    
    This class implements the main data pipeline for SickleGraph.
    """
    
    def run(self, source_names: Optional[List[str]] = None, **kwargs) -> bool:
        """
        Run the data pipeline.
        
        Args:
            source_names (Optional[List[str]]): The names of the data sources to run, or None for all.
            **kwargs: Additional arguments for the data pipeline.
            
        Returns:
            bool: True if the pipeline ran successfully, False otherwise.
        """
        try:
            # Determine which sources to run
            if source_names is None:
                source_names = self.get_sources()
            
            if not source_names:
                logger.warning("No data sources specified")
                return False
            
            logger.info(f"Running data pipeline with sources: {', '.join(source_names)}")
            
            # Run each source
            for source_name in source_names:
                source = self.get_source(source_name)
                
                if not source:
                    logger.warning(f"Data source not found: {source_name}")
                    continue
                
                logger.info(f"Processing data source: {source_name}")
                
                # Fetch data
                data = source.fetch_data(**kwargs)
                
                if not data:
                    logger.warning(f"No data fetched from source: {source_name}")
                    continue
                
                # Transform data
                transformed_data = source.transform_data(data)
                
                if not transformed_data:
                    logger.warning(f"No data transformed from source: {source_name}")
                    continue
                
                # Load data
                success = self.load_data(transformed_data, source_name)
                
                if not success:
                    logger.error(f"Failed to load data from source: {source_name}")
                    return False
            
            logger.info("Data pipeline completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to run data pipeline: {e}")
            return False
    
    def load_data(self, data: List[Dict[str, Any]], source_name: str) -> bool:
        """
        Load data into the graph.
        
        Args:
            data (List[Dict[str, Any]]): The data to load.
            source_name (str): The name of the data source.
            
        Returns:
            bool: True if the data was loaded successfully, False otherwise.
        """
        try:
            if not self.graph:
                raise ValueError("Graph database not initialized")
            
            logger.info(f"Loading {len(data)} entities from source: {source_name}")
            
            # Separate nodes and relationships
            nodes = []
            relationships = []
            
            for entity in data:
                entity_type = entity.get("type")
                
                if entity_type == "node":
                    nodes.append(entity)
                elif entity_type == "relationship":
                    relationships.append(entity)
                else:
                    logger.warning(f"Unknown entity type: {entity_type}")
            
            logger.info(f"Found {len(nodes)} nodes and {len(relationships)} relationships")
            
            # Load nodes first
            node_ids = {}
            
            for node in nodes:
                label = node.get("label")
                properties = node.get("properties", {})
                
                if not label or not properties:
                    logger.warning(f"Invalid node: {node}")
                    continue
                
                # Check if the node already exists
                node_id = properties.get("id")
                if not node_id:
                    logger.warning(f"Node has no ID: {node}")
                    continue
                
                existing_node = self.graph.get_node(node_id, label)
                
                if existing_node:
                    # Update the existing node
                    self.graph.update_node(node_id, label, properties)
                    node_ids[node_id] = label
                else:
                    # Create a new node
                    try:
                        created_id = self.graph.create_node(label, properties)
                        node_ids[created_id] = label
                    except Exception as e:
                        logger.error(f"Failed to create node: {e}")
            
            logger.info(f"Loaded {len(node_ids)} nodes")
            
            # Load relationships
            rel_count = 0
            
            for rel in relationships:
                source_id = rel.get("source_id")
                source_label = rel.get("source_label")
                target_id = rel.get("target_id")
                target_label = rel.get("target_label")
                rel_type = rel.get("relationship_type")
                properties = rel.get("properties", {})
                
                if not source_id or not source_label or not target_id or not target_label or not rel_type:
                    logger.warning(f"Invalid relationship: {rel}")
                    continue
                
                # Check if both nodes exist
                if source_id not in node_ids and self.graph.get_node(source_id, source_label) is None:
                    logger.warning(f"Source node not found: {source_id}")
                    continue
                
                if target_id not in node_ids and self.graph.get_node(target_id, target_label) is None:
                    logger.warning(f"Target node not found: {target_id}")
                    continue
                
                # Create the relationship
                success = self.graph.create_relationship(
                    source_id, source_label, target_id, target_label, rel_type, properties
                )
                
                if success:
                    rel_count += 1
            
            logger.info(f"Loaded {rel_count} relationships")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False