"""
Base data pipeline for SickleGraph.

This module defines the base data pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from sicklegraph.graph import GraphDatabase # type: ignore

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """
    Base data source for SickleGraph.
    
    This class defines the interface for data sources.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the data source.
        
        Returns:
            str: The name of the data source.
        """
        pass
    
    @abstractmethod
    def extract(self, query: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Extract data from the source.
        
        Args:
            query (str): The query to use for extraction.
            max_results (int): The maximum number of results to extract.
            
        Returns:
            List[Dict[str, Any]]: The extracted data.
        """
        pass
    
    @abstractmethod
    def transform(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform the extracted data.
        
        Args:
            data (List[Dict[str, Any]]): The extracted data.
            
        Returns:
            List[Dict[str, Any]]: The transformed data.
        """
        pass
    
    @abstractmethod
    def load(self, data: List[Dict[str, Any]], graph: GraphDatabase) -> bool:
        """
        Load the transformed data into the graph database.
        
        Args:
            data (List[Dict[str, Any]]): The transformed data.
            graph (GraphDatabase): The graph database to load into.
            
        Returns:
            bool: True if the data was loaded successfully, False otherwise.
        """
        pass


class SickleGraphPipeline:
    """
    Data pipeline for SickleGraph.
    
    This class implements the data pipeline for SickleGraph.
    """
    
    def __init__(self, graph: GraphDatabase):
        """
        Initialize the data pipeline.
        
        Args:
            graph (GraphDatabase): The graph database to load into.
        """
        self.graph = graph
        self.sources: Dict[str, DataSource] = {}
    
    def register_source(self, source: DataSource) -> None:
        """
        Register a data source.
        
        Args:
            source (DataSource): The data source to register.
        """
        self.sources[source.name] = source
        logger.info(f"Registered data source: {source.name}")
    
    def run(self, sources: Optional[List[str]] = None, query: str = "sickle cell disease gene therapy") -> bool:
        """
        Run the data pipeline.
        
        Args:
            sources (Optional[List[str]]): The data sources to run. If None, all registered sources are run.
            query (str): The query to use for extraction.
            
        Returns:
            bool: True if the pipeline was run successfully, False otherwise.
        """
        try:
            # Determine which sources to run
            if sources is None:
                sources_to_run = list(self.sources.keys())
            else:
                sources_to_run = sources
            
            logger.info(f"Running data pipeline with query: {query}")
            logger.info(f"Sources to run: {sources_to_run}")
            
            # Run each source
            for source_name in sources_to_run:
                if source_name not in self.sources:
                    logger.warning(f"Source not found: {source_name}")
                    continue
                
                source = self.sources[source_name]
                
                logger.info(f"Running source: {source_name}")
                
                # Extract
                logger.info(f"Extracting data from {source_name}...")
                extracted_data = source.extract(query)
                
                if not extracted_data:
                    logger.warning(f"No data extracted from {source_name}")
                    continue
                
                logger.info(f"Extracted {len(extracted_data)} records from {source_name}")
                
                # Transform
                logger.info(f"Transforming data from {source_name}...")
                transformed_data = source.transform(extracted_data)
                
                if not transformed_data:
                    logger.warning(f"No data transformed from {source_name}")
                    continue
                
                logger.info(f"Transformed {len(transformed_data)} records from {source_name}")
                
                # Load
                logger.info(f"Loading data from {source_name}...")
                if not source.load(transformed_data, self.graph):
                    logger.error(f"Failed to load data from {source_name}")
                    continue
                
                logger.info(f"Loaded data from {source_name}")
            
            logger.info("Data pipeline completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to run data pipeline: {e}")
            return False