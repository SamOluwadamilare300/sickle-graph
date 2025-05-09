"""
PubMed data source for SickleGraph.

This module implements a data source for PubMed.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import xml.etree.ElementTree as ET
from datetime import datetime

from Bio import Entrez # type: ignore

from sicklegraph.pipeline.base import DataSource # type: ignore
from sicklegraph.graph import NodeLabel, RelationshipType # type: ignore

logger = logging.getLogger(__name__)


class PubMedSource(DataSource):
    """
    PubMed data source for SickleGraph.
    
    This class implements a data source for PubMed.
    """
    
    def __init__(self, email: str, api_key: Optional[str] = None, max_results: int = 1000):
        """
        Initialize the PubMed data source.
        
        Args:
            email (str): The email to use for Entrez.
            api_key (Optional[str]): The API key to use for Entrez.
            max_results (int): The maximum number of results to fetch.
        """
        self.email = email
        self.api_key = api_key
        self.max_results = max_results
        
        # Set up Entrez
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
    
    @property
    def name(self) -> str:
        """
        Get the name of the data source.
        
        Returns:
            str: The name of the data source.
        """
        return "pubmed"
    
    def fetch_data(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Fetch data from PubMed.
        
        Args:
            **kwargs: Additional arguments for the data source.
                query (str): The search query to use.
                max_results (int): The maximum number of results to fetch.
                
        Returns:
            List[Dict[str, Any]]: The fetched data.
        """
        query = kwargs.get("query", "sickle cell disease gene therapy")
        max_results = kwargs.get("max_results", self.max_results)
        
        logger.info(f"Fetching PubMed data for query: {query}")
        
        try:
            # Search for the query
            search_handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            # Get the IDs
            id_list = search_results["IdList"]
            logger.info(f"Found {len(id_list)} PubMed articles")
            
            if not id_list:
                return []
            
            # Fetch the articles
            articles = []
            
            # Process in batches to avoid overloading the API
            batch_size = 100
            for i in range(0, len(id_list), batch_size):
                batch_ids = id_list[i:i+batch_size]
                
                logger.debug(f"Fetching batch of {len(batch_ids)} articles")
                
                fetch_handle = Entrez.efetch(
                    db="pubmed",
                    id=batch_ids,
                    retmode="xml"
                )
                
                # Parse the XML
                tree = ET.parse(fetch_handle)
                root = tree.getroot()
                
                # Process each article
                for article_elem in root.findall(".//PubmedArticle"):
                    try:
                        # Extract article data
                        article = self._extract_article_data(article_elem)
                        if article:
                            articles.append(article)
                    except Exception as e:
                        logger.error(f"Failed to extract article data: {e}")
                
                fetch_handle.close()
                
                # Be nice to the API
                time.sleep(1)
            
            logger.info(f"Fetched {len(articles)} PubMed articles")
            return articles
        
        except Exception as e:
            logger.error(f"Failed to fetch PubMed data: {e}")
            return []
    
    def transform_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Transform the fetched PubMed data.
        
        Args:
            data (List[Dict[str, Any]]): The data to transform.
            
        Returns:
            List[Dict[str, Any]]: The transformed data.
        """
        logger.info(f"Transforming {len(data)} PubMed articles")
        
        transformed_data = []
        
        for article in data:
            try:
                # Create a paper node
                paper = {
                    "type": "node",
                    "label": NodeLabel.PAPER,
                    "properties": {
                        "id": f"pubmed_{article['pmid']}",
                        "title": article["title"],
                        "abstract": article.get("abstract", ""),
                        "doi": article.get("doi", ""),
                        "publication_date": article.get("publication_date", ""),
                        "journal": article.get("journal", ""),
                        "citation_count": article.get("citation_count", 0),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/"
                    }
                }
                transformed_data.append(paper)
                
                # Create researcher nodes and relationships
                for author in article.get("authors", []):
                    researcher_id = self._generate_researcher_id(author)
                    
                    researcher = {
                        "type": "node",
                        "label": NodeLabel.RESEARCHER,
                        "properties": {
                            "id": researcher_id,
                            "name": author,
                            "email": "",
                            "orcid": "",
                            "h_index": 0,
                            "publication_count": 1
                        }
                    }
                    transformed_data.append(researcher)
                    
                    # Create authored_by relationship
                    authored_by = {
                        "type": "relationship",
                        "source_id": f"pubmed_{article['pmid']}",
                        "source_label": NodeLabel.PAPER,
                        "target_id": researcher_id,
                        "target_label": NodeLabel.RESEARCHER,
                        "relationship_type": RelationshipType.AUTHORED_BY,
                        "properties": {
                            "author_position": "unknown",
                            "corresponding": False
                        }
                    }
                    transformed_data.append(authored_by)
                
                # Extract and create gene nodes and relationships
                for gene in self._extract_genes(article.get("abstract", "")):
                    gene_id = f"gene_{gene.lower()}"
                    
                    gene_node = {
                        "type": "node",
                        "label": NodeLabel.GENE,
                        "properties": {
                            "id": gene_id,
                            "symbol": gene,
                            "name": gene,
                            "chromosome": "",
                            "start_position": 0,
                            "end_position": 0,
                            "strand": "",
                            "sequence": "",
                            "description": f"Gene mentioned in PubMed article {article['pmid']}"
                        }
                    }
                    transformed_data.append(gene_node)
                    
                    # Create relationship between paper and gene
                    mentions = {
                        "type": "relationship",
                        "source_id": f"pubmed_{article['pmid']}",
                        "source_label": NodeLabel.PAPER,
                        "target_id": gene_id,
                        "target_label": NodeLabel.GENE,
                        "relationship_type": RelationshipType.TARGETS,
                        "properties": {
                            "mechanism": "mentioned",
                            "efficacy": 0.0,
                            "specificity": 0.0
                        }
                    }
                    transformed_data.append(mentions)
            
            except Exception as e:
                logger.error(f"Failed to transform article: {e}")
        
        logger.info(f"Transformed data into {len(transformed_data)} entities")
        return transformed_data
    
    def _extract_article_data(self, article_elem: ET.Element) -> Dict[str, Any]:
        """
        Extract data from a PubMed article XML element.
        
        Args:
            article_elem (ET.Element): The article XML element.
            
        Returns:
            Dict[str, Any]: The extracted article data.
        """
        try:
            # Extract PMID
            pmid_elem = article_elem.find(".//PMID")
            if pmid_elem is None:
                return None
            
            pmid = pmid_elem.text
            
            # Extract title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            # Extract abstract
            abstract_elems = article_elem.findall(".//AbstractText")
            abstract = " ".join([elem.text for elem in abstract_elems if elem.text])
            
            # Extract DOI
            doi_elem = article_elem.find(".//ArticleId[@IdType='doi']")
            doi = doi_elem.text if doi_elem is not None else ""
            
            # Extract journal
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract publication date
            pub_date = ""
            year_elem = article_elem.find(".//PubDate/Year")
            month_elem = article_elem.find(".//PubDate/Month")
            day_elem = article_elem.find(".//PubDate/Day")
            
            if year_elem is not None:
                pub_date = year_elem.text
                if month_elem is not None:
                    pub_date = f"{pub_date}-{month_elem.text}"
                    if day_elem is not None:
                        pub_date = f"{pub_date}-{day_elem.text}"
            
            # Extract authors
            authors = []
            author_elems = article_elem.findall(".//Author")
            
            for author_elem in author_elems:
                last_name_elem = author_elem.find("LastName")
                fore_name_elem = author_elem.find("ForeName")
                
                if last_name_elem is not None:
                    last_name = last_name_elem.text
                    fore_name = fore_name_elem.text if fore_name_elem is not None else ""
                    
                    if fore_name:
                        authors.append(f"{fore_name} {last_name}")
                    else:
                        authors.append(last_name)
            
            # Create the article data
            article = {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "doi": doi,
                "journal": journal,
                "publication_date": pub_date,
                "authors": authors,
                "citation_count": 0  # We don't have this information from the API
            }
            
            return article
        
        except Exception as e:
            logger.error(f"Failed to extract article data: {e}")
            return None
    
    def _generate_researcher_id(self, name: str) -> str:
        """
        Generate a researcher ID from a name.
        
        Args:
            name (str): The researcher's name.
            
        Returns:
            str: The generated ID.
        """
        # Remove non-alphanumeric characters and convert to lowercase
        clean_name = re.sub(r'[^a-zA-Z0-9]', '', name).lower()
        
        # Truncate if too long
        if len(clean_name) > 20:
            clean_name = clean_name[:20]
        
        return f"researcher_{clean_name}"
    
    def _extract_genes(self, text: str) -> List[str]:
        """
        Extract gene symbols from text.
        
        Args:
            text (str): The text to extract from.
            
        Returns:
            List[str]: The extracted gene symbols.
        """
        # This is a simplified implementation
        # In a real-world scenario, you would use a more sophisticated approach,
        # such as named entity recognition or a curated list of gene symbols
        
        # Common SCD-related genes
        scd_genes = ["HBB", "HBA1", "HBA2", "BCL11A", "HMOX1", "NOS3", "VCAM1"]
        
        found_genes = []
        
        for gene in scd_genes:
            if re.search(r'\b' + re.escape(gene) + r'\b', text):
                found_genes.append(gene)
        
        return found_genes