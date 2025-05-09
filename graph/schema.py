"""
Knowledge graph schema definition for SickleGraph.

This module defines the schema for the SickleGraph knowledge graph, including
node types, relationship types, and properties.
"""

from enum import Enum
from typing import Dict, List, Optional, Union
import json


class NodeLabel(str, Enum):
    """Enum for node labels in the knowledge graph."""
    GENE = "Gene"
    MUTATION = "Mutation"
    TREATMENT = "Treatment"
    PAPER = "Paper"
    CLINICAL_TRIAL = "ClinicalTrial"
    RESEARCHER = "Researcher"
    INSTITUTION = "Institution"
    COUNTRY = "Country"
    CONTINENT = "Continent"
    PROTEIN = "Protein"
    PATHWAY = "Pathway"
    DISEASE = "Disease"
    SYMPTOM = "Symptom"
    DRUG = "Drug"


class RelationshipType(str, Enum):
    """Enum for relationship types in the knowledge graph."""
    HAS_MUTATION = "HAS_MUTATION"
    TARGETS = "TARGETS"
    TREATS = "TREATS"
    AUTHORED_BY = "AUTHORED_BY"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    LOCATED_IN = "LOCATED_IN"
    PART_OF = "PART_OF"
    CITES = "CITES"
    CONDUCTED_BY = "CONDUCTED_BY"
    CONDUCTED_IN = "CONDUCTED_IN"
    ENCODES = "ENCODES"
    PARTICIPATES_IN = "PARTICIPATES_IN"
    REGULATES = "REGULATES"
    CAUSES = "CAUSES"
    MANIFESTS_AS = "MANIFESTS_AS"
    INTERACTS_WITH = "INTERACTS_WITH"


class KnowledgeGraphSchema:
    """
    Defines the schema for the SickleGraph knowledge graph.
    
    This class provides methods to generate schema creation statements
    for both Kùzu and Neo4j databases.
    """
    
    def __init__(self):
        """Initialize the schema with node and relationship definitions."""
        self.node_schemas = {
            NodeLabel.GENE: {
                "properties": {
                    "id": "STRING",
                    "symbol": "STRING",
                    "name": "STRING",
                    "chromosome": "STRING",
                    "start_position": "INT64",
                    "end_position": "INT64",
                    "strand": "STRING",
                    "sequence": "STRING",
                    "description": "STRING"
                },
                "primary_key": "id"
            },
            NodeLabel.MUTATION: {
                "properties": {
                    "id": "STRING",
                    "name": "STRING",
                    "type": "STRING",
                    "reference_allele": "STRING",
                    "alternative_allele": "STRING",
                    "position": "INT64",
                    "consequence": "STRING",
                    "population_frequency": "FLOAT64",
                    "african_frequency": "FLOAT64",
                    "nigerian_frequency": "FLOAT64",
                    "description": "STRING"
                },
                "primary_key": "id"
            },
            NodeLabel.TREATMENT: {
                "properties": {
                    "id": "STRING",
                    "name": "STRING",
                    "type": "STRING",
                    "mechanism": "STRING",
                    "development_stage": "STRING",
                    "approval_status": "STRING",
                    "description": "STRING"
                },
                "primary_key": "id"
            },
            NodeLabel.PAPER: {
                "properties": {
                    "id": "STRING",
                    "title": "STRING",
                    "abstract": "STRING",
                    "doi": "STRING",
                    "publication_date": "STRING",
                    "journal": "STRING",
                    "citation_count": "INT64",
                    "url": "STRING"
                },
                "primary_key": "id"
            },
            NodeLabel.CLINICAL_TRIAL: {
                "properties": {
                    "id": "STRING",
                    "title": "STRING",
                    "description": "STRING",
                    "status": "STRING",
                    "phase": "STRING",
                    "start_date": "STRING",
                    "end_date": "STRING",
                    "enrollment": "INT64",
                    "url": "STRING"
                },
                "primary_key": "id"
            },
            NodeLabel.RESEARCHER: {
                "properties": {
                    "id": "STRING",
                    "name": "STRING",
                    "email": "STRING",
                    "orcid": "STRING",
                    "h_index": "INT64",
                    "publication_count": "INT64"
                },
                "primary_key": "id"
            },
            NodeLabel.INSTITUTION: {
                "properties": {
                    "id": "STRING",
                    "name": "STRING",
                    "type": "STRING",
                    "url": "STRING"
                },
                "primary_key": "id"
            },
            NodeLabel.COUNTRY: {
                "properties": {
                    "id": "STRING",
                    "name": "STRING",
                    "iso_code": "STRING",
                    "population": "INT64",
                    "scd_prevalence": "FLOAT64"
                },
                "primary_key": "id"
            },
            NodeLabel.CONTINENT: {
                "properties": {
                    "id": "STRING",
                    "name": "STRING"
                },
                "primary_key": "id"
            },
            NodeLabel.PROTEIN: {
                "properties": {
                    "id": "STRING",
                    "name": "STRING",
                    "uniprot_id": "STRING",
                    "function": "STRING",
                    "sequence": "STRING"
                },
                "primary_key": "id"
            },
            NodeLabel.PATHWAY: {
                "properties": {
                    "id": "STRING",
                    "name": "STRING",
                    "description": "STRING",
                    "kegg_id": "STRING"
                },
                "primary_key": "id"
            },
            NodeLabel.DISEASE: {
                "properties": {
                    "id": "STRING",
                    "name": "STRING",
                    "omim_id": "STRING",
                    "description": "STRING"
                },
                "primary_key": "id"
            },
            NodeLabel.SYMPTOM: {
                "properties": {
                    "id": "STRING",
                    "name": "STRING",
                    "description": "STRING"
                },
                "primary_key": "id"
            },
            NodeLabel.DRUG: {
                "properties": {
                    "id": "STRING",
                    "name": "STRING",
                    "drugbank_id": "STRING",
                    "mechanism": "STRING",
                    "approval_status": "STRING"
                },
                "primary_key": "id"
            }
        }
        
        self.relationship_schemas = {
            RelationshipType.HAS_MUTATION: {
                "source": NodeLabel.GENE,
                "target": NodeLabel.MUTATION,
                "properties": {
                    "effect": "STRING",
                    "confidence": "FLOAT64"
                }
            },
            RelationshipType.TARGETS: {
                "source": NodeLabel.TREATMENT,
                "target": NodeLabel.GENE,
                "properties": {
                    "mechanism": "STRING",
                    "efficacy": "FLOAT64",
                    "specificity": "FLOAT64"
                }
            },
            RelationshipType.TREATS: {
                "source": NodeLabel.TREATMENT,
                "target": NodeLabel.DISEASE,
                "properties": {
                    "efficacy": "FLOAT64",
                    "phase": "STRING",
                    "evidence_level": "STRING"
                }
            },
            RelationshipType.AUTHORED_BY: {
                "source": NodeLabel.PAPER,
                "target": NodeLabel.RESEARCHER,
                "properties": {
                    "author_position": "STRING",
                    "corresponding": "BOOL"
                }
            },
            RelationshipType.AFFILIATED_WITH: {
                "source": NodeLabel.RESEARCHER,
                "target": NodeLabel.INSTITUTION,
                "properties": {
                    "role": "STRING",
                    "start_date": "STRING",
                    "end_date": "STRING"
                }
            },
            RelationshipType.LOCATED_IN: {
                "source": NodeLabel.INSTITUTION,
                "target": NodeLabel.COUNTRY,
                "properties": {
                    "city": "STRING",
                    "address": "STRING"
                }
            },
            RelationshipType.PART_OF: {
                "source": NodeLabel.COUNTRY,
                "target": NodeLabel.CONTINENT,
                "properties": {
                    "region": "STRING"
                }
            },
            RelationshipType.CITES: {
                "source": NodeLabel.PAPER,
                "target": NodeLabel.PAPER,
                "properties": {
                    "context": "STRING",
                    "sentiment": "FLOAT64"
                }
            },
            RelationshipType.CONDUCTED_BY: {
                "source": NodeLabel.CLINICAL_TRIAL,
                "target": NodeLabel.INSTITUTION,
                "properties": {
                    "role": "STRING",
                    "principal_investigator": "STRING"
                }
            },
            RelationshipType.CONDUCTED_IN: {
                "source": NodeLabel.CLINICAL_TRIAL,
                "target": NodeLabel.COUNTRY,
                "properties": {
                    "site_count": "INT64",
                    "enrollment": "INT64"
                }
            },
            RelationshipType.ENCODES: {
                "source": NodeLabel.GENE,
                "target": NodeLabel.PROTEIN,
                "properties": {
                    "transcript_id": "STRING"
                }
            },
            RelationshipType.PARTICIPATES_IN: {
                "source": NodeLabel.PROTEIN,
                "target": NodeLabel.PATHWAY,
                "properties": {
                    "role": "STRING",
                    "evidence": "STRING"
                }
            },
            RelationshipType.REGULATES: {
                "source": NodeLabel.GENE,
                "target": NodeLabel.GENE,
                "properties": {
                    "mechanism": "STRING",
                    "effect": "STRING",
                    "evidence": "STRING"
                }
            },
            RelationshipType.CAUSES: {
                "source": NodeLabel.MUTATION,
                "target": NodeLabel.DISEASE,
                "properties": {
                    "mechanism": "STRING",
                    "evidence_level": "STRING"
                }
            },
            RelationshipType.MANIFESTS_AS: {
                "source": NodeLabel.DISEASE,
                "target": NodeLabel.SYMPTOM,
                "properties": {
                    "frequency": "STRING",
                    "severity": "STRING"
                }
            },
            RelationshipType.INTERACTS_WITH: {
                "source": NodeLabel.DRUG,
                "target": NodeLabel.PROTEIN,
                "properties": {
                    "mechanism": "STRING",
                    "affinity": "FLOAT64",
                    "effect": "STRING"
                }
            }
        }
    
    def generate_kuzu_schema(self) -> List[str]:
        """
        Generate Kùzu schema creation statements.
        
        Returns:
            List[str]: List of Kùzu schema creation statements.
        """
        statements = []
        
        # Create node tables
        for label, schema in self.node_schemas.items():
            props = ", ".join([f"{name} {dtype}" for name, dtype in schema["properties"].items()])
            primary_key = schema["primary_key"]
            stmt = f"CREATE NODE TABLE {label} ({props}, PRIMARY KEY ({primary_key}))"
            statements.append(stmt)
        
        # Create relationship tables
        for rel_type, schema in self.relationship_schemas.items():
            source = schema["source"]
            target = schema["target"]
            if schema["properties"]:
                props = ", ".join([f"{name} {dtype}" for name, dtype in schema["properties"].items()])
                stmt = f"CREATE REL TABLE {rel_type} (FROM {source} TO {target}, {props})"
            else:
                stmt = f"CREATE REL TABLE {rel_type} (FROM {source} TO {target})"
            statements.append(stmt)
        
        return statements
    
    def generate_neo4j_schema(self) -> List[str]:
        """
        Generate Neo4j schema creation statements.
        
        Returns:
            List[str]: List of Neo4j schema creation statements.
        """
        statements = []
        
        # Create constraints for node primary keys
        for label, schema in self.node_schemas.items():
            primary_key = schema["primary_key"]
            stmt = f"CREATE CONSTRAINT {label}_{primary_key}_constraint IF NOT EXISTS FOR (n:{label}) REQUIRE n.{primary_key} IS UNIQUE"
            statements.append(stmt)
        
        # Create indexes for common query patterns
        statements.append(f"CREATE INDEX gene_symbol_idx IF NOT EXISTS FOR (n:{NodeLabel.GENE}) ON (n.symbol)")
        statements.append(f"CREATE INDEX mutation_name_idx IF NOT EXISTS FOR (n:{NodeLabel.MUTATION}) ON (n.name)")
        statements.append(f"CREATE INDEX paper_doi_idx IF NOT EXISTS FOR (n:{NodeLabel.PAPER}) ON (n.doi)")
        statements.append(f"CREATE INDEX trial_status_idx IF NOT EXISTS FOR (n:{NodeLabel.CLINICAL_TRIAL}) ON (n.status)")
        statements.append(f"CREATE INDEX country_name_idx IF NOT EXISTS FOR (n:{NodeLabel.COUNTRY}) ON (n.name)")
        
        return statements
    
    def to_json(self) -> str:
        """
        Convert the schema to a JSON string.
        
        Returns:
            str: JSON representation of the schema.
        """
        schema_dict = {
            "nodes": self.node_schemas,
            "relationships": self.relationship_schemas
        }
        return json.dumps(schema_dict, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'KnowledgeGraphSchema':
        """
        Create a schema from a JSON string.
        
        Args:
            json_str (str): JSON representation of the schema.
            
        Returns:
            KnowledgeGraphSchema: New schema instance.
        """
        schema_dict = json.loads(json_str)
        schema = cls()
        schema.node_schemas = schema_dict["nodes"]
        schema.relationship_schemas = schema_dict["relationships"]
        return schema


def create_initial_schema() -> KnowledgeGraphSchema:
    """
    Create the initial schema for SickleGraph.
    
    Returns:
        KnowledgeGraphSchema: The initial schema.
    """
    return KnowledgeGraphSchema()


if __name__ == "__main__":
    # Example usage
    schema = create_initial_schema()
    
    print("Kùzu Schema Statements:")
    for stmt in schema.generate_kuzu_schema():
        print(stmt)
    
    print("\nNeo4j Schema Statements:")
    for stmt in schema.generate_neo4j_schema():
        print(stmt)
    
    print("\nSchema as JSON:")
    print(schema.to_json())