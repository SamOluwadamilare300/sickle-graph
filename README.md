# SickleGraph

An AI-powered knowledge graph system for accelerating gene therapy innovation for sickle cell disease (SCD) in Africa, with an initial focus on Nigeria.

## Overview

SickleGraph leverages knowledge graph technologies and artificial intelligence to model complex biomedical relationships relevant to SCD and gene therapy. The system integrates diverse biomedical data sources, provides a natural language interface for researchers, offers predictive capabilities to accelerate research, and exposes all functionality through a comprehensive API.

## Core Components

1. **Knowledge Graph Database**: Implements a robust knowledge graph using Kùzu (with Neo4j as an alternative) to model complex biomedical relationships.
2. **Data Pipeline**: Automates the integration of diverse biomedical data sources into the knowledge graph.
3. **ELIZA AI Research Assistant**: Provides an intuitive conversational interface for researchers to query the knowledge graph, with support for African languages.
4. **Advanced Inference Engine**: Offers predictive capabilities to accelerate research and discovery, with a focus on African genetic contexts.
5. **API Layer**: Provides programmatic access to all SickleGraph functionalities using FastAPI.

## Installation

### Prerequisites

- Python 3.9+
- Kùzu or Neo4j
- Required Python packages (see requirements.txt)
