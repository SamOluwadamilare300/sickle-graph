"""
Command-line interface for SickleGraph.

This module provides the command-line interface for SickleGraph.
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from sicklegraph.graph import GraphDatabaseFactory, create_initial_schema # type: ignore
from sicklegraph.pipeline import SickleGraphPipeline, PubMedSource, ClinicalTrialsSource # type: ignore
from sicklegraph.eliza import ElizaAssistant # type: ignore
from sicklegraph.inference import TargetPredictor, TrialMatcher # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def init_graph(args):
    """
    Initialize the knowledge graph.
    
    Args:
        args: The command-line arguments.
    """
    # Create the graph database
    graph = GraphDatabaseFactory.create_from_env()
    
    if not graph:
        logger.error("Failed to create graph database")
        sys.exit(1)
    
    # Connect to the graph database
    if not graph.connect():
        logger.error("Failed to connect to graph database")
        sys.exit(1)
    
    # Initialize the schema
    schema = create_initial_schema()
    
    if not graph.initialize_schema(schema):
        logger.error("Failed to initialize schema")
        sys.exit(1)
    
    logger.info("Knowledge graph initialized successfully")


def run_pipeline(args):
    """
    Run the data pipeline.
    
    Args:
        args: The command-line arguments.
    """
    # Create the graph database
    graph = GraphDatabaseFactory.create_from_env()
    
    if not graph:
        logger.error("Failed to create graph database")
        sys.exit(1)
    
    # Connect to the graph database
    if not graph.connect():
        logger.error("Failed to connect to graph database")
        sys.exit(1)
    
    # Create the pipeline
    pipeline = SickleGraphPipeline(graph)
    
    # Register data sources
    if args.sources == "all" or "pubmed" in args.sources:
        email = os.environ.get("SICKLEGRAPH_PUBMED_EMAIL")
        api_key = os.environ.get("SICKLEGRAPH_PUBMED_API_KEY")
        
        if not email:
            logger.error("SICKLEGRAPH_PUBMED_EMAIL environment variable not set")
            sys.exit(1)
        
        pubmed_source = PubMedSource(email, api_key)
        pipeline.register_source(pubmed_source)
    
    if args.sources == "all" or "clinical_trials" in args.sources:
        clinical_trials_source = ClinicalTrialsSource()
        pipeline.register_source(clinical_trials_source)
    
    # Run the pipeline
    if not pipeline.run(
        None if args.sources == "all" else args.sources.split(","),
        query=args.query
    ):
        logger.error("Failed to run pipeline")
        sys.exit(1)
    
    logger.info("Data pipeline completed successfully")


def eliza_query(args):
    """
    Query the ELIZA AI Research Assistant.
    
    Args:
        args: The command-line arguments.
    """
    # Create the graph database
    graph = GraphDatabaseFactory.create_from_env()
    
    if not graph:
        logger.error("Failed to create graph database")
        sys.exit(1)
    
    # Connect to the graph database
    if not graph.connect():
        logger.error("Failed to connect to graph database")
        sys.exit(1)
    
    # Create the ELIZA assistant
    eliza = ElizaAssistant(graph)
    
    # Query ELIZA
    response = eliza.query(args.query, args.language)
    
    # Print the response
    print(response)


def predict_targets(args):
    """
    Predict gene therapy targets.
    
    Args:
        args: The command-line arguments.
    """
    # Create the graph database
    graph = GraphDatabaseFactory.create_from_env()
    
    if not graph:
        logger.error("Failed to create graph database")
        sys.exit(1)
    
    # Connect to the graph database
    if not graph.connect():
        logger.error("Failed to connect to graph database")
        sys.exit(1)
    
    # Create the target predictor
    predictor = TargetPredictor(graph)
    
    # Predict targets
    targets = predictor.predict_targets(args.mutation, args.population, args.top_n)
    
    # Print the targets
    print(json.dumps(targets, indent=2))


def predict_off_targets(args):
    """
    Predict off-target effects.
    
    Args:
        args: The command-line arguments.
    """
    # Create the graph database
    graph = GraphDatabaseFactory.create_from_env()
    
    if not graph:
        logger.error("Failed to create graph database")
        sys.exit(1)
    
    # Connect to the graph database
    if not graph.connect():
        logger.error("Failed to connect to graph database")
        sys.exit(1)
    
    # Create the target predictor
    predictor = TargetPredictor(graph)
    
    # Predict off-targets
    off_targets = predictor.predict_off_targets(args.gene, args.population, args.top_n)
    
    # Print the off-targets
    print(json.dumps(off_targets, indent=2))


def match_trials(args):
    """
    Match clinical trials.
    
    Args:
        args: The command-line arguments.
    """
    # Create the graph database
    graph = GraphDatabaseFactory.create_from_env()
    
    if not graph:
        logger.error("Failed to create graph database")
        sys.exit(1)
    
    # Connect to the graph database
    if not graph.connect():
        logger.error("Failed to connect to graph database")
        sys.exit(1)
    
    # Create the trial matcher
    matcher = TrialMatcher(graph)
    
    # Match trials
    trials = matcher.match_trials(args.mutation, args.population, args.top_n)
    
    # Print the trials
    print(json.dumps(trials, indent=2))


def predict_treatment_outcome(args):
    """
    Predict treatment outcome.
    
    Args:
        args: The command-line arguments.
    """
    # Create the graph database
    graph = GraphDatabaseFactory.create_from_env()
    
    if not graph:
        logger.error("Failed to create graph database")
        sys.exit(1)
    
    # Connect to the graph database
    if not graph.connect():
        logger.error("Failed to connect to graph database")
        sys.exit(1)
    
    # Create the target predictor
    predictor = TargetPredictor(graph)
    
    # Predict treatment outcome
    outcome = predictor.engine.predict_treatment_outcomes(args.treatment_id, args.mutation, args.population)
    
    # Print the outcome
    print(json.dumps(outcome, indent=2))


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="SickleGraph command-line interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # init-graph command
    init_graph_parser = subparsers.add_parser("init-graph", help="Initialize the knowledge graph")
    
    # run-pipeline command
    run_pipeline_parser = subparsers.add_parser("run-pipeline", help="Run the data pipeline")
    run_pipeline_parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help="Data sources to run (comma-separated, or 'all')"
    )
    run_pipeline_parser.add_argument(
        "--query",
        type=str,
    )
    run_pipeline_parser.add_argument(
        "--query",
        type=str,
        default="sickle cell disease gene therapy",
        help="Query to use for data sources"
    )
    
    # eliza-query command
    eliza_query_parser = subparsers.add_parser("eliza-query", help="Query the ELIZA AI Research Assistant")
    eliza_query_parser.add_argument("query", type=str, help="Query text")
    eliza_query_parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language of the query (en, yo, ha, ig)"
    )
    
    # predict-targets command
    predict_targets_parser = subparsers.add_parser("predict-targets", help="Predict gene therapy targets")
    predict_targets_parser.add_argument(
        "--mutation",
        type=str,
        default="HbSS",
        help="Mutation type (e.g., HbSS)"
    )
    predict_targets_parser.add_argument(
        "--population",
        type=str,
        default=None,
        help="Population (e.g., nigerian)"
    )
    predict_targets_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top targets to return"
    )
    
    # predict-off-targets command
    predict_off_targets_parser = subparsers.add_parser("predict-off-targets", help="Predict off-target effects")
    predict_off_targets_parser.add_argument(
        "--gene",
        type=str,
        required=True,
        help="Gene symbol (e.g., HBB)"
    )
    predict_off_targets_parser.add_argument(
        "--population",
        type=str,
        default=None,
        help="Population (e.g., nigerian)"
    )
    predict_off_targets_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top off-targets to return"
    )
    
    # match-trials command
    match_trials_parser = subparsers.add_parser("match-trials", help="Match clinical trials")
    match_trials_parser.add_argument(
        "--mutation",
        type=str,
        default="HbSS",
        help="Mutation type (e.g., HbSS)"
    )
    match_trials_parser.add_argument(
        "--population",
        type=str,
        default=None,
        help="Population (e.g., nigerian)"
    )
    match_trials_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top trials to return"
    )
    
    # predict-treatment-outcome command
    predict_treatment_outcome_parser = subparsers.add_parser("predict-treatment-outcome", help="Predict treatment outcome")
    predict_treatment_outcome_parser.add_argument(
        "--treatment-id",
        type=str,
        required=True,
        help="Treatment ID"
    )
    predict_treatment_outcome_parser.add_argument(
        "--mutation",
        type=str,
        default="HbSS",
        help="Mutation type (e.g., HbSS)"
    )
    predict_treatment_outcome_parser.add_argument(
        "--population",
        type=str,
        default=None,
        help="Population (e.g., nigerian)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "init-graph":
        init_graph(args)
    elif args.command == "run-pipeline":
        run_pipeline(args)
    elif args.command == "eliza-query":
        eliza_query(args)
    elif args.command == "predict-targets":
        predict_targets(args)
    elif args.command == "predict-off-targets":
        predict_off_targets(args)
    elif args.command == "match-trials":
        match_trials(args)
    elif args.command == "predict-treatment-outcome":
        predict_treatment_outcome(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()