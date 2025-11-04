"""MCP tool for workflow summary and planning."""

from fastmcp.server import Context

from ..models.workflow import WorkflowFunction, WorkflowSummaryRequest, WorkflowSummaryResponse


async def get_workflow_summary(
    request: WorkflowSummaryRequest, context: Context
) -> WorkflowSummaryResponse:
    """Get comprehensive summary of the EDAM mapping workflow.

    This entry point function provides a complete overview of the EDAM ontology mapping
    workflow, including all available functions, their expected inputs/outputs, configurable
    options, and workflow steps. This is designed for copilot planning and assessment of
    the current state of the mapping process.

    Args:
        request: Workflow summary request (no parameters needed).
        context: MCP context for logging.

    Returns:
        Workflow summary response with complete workflow information.
    """
    context.info("Generating workflow summary...")

    # Define all workflow functions
    functions = [
        WorkflowFunction(
            name="segment_text",
            description=(
                "Detects whether input text represents a single unit or multiple concepts. "
                "Analyzes text structure, length, and semantic content to determine if segmentation "
                "is needed. Returns either a single string for simple inputs or a list of segmented "
                "items for sequential processing."
            ),
            input_format={
                "text": "str - Input text (can be single term, summary, paragraph, vignette, or structured metadata)",
                "options": {
                    "max_segment_length": "int (optional) - Maximum length for a single segment",
                    "delimiter_patterns": "list[str] (optional) - Patterns to detect segment boundaries",
                    "semantic_threshold": "float (optional) - Threshold for semantic similarity between segments",
                },
            },
            output_format={
                "is_segmented": "bool - Whether text was segmented",
                "items": "str | list[str] - Single string or list of segmented items",
                "segmentation_method": "str - Method used for segmentation (if segmented)",
                "confidence": "float - Confidence in segmentation decision",
            },
            configurable_options=[
                "max_segment_length",
                "delimiter_patterns",
                "semantic_threshold",
            ],
            dependencies=[],
        ),
        WorkflowFunction(
            name="map_to_edam",
            description=(
                "Main mapping function for linking text items to EDAM ontology terms. "
                "Performs semantic search using embeddings to find the most appropriate EDAM concepts. "
                "Supports multiple embedding models, optional synthetic term generation to improve search, "
                "and returns additional ontology statistics (e.g., concept depth, number of children, "
                "hierarchy position) for downstream processing."
            ),
            existing_implementation=(
                "map_to_edam_concept_tool (partial - provides basic semantic matching but lacks "
                "synthetic term generation and ontology statistics features)"
            ),
            input_format={
                "item": "str - Single text item to map to EDAM",
                "context": "str | None (optional) - Additional context about the item",
                "options": {
                    "embedding_model": "str (optional) - Embedding model to use (e.g., 'all-MiniLM-L6-v2')",
                    "generate_synthetic_terms": "bool (optional) - Whether to generate synthetic terms for better matching",
                    "include_ontology_stats": "bool (optional) - Whether to include ontology statistics",
                    "max_results": "int (optional) - Maximum number of matches to return",
                    "min_confidence": "float (optional) - Minimum confidence threshold",
                },
            },
            output_format={
                "matches": "list[ConceptMatch] - List of matched EDAM concepts",
                "best_match": "ConceptMatch | None - Best matching concept",
                "ontology_stats": {
                    "concept_depth": "int - Depth of concept in ontology hierarchy",
                    "num_children": "int - Number of child concepts",
                    "num_parents": "int - Number of parent concepts",
                    "hierarchy_path": "list[str] - Path from root to concept",
                },
                "embedding_model_used": "str - Embedding model used for matching",
                "synthetic_terms_generated": "list[str] - Synthetic terms generated (if enabled)",
            },
            configurable_options=[
                "embedding_model",
                "generate_synthetic_terms",
                "include_ontology_stats",
                "max_results",
                "min_confidence",
            ],
            dependencies=["segment_text"],
        ),
        WorkflowFunction(
            name="commonsense_check",
            description=(
                "Validates whether a mapped EDAM term faithfully represents the input text. "
                "Uses unsupervised embedding comparison and ontology structure traversal to detect "
                "overly generic or overly specific mappings. If the term is too generic, traverses "
                "children to find more specific matches. If too specific, traverses parents to find "
                "more general matches. Returns adjusted mapping with rationale."
            ),
            input_format={
                "input_text": "str - Original input text that was mapped",
                "mapped_concept": "ConceptMatch - The mapped EDAM concept to validate",
                "options": {
                    "similarity_threshold": "float (optional) - Minimum similarity threshold for validation",
                    "max_traversal_depth": "int (optional) - Maximum depth for ontology traversal",
                    "prefer_specific": "bool (optional) - Prefer more specific over generic terms",
                },
            },
            output_format={
                "is_valid": "bool - Whether the mapping is considered valid",
                "adjusted_match": "ConceptMatch | None - Adjusted concept if validation failed",
                "validation_confidence": "float - Confidence in the validation result",
                "traversal_path": "list[str] - Path traversed in ontology (if adjusted)",
                "rationale": "str - Explanation of validation result or adjustment",
            },
            configurable_options=[
                "similarity_threshold",
                "max_traversal_depth",
                "prefer_specific",
            ],
            dependencies=["map_to_edam"],
        ),
        WorkflowFunction(
            name="merge_results",
            description=(
                "Combines results from single or multiple item mappings into a unified structure. "
                "Handles aggregation of confidence scores, deduplication of concepts, and merging "
                "of ontology statistics. Includes flags for unresolved items and mapping quality metrics."
            ),
            input_format={
                "mapping_results": "list[MappingResult] - Results from individual item mappings",
                "options": {
                    "deduplicate": "bool (optional) - Whether to deduplicate identical concepts",
                    "confidence_aggregation": "str (optional) - Method for aggregating confidence ('max', 'mean', 'weighted')",
                    "include_unresolved": "bool (optional) - Whether to include unresolved items in output",
                },
            },
            output_format={
                "merged_matches": "list[ConceptMatch] - Merged and deduplicated concept matches",
                "unresolved_items": "list[str] - Items that could not be mapped",
                "aggregated_confidence": "float - Overall confidence score",
                "mapping_coverage": "float - Percentage of items successfully mapped",
                "statistics": {
                    "total_items": "int - Total number of items processed",
                    "resolved_items": "int - Number of successfully mapped items",
                    "unique_concepts": "int - Number of unique EDAM concepts found",
                },
            },
            configurable_options=[
                "deduplicate",
                "confidence_aggregation",
                "include_unresolved",
            ],
            dependencies=["map_to_edam", "commonsense_check"],
        ),
        WorkflowFunction(
            name="report_summary",
            description=(
                "Generates a concise summary report of the mapping session. Provides statistics "
                "on mapping coverage, confidence distribution, flagged items for review, and "
                "overall mapping quality metrics. Useful for assessing the success of the mapping "
                "process and identifying areas that need manual review."
            ),
            input_format={
                "merged_results": "MergedResults - Results from merge_results function",
                "options": {
                    "include_confidence_distribution": "bool (optional) - Include confidence score distribution",
                    "include_flagged_items": "bool (optional) - Include list of items flagged for review",
                    "include_statistics": "bool (optional) - Include detailed statistics",
                },
            },
            output_format={
                "summary": {
                    "total_items_processed": "int - Total number of items processed",
                    "successfully_mapped": "int - Number of successfully mapped items",
                    "mapping_coverage": "float - Percentage coverage",
                    "average_confidence": "float - Average confidence score",
                    "confidence_distribution": "dict[str, int] - Distribution of confidence scores by range",
                },
                "flagged_items": "list[dict] - Items flagged for manual review with reasons",
                "recommendations": "list[str] - Recommendations for improving mapping quality",
                "quality_metrics": {
                    "high_confidence_mappings": "int - Number of high confidence mappings (>0.8)",
                    "medium_confidence_mappings": "int - Number of medium confidence mappings (0.5-0.8)",
                    "low_confidence_mappings": "int - Number of low confidence mappings (<0.5)",
                },
            },
            configurable_options=[
                "include_confidence_distribution",
                "include_flagged_items",
                "include_statistics",
            ],
            dependencies=["merge_results"],
        ),
        WorkflowFunction(
            name="update_opts",
            description=(
                "Interface to modify mapping parameters and re-run selected workflow steps. "
                "Allows dynamic adjustment of embedding models, confidence thresholds, traversal "
                "behavior, and other mapping parameters. Supports re-running specific workflow steps "
                "with updated parameters without restarting the entire workflow."
            ),
            input_format={
                "parameter_updates": "dict - Dictionary of parameter names and new values",
                "affected_steps": "list[str] (optional) - Workflow steps to re-run with new parameters",
                "options": {
                    "reset_cache": "bool (optional) - Whether to reset cached embeddings/results",
                    "validate_updates": "bool (optional) - Whether to validate parameter updates",
                },
            },
            output_format={
                "updated_parameters": "dict - Confirmed updated parameters",
                "re_run_results": "dict | None - Results from re-running affected steps (if specified)",
                "validation_errors": "list[str] - Any validation errors encountered",
            },
            configurable_options=[
                "reset_cache",
                "validate_updates",
            ],
            dependencies=[],  # Can be called at any point in workflow
        ),
    ]

    # Define global configurable options
    configurable_options = {
        "embedding_model": {
            "type": "str",
            "default": "all-MiniLM-L6-v2",
            "description": "Sentence transformer model for generating embeddings",
            "options": ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"],
        },
        "min_confidence_threshold": {
            "type": "float",
            "default": 0.5,
            "description": "Minimum confidence threshold for accepting mappings",
            "range": [0.0, 1.0],
        },
        "max_results": {
            "type": "int",
            "default": 5,
            "description": "Maximum number of concept matches to return per item",
            "range": [1, 20],
        },
        "generate_synthetic_terms": {
            "type": "bool",
            "default": False,
            "description": "Whether to generate synthetic terms to improve semantic search",
        },
        "include_ontology_stats": {
            "type": "bool",
            "default": True,
            "description": "Whether to include ontology statistics (depth, children count, etc.)",
        },
        "commonsense_check_enabled": {
            "type": "bool",
            "default": True,
            "description": "Whether to perform commonsense validation of mappings",
        },
        "max_traversal_depth": {
            "type": "int",
            "default": 3,
            "description": "Maximum depth for ontology traversal during commonsense check",
            "range": [1, 10],
        },
        "prefer_specific_terms": {
            "type": "bool",
            "default": True,
            "description": "Prefer more specific terms over generic ones during traversal",
        },
    }

    # Define workflow flow
    workflow_flow = """
    EDAM Ontology Mapping Workflow:

    1. **Input Processing (segment_text)**
       - Accept text input (single term, summary, paragraph, vignette, or structured metadata)
       - Determine if input represents single unit or multiple concepts
       - Return single string or list of segmented items

    2. **Mapping (map_to_edam)**
       - For each item (or single item), perform semantic search against EDAM ontology
       - Use selected embedding model to find similar concepts
       - Optionally generate synthetic terms to improve search
       - Return matched concepts with confidence scores and ontology statistics

    3. **Validation (commonsense_check)**
       - Validate that mapped terms faithfully represent input
       - Use ontology traversal to adjust overly generic/specific mappings
       - Return validated/adjusted mappings with rationale

    4. **Result Aggregation (merge_results)**
       - Combine results from single or multiple item mappings
       - Deduplicate concepts and aggregate confidence scores
       - Flag unresolved items

    5. **Reporting (report_summary)**
       - Generate summary statistics of mapping session
       - Show mapping coverage, confidence distribution, and flagged items
       - Provide recommendations for improvement

    6. **Parameter Updates (update_opts)**
       - Can be called at any point to modify mapping parameters
       - Supports re-running specific workflow steps with updated parameters
    """

    response = WorkflowSummaryResponse(
        workflow_name="EDAM Ontology Mapping Workflow",
        workflow_description=(
            "A comprehensive workflow for mapping text descriptions (terms, summaries, paragraphs, "
            "vignettes, or structured metadata) to concepts in the EDAM ontology. The workflow supports "
            "single and multi-item processing, semantic search with configurable embeddings, validation "
            "through ontology traversal, result aggregation, and comprehensive reporting."
        ),
        workflow_steps=[
            "segment_text",
            "map_to_edam",
            "commonsense_check",
            "merge_results",
            "report_summary",
            "update_opts",
        ],
        functions=functions,
        configurable_options=configurable_options,
        workflow_flow=workflow_flow,
    )

    context.info("Workflow summary generated successfully")
    return response

