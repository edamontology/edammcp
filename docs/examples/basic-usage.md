# Basic Usage Examples

This guide provides practical examples of how to use the EDAM MCP Server for common tasks.

## 🚀 Quick Examples

### 1. Basic Concept Mapping

Map a simple description to EDAM concepts:

```python
import asyncio
from edam_mcp.tools.mapping import map_description_to_concepts

async def basic_mapping():
    """Basic example of mapping a description to EDAM concepts."""
    
    response = await map_description_to_concepts(
        description="sequence alignment tool",
        context="bioinformatics software",
        max_results=3,
        min_confidence=0.5
    )
    
    print(f"Found {response.total_matches} matches:")
    for match in response.matches:
        print(f"  - {match.concept_label} ({match.confidence:.2f})")
        print(f"    URI: {match.concept_uri}")
        print(f"    Type: {match.concept_type}")

# Run the example
asyncio.run(basic_mapping())
```

**Expected Output:**
```
Found 3 matches:
  - Sequence alignment (0.85)
    URI: http://edamontology.org/operation_0296
    Type: Operation
  - Multiple sequence alignment (0.72)
    URI: http://edamontology.org/operation_0492
    Type: Operation
  - Pairwise sequence alignment (0.68)
    URI: http://edamontology.org/operation_0293
    Type: Operation
```

### 2. Concept Suggestion

Suggest new concepts when no suitable match exists:

```python
import asyncio
from edam_mcp.tools.suggestion import suggest_concepts_for_description

async def basic_suggestion():
    """Basic example of suggesting new EDAM concepts."""
    
    response = await suggest_concepts_for_description(
        description="quantum computing for protein folding",
        concept_type="Operation",
        max_suggestions=3
    )
    
    print(f"Generated {response.total_suggestions} suggestions:")
    for suggestion in response.suggestions:
        print(f"  - {suggestion.suggested_label}")
        print(f"    Confidence: {suggestion.confidence:.2f}")
        print(f"    Definition: {suggestion.definition}")
        if suggestion.parent_concept:
            print(f"    Parent: {suggestion.parent_concept}")

# Run the example
asyncio.run(basic_suggestion())
```

**Expected Output:**
```
Generated 3 suggestions:
  - Quantum Protein Folding
    Confidence: 0.75
    Definition: Computational protein structure prediction using quantum computing algorithms
    Parent: http://edamontology.org/operation_0296
  - Quantum Molecular Dynamics
    Confidence: 0.68
    Definition: Molecular dynamics simulation using quantum mechanical methods
    Parent: http://edamontology.org/operation_0296
  - Quantum Structure Prediction
    Confidence: 0.62
    Definition: Structure prediction using quantum computing approaches
    Parent: http://edamontology.org/operation_0296
```

## 🔧 Advanced Examples

### 3. Batch Processing

Process multiple descriptions efficiently:

```python
import asyncio
from edam_mcp.tools.mapping import map_description_to_concepts

async def batch_mapping():
    """Process multiple descriptions in batch."""
    
    descriptions = [
        "DNA sequence analysis",
        "protein structure prediction",
        "gene expression analysis",
        "phylogenetic tree construction",
        "metabolomics data processing"
    ]
    
    results = {}
    
    for desc in descriptions:
        print(f"\nProcessing: {desc}")
        response = await map_description_to_concepts(
            description=desc,
            max_results=2,
            min_confidence=0.6
        )
        
        results[desc] = response.matches
        print(f"  Top match: {response.matches[0].concept_label} ({response.matches[0].confidence:.2f})")
    
    return results

# Run the example
results = asyncio.run(batch_mapping())
```

### 4. Filtering by Concept Type

Map descriptions to specific concept types:

```python
import asyncio
from edam_mcp.tools.mapping import map_description_to_concepts

async def filter_by_type():
    """Map descriptions and filter by concept type."""
    
    # Map to all types
    response = await map_description_to_concepts(
        description="FASTQ format",
        max_results=10,
        min_confidence=0.3
    )
    
    # Filter by type
    operations = [m for m in response.matches if m.concept_type == "Operation"]
    data_types = [m for m in response.matches if m.concept_type == "Data"]
    formats = [m for m in response.matches if m.concept_type == "Format"]
    
    print("Operations:")
    for op in operations[:3]:
        print(f"  - {op.concept_label} ({op.confidence:.2f})")
    
    print("\nData types:")
    for dt in data_types[:3]:
        print(f"  - {dt.concept_label} ({dt.confidence:.2f})")
    
    print("\nFormats:")
    for fmt in formats[:3]:
        print(f"  - {fmt.concept_label} ({dt.confidence:.2f})")

# Run the example
asyncio.run(filter_by_type())
```

### 5. Context-Aware Mapping

Use context to improve mapping accuracy:

```python
import asyncio
from edam_mcp.tools.mapping import map_description_to_concepts

async def context_aware_mapping():
    """Demonstrate how context improves mapping accuracy."""
    
    # Without context
    response1 = await map_description_to_concepts(
        description="alignment",
        max_results=3,
        min_confidence=0.5
    )
    
    print("Without context:")
    for match in response1.matches:
        print(f"  - {match.concept_label} ({match.confidence:.2f})")
    
    # With bioinformatics context
    response2 = await map_description_to_concepts(
        description="alignment",
        context="bioinformatics sequence analysis",
        max_results=3,
        min_confidence=0.5
    )
    
    print("\nWith bioinformatics context:")
    for match in response2.matches:
        print(f"  - {match.concept_label} ({match.confidence:.2f})")
    
    # With image processing context
    response3 = await map_description_to_concepts(
        description="alignment",
        context="image processing computer vision",
        max_results=3,
        min_confidence=0.5
    )
    
    print("\nWith image processing context:")
    for match in response3.matches:
        print(f"  - {match.concept_label} ({match.confidence:.2f})")

# Run the example
asyncio.run(context_aware_mapping())
```

## 🎯 Real-World Scenarios

### 6. Bioinformatics Tool Annotation

Annotate a bioinformatics tool with EDAM concepts:

```python
import asyncio
from edam_mcp.tools.mapping import map_description_to_concepts

async def annotate_tool():
    """Annotate a bioinformatics tool with EDAM concepts."""
    
    tool_info = {
        "name": "BLAST",
        "description": "Basic Local Alignment Search Tool for comparing biological sequences",
        "functionality": "sequence similarity search and alignment",
        "input_format": "FASTA sequence files",
        "output_format": "alignment results and statistics"
    }
    
    annotations = {}
    
    # Map main functionality
    func_response = await map_description_to_concepts(
        description=tool_info["functionality"],
        context="bioinformatics sequence analysis",
        max_results=3,
        min_confidence=0.6
    )
    annotations["operations"] = func_response.matches
    
    # Map input format
    input_response = await map_description_to_concepts(
        description=tool_info["input_format"],
        context="sequence file format",
        max_results=2,
        min_confidence=0.6
    )
    annotations["input_formats"] = input_response.matches
    
    # Map output format
    output_response = await map_description_to_concepts(
        description=tool_info["output_format"],
        context="alignment results",
        max_results=2,
        min_confidence=0.6
    )
    annotations["output_formats"] = output_response.matches
    
    # Print results
    print(f"EDAM Annotations for {tool_info['name']}:")
    print("\nOperations:")
    for op in annotations["operations"]:
        print(f"  - {op.concept_label} ({op.confidence:.2f})")
    
    print("\nInput Formats:")
    for fmt in annotations["input_formats"]:
        print(f"  - {fmt.concept_label} ({fmt.confidence:.2f})")
    
    print("\nOutput Formats:")
    for fmt in annotations["output_formats"]:
        print(f"  - {fmt.concept_label} ({fmt.confidence:.2f})")
    
    return annotations

# Run the example
annotations = asyncio.run(annotate_tool())
```

### 7. Workflow Component Mapping

Map workflow components to EDAM concepts:

```python
import asyncio
from edam_mcp.tools.mapping import map_description_to_concepts

async def map_workflow():
    """Map workflow components to EDAM concepts."""
    
    workflow_steps = [
        {
            "step": 1,
            "name": "Data Import",
            "description": "Load sequencing data from FASTQ files"
        },
        {
            "step": 2,
            "name": "Quality Control",
            "description": "Filter low-quality reads and trim adapters"
        },
        {
            "step": 3,
            "name": "Alignment",
            "description": "Map reads to reference genome"
        },
        {
            "step": 4,
            "name": "Variant Calling",
            "description": "Identify genetic variants from aligned reads"
        },
        {
            "step": 5,
            "name": "Annotation",
            "description": "Annotate variants with functional information"
        }
    ]
    
    workflow_annotations = {}
    
    for step in workflow_steps:
        print(f"\nStep {step['step']}: {step['name']}")
        
        response = await map_description_to_concepts(
            description=step["description"],
            context="bioinformatics workflow",
            max_results=2,
            min_confidence=0.6
        )
        
        workflow_annotations[step["step"]] = {
            "name": step["name"],
            "concepts": response.matches
        }
        
        for match in response.matches:
            print(f"  - {match.concept_label} ({match.confidence:.2f})")
    
    return workflow_annotations

# Run the example
workflow = asyncio.run(map_workflow())
```

## 🔍 Error Handling

### 8. Robust Error Handling

Handle errors gracefully in your applications:

```python
import asyncio
import logging
from edam_mcp.tools.mapping import map_description_to_concepts
from edam_mcp.tools.suggestion import suggest_concepts_for_description

async def robust_mapping():
    """Demonstrate robust error handling."""
    
    descriptions = [
        "sequence alignment",  # Should work
        "",  # Empty description
        "x" * 10000,  # Very long description
        "valid description"  # Should work
    ]
    
    for desc in descriptions:
        try:
            print(f"\nProcessing: '{desc[:50]}{'...' if len(desc) > 50 else ''}'")
            
            if not desc.strip():
                print("  Error: Empty description")
                continue
            
            if len(desc) > 10000:
                print("  Error: Description too long")
                continue
            
            response = await map_description_to_concepts(
                description=desc,
                max_results=1,
                min_confidence=0.5
            )
            
            if response.matches:
                print(f"  Success: {response.matches[0].concept_label}")
            else:
                print("  No matches found")
                
        except Exception as e:
            print(f"  Error: {type(e).__name__}: {e}")

# Run the example
asyncio.run(robust_mapping())
```

## 📊 Performance Tips

### 9. Optimizing Performance

```python
import asyncio
import time
from edam_mcp.tools.mapping import map_description_to_concepts

async def performance_test():
    """Test and optimize performance."""
    
    descriptions = [
        "sequence alignment",
        "protein structure prediction",
        "gene expression analysis",
        "phylogenetic analysis",
        "metabolomics analysis"
    ]
    
    # Test with different parameters
    configs = [
        {"max_results": 1, "min_confidence": 0.8},
        {"max_results": 5, "min_confidence": 0.5},
        {"max_results": 10, "min_confidence": 0.3}
    ]
    
    for config in configs:
        print(f"\nTesting config: {config}")
        start_time = time.time()
        
        results = []
        for desc in descriptions:
            response = await map_description_to_concepts(
                description=desc,
                **config
            )
            results.append(len(response.matches))
        
        end_time = time.time()
        avg_matches = sum(results) / len(results)
        
        print(f"  Time: {end_time - start_time:.2f}s")
        print(f"  Avg matches: {avg_matches:.1f}")

# Run the example
asyncio.run(performance_test())
```

## 🎓 Learning Exercises

### Exercise 1: Custom Mapping Function

Create a function that maps descriptions and returns only high-confidence matches:

```python
async def get_high_confidence_matches(description: str, threshold: float = 0.8):
    """Get only high-confidence concept matches."""
    # Your implementation here
    pass
```

### Exercise 2: Concept Type Analyzer

Create a function that analyzes the distribution of concept types in mapping results:

```python
async def analyze_concept_types(description: str):
    """Analyze the distribution of concept types in mapping results."""
    # Your implementation here
    pass
```

### Exercise 3: Suggestion Validator

Create a function that validates suggestion quality:

```python
async def validate_suggestion(description: str, suggested_label: str):
    """Validate the quality of a concept suggestion."""
    # Your implementation here
    pass
```

These examples demonstrate the core functionality of the EDAM MCP Server and provide a foundation for building more complex applications. Experiment with different parameters and contexts to find the best approach for your specific use case. 