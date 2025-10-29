# API Reference

This document provides a complete reference for the EDAM MCP Server API, including all tools, models, and utilities.

## 🛠️ MCP Tools

### `map_to_edam_concept`

Maps a description to existing EDAM concepts with confidence scores.

#### Request

```python
class MappingRequest(BaseModel):
    description: str                    # Text to map (required)
    context: Optional[str] = None      # Additional context
    max_results: Optional[int] = 5     # Maximum matches to return
    min_confidence: Optional[float] = 0.5  # Minimum confidence threshold
```

#### Response

```python
class MappingResponse(BaseModel):
    matches: List[ConceptMatch]        # List of matched concepts
    total_matches: int                 # Total number of matches
    has_exact_match: bool              # Whether exact match found
    confidence_threshold: float        # Threshold used for filtering

class ConceptMatch(BaseModel):
    concept_uri: str                   # EDAM concept URI
    concept_label: str                 # Human-readable label
    confidence: float                  # Confidence score (0.0-1.0)
    concept_type: str                  # Type (Operation, Data, Format, Topic)
    definition: Optional[str] = None   # Concept definition
    synonyms: List[str] = []           # Alternative names
```

#### Example Usage

```python
from edam_mcp.tools.mapping import map_to_edam_concept
from edam_mcp.models.requests import MappingRequest

# Create request
request = MappingRequest(
    description="sequence alignment tool",
    context="bioinformatics software",
    max_results=3,
    min_confidence=0.7
)

# Call tool
response = await map_to_edam_concept(request, context)

# Process results
for match in response.matches:
    print(f"{match.concept_label}: {match.confidence:.2f}")
```

### `suggest_new_concept`

Suggests new EDAM concepts when no suitable existing concept is found.

#### Request

```python
class SuggestionRequest(BaseModel):
    description: str                    # Description of concept to suggest
    concept_type: Optional[str] = None # Type (Operation, Data, Format, Topic)
    parent_concept: Optional[str] = None # Suggested parent concept URI
    rationale: Optional[str] = None    # Rationale for suggestion
```

#### Response

```python
class SuggestionResponse(BaseModel):
    suggestions: List[SuggestedConcept] # List of suggested concepts
    total_suggestions: int              # Total number of suggestions
    mapping_attempted: bool             # Whether mapping was tried first
    mapping_failed_reason: Optional[str] = None  # Why mapping failed

class SuggestedConcept(BaseModel):
    suggested_label: str               # Suggested concept label
    suggested_uri: str                 # Suggested concept URI
    concept_type: str                  # Type of concept
    definition: str                    # Suggested definition
    parent_concept: Optional[str] = None # Parent concept URI
    rationale: str                     # Rationale for suggestion
    confidence: float                  # Confidence in suggestion (0.0-1.0)
```

#### Example Usage

```python
from edam_mcp.tools.suggestion import suggest_new_concept
from edam_mcp.models.requests import SuggestionRequest

# Create request
request = SuggestionRequest(
    description="quantum computing for protein folding",
    concept_type="Operation",
    rationale="New computational approach for protein structure prediction"
)

# Call tool
response = await suggest_new_concept(request, context)

# Process suggestions
for suggestion in response.suggestions:
    print(f"{suggestion.suggested_label}: {suggestion.confidence:.2f}")
```

## 📊 Models

### Request Models

#### `MappingRequest`

```python
class MappingRequest(BaseModel):
    description: str = Field(
        ...,
        description="Text description or metadata to map to EDAM concepts",
        min_length=1,
        max_length=10000
    )

    context: Optional[str] = Field(
        None,
        description="Additional context about the description",
        max_length=2000
    )

    max_results: Optional[int] = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of concept matches to return"
    )

    min_confidence: Optional[float] = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for matches"
    )
```

#### `SuggestionRequest`

```python
class SuggestionRequest(BaseModel):
    description: str = Field(
        ...,
        description="Description of the concept that needs to be suggested",
        min_length=1,
        max_length=10000
    )

    concept_type: Optional[str] = Field(
        None,
        description="Type of concept (Operation, Data, Format, Topic, Identifier)",
        pattern="^(Operation|Data|Format|Topic|Identifier)$"
    )

    parent_concept: Optional[str] = Field(
        None,
        description="Suggested parent concept URI or label",
        max_length=500
    )

    rationale: Optional[str] = Field(
        None,
        description="Rationale for why this concept should be added",
        max_length=2000
    )
```

### Response Models

#### `ConceptMatch`

```python
class ConceptMatch(BaseModel):
    concept_uri: str = Field(
        ...,
        description="URI of the matched EDAM concept"
    )

    concept_label: str = Field(
        ...,
        description="Human-readable label of the concept"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the match (0.0 to 1.0)"
    )

    concept_type: str = Field(
        ...,
        description="Type of the concept (Operation, Data, Format, Topic, Identifier)"
    )

    definition: Optional[str] = Field(
        None,
        description="Definition of the concept"
    )

    synonyms: List[str] = Field(
        default_factory=list,
        description="List of synonyms for the concept"
    )
```

#### `SuggestedConcept`

```python
class SuggestedConcept(BaseModel):
    suggested_label: str = Field(
        ...,
        description="Suggested label for the new concept"
    )

    suggested_uri: str = Field(
        ...,
        description="Suggested URI for the new concept"
    )

    concept_type: str = Field(
        ...,
        description="Type of the suggested concept"
    )

    definition: str = Field(
        ...,
        description="Definition for the suggested concept"
    )

    parent_concept: Optional[str] = Field(
        None,
        description="Suggested parent concept URI"
    )

    rationale: str = Field(
        ...,
        description="Rationale for suggesting this concept"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the suggestion quality"
    )
```

## 🔧 Configuration

### Settings Class

```python
class Settings(BaseSettings):
    # EDAM Ontology Configuration
    ontology_url: str = Field(
        default="https://raw.githubusercontent.com/edamontology/edamontology/master/EDAM_dev.owl",
        description="URL to the EDAM ontology OWL file"
    )

    # Matching Configuration
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for concept mappings"
    )

    max_suggestions: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of suggestions to return"
    )

    # Model Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for text embeddings"
    )

    # Cache Configuration
    cache_ttl: int = Field(
        default=3600,
        description="Cache TTL in seconds for ontology data"
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    model_config = {
        "env_prefix": "EDAM_",
        "case_sensitive": False
    }
```

### Environment Variables

| Variable                    | Default          | Description                   |
| --------------------------- | ---------------- | ----------------------------- |
| `EDAM_ONTOLOGY_URL`         | EDAM dev OWL     | URL to EDAM ontology file     |
| `EDAM_SIMILARITY_THRESHOLD` | 0.7              | Minimum confidence threshold  |
| `EDAM_MAX_SUGGESTIONS`      | 5                | Maximum suggestions to return |
| `EDAM_EMBEDDING_MODEL`      | all-MiniLM-L6-v2 | Sentence transformer model    |
| `EDAM_CACHE_TTL`            | 3600             | Cache TTL in seconds          |
| `EDAM_LOG_LEVEL`            | INFO             | Logging level                 |

## 🚀 Direct API Usage

### Standalone Functions

For direct usage without MCP, you can use the standalone functions:

```python
from edam_mcp.tools.mapping import map_description_to_concepts
from edam_mcp.tools.suggestion import suggest_concepts_for_description

# Direct mapping
response = await map_description_to_concepts(
    description="sequence alignment tool",
    context="bioinformatics tool",
    max_results=5,
    min_confidence=0.7
)

# Direct suggestion
response = await suggest_concepts_for_description(
    description="quantum protein folding",
    concept_type="Operation",
    max_suggestions=3
)
```

### Ontology Access

```python
from edam_mcp.ontology import OntologyLoader, ConceptMatcher, ConceptSuggester

# Load ontology
loader = OntologyLoader()
loader.load_ontology()

# Create matcher
matcher = ConceptMatcher(loader)

# Create suggester
suggester = ConceptSuggester(loader, matcher)

# Use directly
matches = matcher.match_concepts("sequence alignment")
suggestions = suggester.suggest_concepts("new concept")
```

## 🔍 Error Handling

### Common Exceptions

```python
from edam_mcp.exceptions import (
    OntologyLoadError,
    ConceptNotFoundError,
    InvalidRequestError,
    EmbeddingError
)

try:
    response = await map_to_edam_concept(request, context)
except OntologyLoadError as e:
    print(f"Failed to load ontology: {e}")
except ConceptNotFoundError as e:
    print(f"Concept not found: {e}")
except InvalidRequestError as e:
    print(f"Invalid request: {e}")
except EmbeddingError as e:
    print(f"Embedding generation failed: {e}")
```

### Error Response Format

```python
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict] = None
    timestamp: datetime
```

## 📈 Performance Considerations

### Caching

- Ontology data is cached after first load
- Embeddings are cached per concept
- ML models are loaded once and reused

### Memory Management

- Lazy loading of heavy dependencies
- Garbage collection of unused embeddings
- Configurable cache TTL

### Async Operations

- All I/O operations are async
- Concurrent request handling
- Non-blocking ontology loading

## 🔐 Security

### Input Validation

- All inputs validated with Pydantic
- SQL injection protection
- XSS protection through proper escaping

### Rate Limiting

- Configurable rate limits per client
- Request throttling
- Resource usage monitoring

### Authentication

- Optional API key authentication
- JWT token support
- Role-based access control

