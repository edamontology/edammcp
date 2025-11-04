"""Concept matching functionality for mapping descriptions to EDAM concepts."""

import logging
from enum import StrEnum

import requests

from ..models.responses import BioToolsInfo, ConceptMatch

logger = logging.getLogger(__name__)


class OntologyTypes(StrEnum):
    OPERATION = "operation"
    TOPIC = "topic"
    INPUT = "input"
    OUTPUT = "output"


class BiotoolsScraper:
    """Obtains EDAM concepts form a tool in bio.tools."""

    def __init__(self, ontology_loader=None):
        """Initialize the bio.tools scraper.

        Args:
            ontology_loader: Optional ontology loader instance. If not provided, a new one will be created.
        """
        if ontology_loader is None:
            from .loader import OntologyLoader

            self.ontology_loader = OntologyLoader()
            # Load ontology if not already loaded
            if not self.ontology_loader.concepts:
                self.ontology_loader.load_ontology()
        else:
            self.ontology_loader = ontology_loader

    def get_concepts_from_biotools(
        self,
        tool_name: str | None,
        tool_curie: str | None = None,
        ontology_type: str = OntologyTypes.OPERATION,
    ) -> list[ConceptMatch]:
        """Get EDAM ontology concepts from bio.tools.

        Args:
            tool_name (str|None): The name of the tool.
            tool_curie (str|None): The biotoolsCURIE of the tool.
            ontology_type (str): What ontology terms to retrieve. Can be one of [operation, input, output, topic]

        Returns:
            List of concept matches.
        """
        if ontology_type not in OntologyTypes:
            raise ValueError("Not a valid ontology type")

        if not tool_name and not tool_curie:
            raise ValueError("Either tool_name or tool_curie must be provided")

        if ontology_type not in OntologyTypes:
            raise ValueError("Not a valid ontology type")

        tool_info = self._get_biotools_ontology(tool_name, tool_curie)
        try:
            ontology_terms = getattr(tool_info, ontology_type)
        except AttributeError:
            # Info not found
            logger.error(f"Bio.tools information for tool {tool_name} not found.")
            return []

        matches = []
        if ontology_type in [OntologyTypes.OPERATION, OntologyTypes.TOPIC]:
            for term in ontology_terms:
                concept = self.ontology_loader.get_concept(term.get("uri"))
                match = ConceptMatch(
                    concept_uri=term.get("uri"),
                    concept_label=concept["label"],
                    confidence=None,
                    concept_type=concept["type"],
                    definition=concept["definition"],
                    synonyms=concept["synonyms"],
                )
                matches.append(match)
        elif ontology_type in [OntologyTypes.INPUT, OntologyTypes.OUTPUT]:
            for io in ontology_terms:
                data_term = io.get("data")
                format_terms = io.get("format")
                concept = self.ontology_loader.get_concept(data_term.get("uri"))
                match = ConceptMatch(
                    concept_uri=data_term.get("uri"),
                    concept_label=concept["label"],
                    confidence=None,
                    concept_type=concept["type"],
                    definition=concept["definition"],
                    synonyms=concept["synonyms"],
                )
                matches.append(match)
                for term in format_terms:
                    concept = self.ontology_loader.get_concept(term.get("uri"))
                    match = ConceptMatch(
                        concept_uri=term.get("uri"),
                        concept_label=concept["label"],
                        confidence=None,
                        concept_type=concept["type"],
                        definition=concept["definition"],
                        synonyms=concept["synonyms"],
                    )
                    matches.append(match)

        return matches

    def _get_biotools_ontology(self, tool_name: str | None, biotools_curie: str | None) -> BioToolsInfo | None:
        """
        Given a specific entry of the tools list associated to the module, return the biotools input ontology ID.
        Get the associated ontology terms from a tool in bio.tools.

        Args:
            tool_name (str|None): The name of the tool to get the bio.tools information for.
            biotools_curie (str|None): The biotools CURIE to get the bio.tools information for.

        Returns:
            BioToolsInfo: The information extracted from bio.tools for the tool.
        """
        # Use biotools_curie for search if tool_name is not available
        search_term = tool_name or biotools_curie or ""
        url = f"https://bio.tools/api/t/?q={search_term}&format=json"
        try:
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()
            data_list = data.get("list", [])

        except requests.exceptions.RequestException:
            logger.error(f"Could not find the bio.tools entry for the tool {tool_name}.")
            return None

        selected_tool = []
        try:
            if biotools_curie:
                selected_tool = [
                    tool for tool in data_list if tool.get("biotoolsCURIE", None).lower() == biotools_curie.lower()
                ][0]
            else:
                selected_tool = [tool for tool in data_list if tool.get("name", None).lower() == tool_name.lower()][0]
        except IndexError:
            pass

        if not selected_tool:
            logger.error(f"The tool '{tool_name}' with biotools CUIRE '{biotools_curie}' was not found.")
            return None

        tool_functions = selected_tool.get("function", [])

        operations = []
        inputs = []
        outputs = []
        for function in tool_functions:
            operations += function.get(OntologyTypes.OPERATION, "")
            inputs += function.get(OntologyTypes.INPUT, "")
            outputs += function.get(OntologyTypes.OUTPUT, "")

        tool_info = BioToolsInfo(
            name=selected_tool.get("name", ""),
            biotools_curie=selected_tool.get("biotoolsCURIE", ""),
            description=selected_tool.get("description", ""),
            operation=operations,
            input=inputs,
            output=outputs,
            topic=selected_tool.get(OntologyTypes.TOPIC, []),
        )

        return tool_info
