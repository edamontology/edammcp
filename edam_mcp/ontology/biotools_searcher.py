from time import sleep
import logging
import requests
from ..models.query import BiotoolsHit

logger = logging.getLogger(__name__)

class BiotoolsSearcher:
    
    def _explore_api_url(self, matches, url):
        r = requests.get(url)
        data = r.json()
        next_page = data['next']

        for d in data['list']:
            name = d['name']
            description = d['description']
            biotools_link = f"https:://bio.tools/{d['biotoolsID']}"

            topic_list = list( map( lambda x: x['uri'], d['topic'] ))
            
            data_list = []
            format_list = []
            operation_list = [] 
            function_elements = d['function']
            for fel in function_elements:
                if( "operation" in fel ):
                    operation_list += list( map( lambda x: x['uri'], fel['operation'] ))
                
                if( "input" in fel ):
                    inputs = fel['input']
                    for el in inputs:
                        data_list += [ el['data']['uri'] ]
                        format_list += list( map( lambda x: x['uri'], el['format'] ))
                
                if( "output" in fel ):
                    outputs = fel['output']
                    for el in outputs:
                        data_list += [ el['data']['uri'] ]
                        format_list += list( map( lambda x: x['uri'], el['format'] ))
            
            if( (len(topic_list) > 0) or (len(operation_list) > 0) or (len(data_list) > 0) or (len(format_list) > 0) ):
                h = BiotoolsHit(
                    name = name,
                    description = description,
                    biotools_link = biotools_link,
                    topic_terms = list(set(topic_list)),
                    operation_terms = list(set(operation_list)),
                    data_terms = list(set(data_list)),
                    format_terms = list(set(format_list)),
                )
            matches.append(h)
            sleep(1)

        return next_page, matches

    def query(
        self, 
        seed: str,
        max_results: int = 10,
    ) -> list[BiotoolsHit]:
        """Search for a tool in Biotools database using a keyword as seed

        Args:
            seed: Search keyword to search in tool names and descriptions.
            max_results: Maximum number of matches to return.

        Returns:
            List of Biotools tools that matched the keyword
        """
        matches = []

        url = f"https://bio.tools/api/t/?q={seed}&format=json&per_page=100"
        r = requests.get(url)
        data = r.json()
        total = data['count']
        next_page = data['next']
        matches += self._explore_api_url( matches, url)

        while next_page is not None:
            url_search = f"{url}{next_page}"
            next_page, matches = self._explore_api_url( matches, url_search)
            if( len(matches) > max_results ):
            	break

        logger.info(f"There are {total} tools in bio.tools related to your keyword, displaying only {max_results}")

        return matches[:max_results]
