import pandas as pd
from smolagents import Tool
from faiss_setup import MovieFAISS
from network_setup import MovieGraph
import warnings

warnings.filterwarnings("ignore")
from typing import List, Dict, Any


class QueryGraphTool(Tool):
    name = "query_graph_tool"
    description = """
        Find entity information and relationships in the movie graph.
        Input: entity (movie id(int), movie title, person name, genre, or year), optional relation filter.
        Returns: entity details and connected nodes with relationship types.
        Use relation parameter to get specific connections like 'ACTED_IN_MOVIES' or 'DIRECTED_MOVIES'.
    """

    inputs = {
        "entity": {
            "type": "string",
            "description": "Entity to query: movie title, actor, genre, director, writer, or release year.",
        },
        "relation": {
            "type": "string",
            "description": "Optional relation filter to only return neighbors with this relation type.",
            "nullable": True,
        },
    }
    output_type = "array"

    def __init__(self, movie_graph: MovieGraph):
        super().__init__()
        self.movie_graph = movie_graph

    def forward(self, entity: str, relation: str = None) -> List[Dict[str, Any]]:
        try:
            # Always treat entity as string
            return self.movie_graph.query_entity_graph(entity, relation)
        except Exception as e:
            return [{"error": str(e)}]


class QueryMovieIDTool(Tool):
    name = "query_movie_id_tool"
    description = """
        Get complete movie metadata by movie ID.
        Input: movie_id (integer).
        Returns: Return movie node details from the knowledge graph.
    """
    inputs = {"movie_id": {"type": "integer", "description": "Movie ID to query."}}
    output_type = "object"

    def __init__(self, movie_graph: MovieGraph):
        super().__init__()
        self.movie_graph = movie_graph

    def forward(self, movie_id: int) -> Dict[str, Any]:
        node_data = self.movie_graph.query_movie_id(movie_id)
        if not node_data:
            return {"error": f"Movie ID {movie_id} not found"}
        return node_data


class NearestGraphTool(Tool):
    name = "nearest_graph_tool"
    inputs = {
        "entity1": {
            "type": "string",
            "description": "First entity (actor, director, movie)",
        },
        "entity2": {
            "type": "string",
            "description": "Second entity (actor, director, movie)",
        },
    }
    description = """
        Find connection paths between two entities in the movie graph.
        Input: entity1, entity2 (any movie titles or person names)
        Returns: all paths showing how the entities are connected through collaborations.
    """
    output_type = "array"

    def __init__(self, movie_graph: MovieGraph):
        super().__init__()
        self.movie_graph = movie_graph

    def forward(self, entity1: str, entity2: str):
        """
        Args:
            entity1: First entity
            entity2: Second entity
        """
        try:
            paths = self.movie_graph.all_paths_query(entity1, entity2)
            return paths if paths else []
        except Exception as e:
            return [{"error": str(e)}]


class FaissTool(Tool):
    name = "faiss_tool"
    description = """
        Semantic search for movies by plot, theme, or content description.
        Input: retrieval_query (describe plot elements, themes, or movie characteristics)
        Returns: movies matching the semantic description with metadata.
    """
    inputs = {
        "retrieval_query": {
            "type": "string",
            "description": "Optimized search query for semantic retrieval (extract key themes/concepts from user query)",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of similar movies json to return (default: 20)",
            "nullable": True,
        },
    }
    output_type = "array"

    def __init__(self, movie_faiss: MovieFAISS, movies_df: pd.DataFrame):
        super().__init__()
        self.movie_faiss = movie_faiss
        self.movies_df = movies_df

    def forward(self, retrieval_query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Perform a semantic search over the movie overviews using the provided query.
        Args:
            retrieval_query: Optimized query for semantic search
            k: Number of results to return

        """
        try:
            movie_ids = self.movie_faiss.search(retrieval_query, top_k)
            results = []
            for movie_id in movie_ids:
                movie_row = self.movies_df[self.movies_df["id"] == movie_id]
                if not movie_row.empty:
                    movie_json = {
                        k: movie_row.iloc[0].to_dict()[k]
                        for k in [
                            "id",
                            "overview",
                            "original_title",
                            "original_language",
                            "release_date",
                            "budget",
                            "revenue",
                        ]
                    }
                    results.append(movie_json)
            return results
        except Exception as e:
            return [{"error": str(e)}]


class FilterMoviesByPersonTool(Tool):
    name = "filter_movies_by_person_tool"
    description = """
        Filter a list of movie IDs to only those connected to a specific person.
        Input: person_name (string), movie_ids (list of integers)
        Returns: filtered list of movie IDs that have connection to the person.
        Much faster than checking each movie individually.
    """

    inputs = {
        "person_name": {
            "type": "string",
            "description": "Person to filter by (actor, director, producer, or writer).",
        },
        "movie_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "List of movie IDs to filter",
        },
    }
    output_type = "array"

    def __init__(self, movie_graph: MovieGraph):
        super().__init__()
        self.movie_graph = movie_graph

    def forward(self, person_name: str, movie_ids: list) -> list:
        """Filter movie IDs to only those connected to person"""
        person_name = person_name.strip().lower()

        if person_name not in self.movie_graph.Graph:
            return []

        # Get all movies connected to this person
        person_movies = set()
        for neighbor in self.movie_graph.Graph.neighbors(person_name):
            if self.movie_graph.Graph.nodes[neighbor].get("label") == "movie":
                person_movies.add(neighbor)

        # Filter the input movie_ids
        return [mid for mid in movie_ids if mid in person_movies]
