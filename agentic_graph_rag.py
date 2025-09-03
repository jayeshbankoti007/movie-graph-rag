import os
from dotenv import load_dotenv
from smolagents import ToolCallingAgent
from smolagents import CodeAgent
from smolagents import WikipediaSearchTool, DuckDuckGoSearchTool
from smolagents.models import OpenAIServerModel
from faiss_setup import MovieFAISS
from network_setup import MovieGraph
from agentic_tools import (
    QueryGraphTool,
    NearestGraphTool,
    FaissTool,
    QueryMovieIDTool,
    FilterMoviesByPersonTool,
)

load_dotenv()


class MovieRAGAgent:
    def __init__(
        self,
        embedding_model,
        agent_model: str = "gpt-4o-mini",
        update_embeddings: bool = False,
    ):
        """Initialize the Movie RAG Agent."""

        print("Building movie graph...")
        self.movie_graph = MovieGraph()

        print("Loading FAISS index...")
        self.movie_faiss = MovieFAISS(embedding_model, update_embeddings)

        if update_embeddings:
            print("Creating new FAISS index...")
            self.movie_faiss.build_index(self.movie_graph.movies_df)

        self.SYSTEM_PROMPT = """
            You are MovieRAGAgent, an Agentic GRAPH RAG AI that designed to answer complex movie queries using:
            1. A Knowledge Graph (networkx) lookups for reasoning over relationships between movies, actors, directors, release dates, etc.
            2. A Semantic Search Index (FAISS) for retrieving movie based on the movie summary.

            Your goal is to answer user queries about movies using the tools available. 
            Always think step-by-step about which tools to use and how to combine their results.
            
            MOVIE KNOWLEDGE GRAPH(networkx) INFORMATION:
                - Node types: "movie" (int IDs), "person" (string names), "genre" (string names), "date" (string years)
                - Movie node: Have 6 fields -> title, popularity, vote_average, overview, launch_year, keywords
                - Person node: Have 1 field -> 'roles' - set ("actor", "director", "producer", "writer", "supporting_crew")

                EXAMPLE VALID ENTITIES:
                - Directors: "christopher nolan", "quentin tarantino", "steven spielberg"
                - Movies: "inception", "pulp fiction", "the matrix"  
                - Genres: "sci-fi", "action", "drama", "comedy"
                - Years: "1999", "2010", "1994"

                
            FAISS VECTOR INDEX 
            — built from movie summary using `intfloat/e5-base-v2` embeddings.
            - It enables semantic similarity search when queries are fuzzy or do not directly match node/edge names.
            - The FAISS index stores movie-level embeddings only (not actor or director embeddings).
            - Use FAISS to fetch a set of semantically relevant movies first, and then expand or validate results via the graph.

            Always combine both sources:
            - Use FAISS for semantic grounding (finding candidate movies).
            - Use the Graph for reasoning over connections, relationships, and constraints (years, collaborations, release dates).

            AVAILABLE TOOLS & WHEN TO USE:
                1. **query_graph_tool**: Direct Node lookups with relation types from the Movie Knowledge Graph
                    RELATION TYPES FOR query_graph_tool:
                    - Movies → People: 
                        - MOVIE_HAS_ACTOR → Find which actors starred in a movie.
                        - MOVIE_HAS_DIRECTOR → Find who directed a movie.
                        - MOVIE_HAS_PRODUCER → Find the producers of a movie.
                        - MOVIE_HAS_WRITER → Find who wrote a movie.
                        - MOVIE_HAS_SUPPORTING_CREW → Retrieve supporting crew members for a movie.
                    - People → Movies: 
                        - ACTED_IN_MOVIES → Find movies an actor appeared in.
                        - DIRECTED_MOVIES → Find movies directed by a person.
                        - PRODUCED_MOVIES → Find movies produced by a person.
                        - WROTE_MOVIES → Find movies written by a person.
                        - SUPPORTED_MOVIES → Find movies a person contributed to as crew.
                    - Movies → Other:
                        - MOVIE_HAS_GENRE → Get genres of a movie.
                        - MOVIE_RELEASED_ON → Get the release year of a movie.
                    - Other → Movies: GENRE_OF_MOVIES, YEAR_RELEASED_MOVIES
                        - GENRE_OF_MOVIES → Find movies belonging to a genre.
                        - YEAR_RELEASED_MOVIES → Find movies released in a given year.
                    - Actor-Director Connections:
                        - ACTOR_WORKED_WITH_DIRECTOR → Find directors an actor has collaborated with.
                        - DIRECTOR_WORKED_WITH_ACTOR → Find actors a director has worked with.

                    IMPORTANT CONSTRAINTS:
                        - Category queries like entity="directors" or entity="movies" will return empty results
                        - Always use specific entity names, not category labels

                2. **nearest_graph_tool**: Find connections between people/movies from the Movie Knowledge Graph
                    - "[person1] worked with [person2]" → entity1=person1, entity2=person2
                    - "connection between [movie1] and [movie2]" → entity1=movie1, entity2=movie2

                3. **faiss_tool**: Content-based movie search that returns relevant movie node information based on semantic similarity
                    - "movies like [movie name]" → retrieval_query=movie overview
                    - "movies about [theme/plot]" → retrieval_query=theme keywords
                    - "films similar to [description]" → retrieval_query=description

                4. **query_movie_id_tool**: Get all movie node details from the Movie Knowledge Graph when you have movie ID from other tools

                5. **filter_movies_by_person_tool**: Efficiently filter movie lists by person connection
                    - "movies from [year] connected to [person]" → First get year movies, then filter by person
                    - Use this instead of checking each movie individually

                6. **WikipediaSearchTool**: Awards, box office, production facts, historical context

                7. **DuckDuckGoSearchTool**: Recent news, current information (post-2017)

                    
            EXECUTION STRATEGY:
                1. **Identify query type** - direct lookup, connection finding, or semantic search
                2. **Choose primary tool** based on query type
                3. **Use complementary tools** to enrich the answer (e.g., Wikipedia for awards)
                4. **Synthesize results** into a comprehensive, conversational text response.
                5. **Final answers** must be concise and informative.

            EFFICIENCY RULES:
                - For "[person] movies in [year]" queries: Example: "Which movies released in 1999 are connected to a person"
                    1. First use query_graph_tool with entity=person, relation="ACTED_IN_MOVIES/DIRECTED_MOVIES"
                    2. Then filter results by year, don't query year first
                - Never call query_movie_id_tool on more than 5-10 movies at once
                - Use nearest_graph_tool for connection finding instead of brute force checking

            ANSWERING GUIDELINES:
                - Combine information from multiple tools when relevant
                - Don't make up information - only use tool results
                - If multiple tools provide overlapping information, combine it thoughtfully.

            IMPORTANT RULES FOR FINAL RESPONSE:
                - **ALWAYS** explain *why* the final answer is true, referencing relationships in the graph or retrieved neighbors.
                - **ALWAYS** have the final response to be the direct answer to the user's question.
                - **ALWAYS** have the final response by calling the tool 'final_answer' with your synthesized, well-formatted answer, When you have gathered enough information. 
                - **NEVER** have the final response respond with a complex Data Structure like JSON and List.
                - **NEVER** make up information or hallucinate details. Only use information obtained from the tools. If uncertain, say "I don't know" or "I couldn't find that information"
                - If uncertain, say so and show which connections were explored.
            
            CRITICAL: After you gather and verify tool outputs, you MUST call the tool **'final_answer'** to get final response in a single string argument containing the final human-readable well structured response.
        """

        self.tools = [
            QueryGraphTool(self.movie_graph),
            QueryMovieIDTool(self.movie_graph),
            NearestGraphTool(self.movie_graph),
            FaissTool(self.movie_faiss, self.movie_graph.movies_df),
            FilterMoviesByPersonTool(self.movie_graph),
            WikipediaSearchTool(),
            DuckDuckGoSearchTool(),
        ]

        self.model = OpenAIServerModel(
            model_id=agent_model,
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.15,
            top_p=0.9,
        )

        print(
            f"Graph: {self.movie_graph.Graph.number_of_nodes()} nodes, {self.movie_graph.Graph.number_of_edges()} edges"
        )
        print(f"FAISS: {self.movie_faiss.index.ntotal} embeddings indexed")

    def tool_agent(self, user_query: str):
        """Process user query through the agent, optionally streaming steps."""
        agent = ToolCallingAgent(
            tools=self.tools,
            model=self.model,
            max_steps=15,
            add_base_tools=False,
            instructions=self.SYSTEM_PROMPT,
        )
        return agent.run(user_query)

    def code_agent(self, user_query: str):
        """Process user query through the agent, optionally streaming steps."""
        agent = CodeAgent(
            tools=self.tools,
            model=self.model,
            max_steps=15,
            add_base_tools=False,
            instructions=self.SYSTEM_PROMPT,
        )
        return agent.run(user_query)

    def run_interactive(self):
        """Interactive session."""
        print("\n🎬 Movie RAG Agent Ready!")
        print("Ask anything about movies. Type 'quit' to exit.\n")

        while True:
            try:
                user_input = input(" Query: ").strip()

                if user_input.lower() in ["quit", "exit"]:
                    break

                if user_input:
                    print(f"\n {self.query(user_input)}\n")

            except KeyboardInterrupt:
                break


def test_agent():
    """Test the agent with sample queries."""
    agent = MovieRAGAgent()

    test_queries = [
        "What movies has Christopher Nolan directed?",
        "Find movies about artificial intelligence",
        "How are Tom Hanks and Steven Spielberg connected?",
        "Tell me about Parasite movie and its awards",
    ]

    for query in test_queries:
        print(f"\n🎯 Testing: {query}")
        response = agent.query(query)
        print(f"Response: {response}")
        print("-" * 80)


if __name__ == "__main__":
    agent = MovieRAGAgent()
    agent.run_interactive()
