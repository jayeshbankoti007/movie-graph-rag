import pandas as pd
import ast
import networkx as nx
from tqdm.autonotebook import tqdm


class MovieGraph:
    def __init__(self):
        self.Graph = nx.MultiDiGraph()

        movies = pd.read_csv("movies_metadata.csv", low_memory=False)
        credits = pd.read_csv("credits.csv")
        keywords = pd.read_csv("keywords.csv")

        movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
        credits["id"] = pd.to_numeric(credits["id"], errors="coerce")
        keywords["id"] = pd.to_numeric(keywords["id"], errors="coerce")

        self.movies_df = movies.merge(credits, on="id").merge(keywords, on="id")
        self.movies_df["id"] = self.movies_df["id"].astype(int)
        self.movies_df = self.movies_df.drop_duplicates(subset=["id"])
        self.movies_df = self.movies_df.reset_index(drop=True)
        self.movies_df = self.movies_df.dropna(subset=["release_date"])

        del movies
        del credits
        del keywords

        self.build_graph()

        # self.movies_df = self.movies_df[:50]  # For testing, limit to first 50 entries

    # Build the graph
    def build_graph(self):
        for _, row in tqdm(self.movies_df.iterrows(), total=self.movies_df.shape[0]):
            movie_id = int(row["id"])
            launch_year = row["release_date"].split("-")[0]

            # Add movie node
            self.Graph.add_node(
                movie_id,
                label="movie",
                title=row["title"].strip().lower(),
                popularity=row.get("popularity"),
                vote_average=row.get("vote_average"),
                overview=row.get("overview"),
                launch_year=launch_year,
                keywords=[k["name"] for k in ast.literal_eval(row["keywords"])],
            )

            # Launch Date
            self.Graph.add_node(launch_year, label="date")
            self.Graph.add_edge(movie_id, launch_year, relation="MOVIE_RELEASED_ON")
            self.Graph.add_edge(launch_year, movie_id, relation="YEAR_RELEASED_MOVIES")

            # Genres
            genres = ast.literal_eval(row["genres"])
            for g in genres:
                genre_name = g["name"].strip().lower()
                self.Graph.add_node(genre_name, label="genre")
                self.Graph.add_edge(movie_id, genre_name, relation="MOVIE_HAS_GENRE")
                self.Graph.add_edge(genre_name, movie_id, relation="GENRE_OF_MOVIES")

            # Actors
            cast = ast.literal_eval(row["cast"])
            actor_nodes = []
            for c in cast:
                actor_name = c["name"].strip().lower()

                if actor_name not in self.Graph:
                    self.Graph.add_node(actor_name, label="person", roles={"actor"})
                else:
                    self.Graph.nodes[actor_name].setdefault("roles", set()).add("actor")

                self.Graph.add_edge(movie_id, actor_name, relation="MOVIE_HAS_ACTOR")
                self.Graph.add_edge(actor_name, movie_id, relation="ACTED_IN_MOVIES")
                actor_nodes.append(actor_name)

            # Crew
            crew = ast.literal_eval(row["crew"])
            director_nodes = []
            for member in crew:
                job = member["job"].strip().lower()
                crew_member_name = member["name"].strip().lower()

                if job not in {"director", "producer", "writer"}:
                    job = "supporting_crew"

                if crew_member_name not in self.Graph:
                    self.Graph.add_node(crew_member_name, label="person", roles={job})
                else:
                    self.Graph.nodes[crew_member_name].setdefault("roles", set()).add(
                        job
                    )

                if job == "director":
                    director_nodes.append(crew_member_name)
                    relation = "DIRECTED_MOVIES"
                elif job == "producer":
                    relation = "PRODUCED_MOVIES"
                elif job == "writer":
                    relation = "WROTE_MOVIES"
                else:
                    relation = "SUPPORTED_MOVIES"

                self.Graph.add_edge(
                    movie_id, crew_member_name, relation=f"MOVIE_HAS_{job.upper()}"
                )
                self.Graph.add_edge(crew_member_name, movie_id, relation=relation)

            for actor in actor_nodes:
                for director in director_nodes:
                    self.Graph.add_edge(
                        actor, director, relation="ACTOR_WORKED_WITH_DIRECTOR"
                    )
                    self.Graph.add_edge(
                        director, actor, relation="DIRECTOR_WORKED_WITH_ACTOR"
                    )

        print(
            f"Graph built with {self.Graph.number_of_nodes()} nodes and {self.Graph.number_of_edges()} edges"
        )

    def query_movie_id(self, movie_id: int):
        if movie_id not in self.Graph:
            return None

        return self.Graph.nodes[movie_id]

    def query_entity_graph(self, entity, relation=None):
        if isinstance(entity, int) and entity in self.Graph.nodes:
            entity_ids = [entity]
        else:
            entity = entity.strip().lower()

            entity_ids = []
            matches = self.movies_df[
                self.movies_df["title"].str.lower() == entity.lower()
            ]

            if not matches.empty:
                entity_ids = matches["id"].tolist()
            elif entity in self.Graph.nodes:
                entity_ids = [entity]
            else:
                return []

        results = []
        for eid in entity_ids:
            if eid not in self.Graph:
                continue
            node_data = self.Graph.nodes[eid]
            neighbors = []
            for nbr, edges in self.Graph[eid].items():
                for _, edge_data in edges.items():
                    rel = edge_data.get("relation")
                    if relation and rel != relation:
                        continue
                    neighbors.append({"neighbor": nbr, "relation": rel})

            results.append(
                {
                    "node": eid,
                    "label": node_data.get("label"),
                    "node_data": node_data,
                    "neighbors": neighbors,
                }
            )
        return results

    def all_paths_query(self, node1, node2, max_len=3):
        """
        Returns all simple paths between node1 and node2 up to max_len edges.
        Each path includes:
            - Full node metadata
            - Relation to next node
        """
        if node1 not in self.Graph or node2 not in self.Graph:
            return []

        paths = list(
            nx.all_simple_paths(self.Graph, source=node1, target=node2, cutoff=max_len)
        )
        result = []

        for path in paths:
            path_data = []
            for i, node in enumerate(path):
                node_info = self.Graph.nodes[node]
                node_dict = {
                    "node": node,
                    "label": node_info.get("label"),
                    "title": (
                        node_info.get("title")
                        if node_info.get("label") == "movie"
                        else None
                    ),
                    "roles": (
                        node_info.get("roles")
                        if node_info.get("label") == "person"
                        else None
                    ),
                }

                if i < len(path) - 1:
                    edge_infos = self.Graph[node][path[i + 1]]
                    relations = [
                        edata.get("relation") for _, edata in edge_infos.items()
                    ]
                    node_dict["relation_to_next"] = relations

                path_data.append(node_dict)
            result.append(path_data)

        return result

    def shortest_path_query(self, source, target):
        """Return the shortest path between two nodes if exists."""
        if source not in self.Graph or target not in self.Graph:
            return None
        try:
            path = nx.shortest_path(self.Graph, source=source, target=target)

            # Attach relation info
            path_with_rels = []
            for i, node in enumerate(path):
                node_info = self.Graph.nodes[node]
                node_dict = {
                    "node": node,
                    "label": node_info.get("label"),
                    "title": (
                        node_info.get("title")
                        if node_info.get("label") == "movie"
                        else None
                    ),
                }
                if i < len(path) - 1:
                    edge_infos = self.Graph[node][path[i + 1]]
                    relations = [
                        edata.get("relation") for _, edata in edge_infos.items()
                    ]
                    node_dict["relation_to_next"] = relations
                path_with_rels.append(node_dict)

            return path_with_rels

        except nx.NetworkXNoPath:
            return None
