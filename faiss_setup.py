from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from tqdm.autonotebook import tqdm
import math
import os


class MovieFAISS:
    def __init__(self, model_name, update_embeddings):
        self.selected_model_name = model_name.split("/")[-1]
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.index_to_id = {}
        self.faiss_index_file = f"{self.selected_model_name}_faiss_index.bin"
        self.id_mapping_file = f"{self.selected_model_name}_mapping.pkl"

        if not update_embeddings:
            self.load(self.faiss_index_file, self.id_mapping_file)

    def format_for_e5(self, text_value: str) -> str:
        return f"passage: {text_value.strip()}"

    def build_index(self, movies_df, embedding_column="overview", batch_size=64):
        movie_ids = movies_df["id"].tolist()
        texts = movies_df[embedding_column].fillna("").tolist()

        if "e5" in self.selected_model_name:
            texts = [self.format_for_e5(text) for text in texts]
        else:
            texts = [text for text in texts]

        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        embeddings = np.zeros((len(texts), embedding_dim), dtype="float32")

        num_batches = math.ceil(len(texts) / batch_size)

        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(texts))

            batch_texts = texts[start_idx:end_idx]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            embeddings[start_idx:end_idx] = batch_embeddings

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.index_to_id = {i: movie_ids[i] for i in range(len(movie_ids))}

        self.save(self.faiss_index_file, self.id_mapping_file)
        print(f"FAISS index built with {self.index.ntotal} vectors of dimension {dim}.")

    def search(self, query_text, top_k=15):
        query_emb = self.embedding_model.encode(
            [f"query: {query_text}"], normalize_embeddings=True, convert_to_numpy=True
        )
        D, I = self.index.search(query_emb, top_k)
        return [self.index_to_id[i] for i in I[0]]

    def save(self, index_path, mapping_path):
        faiss.write_index(self.index, index_path)
        with open(mapping_path, "wb") as f:
            pickle.dump(self.index_to_id, f)

        print("Index or mapping file svaed/updated.")

    def load(self, index_path, mapping_path):
        if not os.path.exists(index_path) or not os.path.exists(mapping_path):
            print("No Index or mapping file exists.")
            return

        self.index = faiss.read_index(index_path)
        with open(mapping_path, "rb") as f:
            self.index_to_id = pickle.load(f)

        print("Index or mapping file loaded.")
