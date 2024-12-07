import json
import gzip
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel


class ColQwenRetriever:
    def __init__(self):
        self.index_name = 'index'
        self.index_root = '.byaldi'
        self.device = 'cpu'

    def map_ids(self, mapping_path, results):
        with gzip.open(mapping_path, mode='rt', encoding='utf-8') as f:
            mapping = json.loads(f.read())

        mapping = {k: v.split('/')[-1] for k, v in mapping.items()}

        results_new = [mapping[str(res['doc_id'])] + '/' + mapping[str(
            # сделать бы так чтоб возвращало как путь
            res['doc_id'])] + "_page" + str(res['page_num']) + '.jpg' for res in results]
        return results_new

    def retrieve(self, query):
        # results = RAG.search(query/, k=5)  # ретрив (топ к)
        # img_paths = map_ids(index_root + '/' + index_name +
        #                     "/doc_ids_to_file_names.json.gz", results)  # маппинг
        pass


class BGERetriever:
    def __init__(self, device: str = "cpu", top_k: int = 10):
        model_name = "BAAI/bge-m3"
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.faiss_index = faiss.load
        with open("data/index_BGE/docs_meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def embed_query(self, query: str) -> np.ndarray:

        inputs = self.tokenizer(
            query, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].squeeze(0)

        embeddings = embeddings.cpu().numpy().tolist()
        return embeddings

    def retrieve(self, query: str):
        query_embedding = self.embed_query(query)
        top_k_docs = self.faiss_index.search(query_embedding, k=self.top_k)
        metas = [self.meta[i] for i in top_k_docs]
        return [f"{meta["pdf"]}/{meta["jpeg"]}" for meta in metas]


class RetrievePipeline:
    def __init__(self):
        self.bge_retriever = BGERetriever()
        self.colqwen_retriever = ColQwenRetriever()

    def retrieve(self, query: str, strategy: str = "Intersection"):
        if strategy == "SummaryEmb":
            images = self.bge_retriever.retrieve(query)
        elif strategy == "ColQwen":
            images = self.colqwen_retriever.retrieve(query)
        elif strategy == "Intersection":
            colqwen_images = self.colqwen_retriever.retrieve(query)
            bge_emages = self.bge_retriever.retrieve(query)
            # пересечение сделать
        return images
