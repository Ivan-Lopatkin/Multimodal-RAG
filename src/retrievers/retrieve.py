import json
import gzip
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from byaldi import RAGMultiModalModel
import os


class ColQwenRetriever:
    def __init__(self, index_name='index_colqwen', index_root='data', device='cpu'):
        self.index_name = index_name
        self.index_root = index_root
        self.device = device

        self.rag_model = RAGMultiModalModel.from_index(self.index_name, index_root=self.index_root, device=self.device) # создание

    def map_ids(self, mapping_path, results):
        with gzip.open(mapping_path, mode='rt', encoding='utf-8') as f:
            mapping = json.loads(f.read())

        mapping = {k: v.split('/')[-1] for k, v in mapping.items()}

        results_new = ['data/images/' + mapping[str(res['doc_id'])].replace('.pdf', '') + '/' + mapping[str(res['doc_id'])].replace('.pdf', '') + "_page" + str(res['page_num']) + '.jpg' for res in results]
        return results_new

    def retrieve(self, query, k=10):
        results = self.rag_model.search(query, k=k)  # ретрив (топ к)
        img_paths = self.map_ids(self.index_root + '/' + self.index_name +"/doc_ids_to_file_names.json.gz", results)  # маппинг
        
        return img_paths


class BGERetriever:
    def __init__(self, device: str = "cpu"):
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        model_name = "BAAI/bge-m3"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.faiss_index = faiss.read_index('data/index_BGE/faiss_index.bin')
        with open("data/index_BGE/docs_meta.json", "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def embed_query(self, query: str) -> np.ndarray:

        inputs = self.tokenizer(
            query, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings = embeddings.cpu().numpy()
        return embeddings

    def retrieve(self, query: str, top_k: int = 10):
        query_embedding = self.embed_query(query)
        top_k_docs = self.faiss_index.search(query_embedding, k=top_k)[1][0]

        metas = [self.meta[i] for i in top_k_docs]
        return [f"data/images/{meta['pdf']}/{meta['jpeg']}" for meta in metas]


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
            bge_images = self.bge_retriever.retrieve(query)
            images = list(set(colqwen_images) & set(bge_images))
        return images
