import json
import gzip
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
from byaldi import RAGMultiModalModel
import os
from src.utils import summarize_image

class ColQwenRetriever:
    def __init__(self, index_name='index_colqwen', index_root='data', device='cpu'):
        self.index_name = index_name
        self.index_root = index_root
        self.device = device

        self.rag_model = RAGMultiModalModel.from_index(self.index_name, index_root=self.index_root, device=self.device, verbose=0) # создание

    def map_ids(self, mapping_path, results):
        with gzip.open(mapping_path, mode='rt', encoding='utf-8') as f:
            mapping = json.loads(f.read())

        mapping = {k: v.split('/')[-1] for k, v in mapping.items()}

        results_new = ['data/images/' + mapping[str(res['doc_id'])].replace('.pdf', '') + '/' + mapping[str(res['doc_id'])].replace('.pdf', '') + "_page" + str(res['page_num']) + '.jpg' for res in results]
        return results_new

    def retrieve(self, query, k=2):
        results = self.rag_model.search(query, k=k)  # ретрив (топ к)
        img_paths = self.map_ids(self.index_root + '/' + self.index_name +"/doc_ids_to_file_names.json.gz", results)  # маппинг
        
        return img_paths
    
    def add_to_index(self, pdf_path):
        self.rag_model.add_to_index(pdf_path, store_collection_with_index=False)


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

        embeddings = embeddings.float().cpu().numpy()
        return embeddings

    def retrieve(self, query: str, top_k: int = 2):
        query_embedding = self.embed_query(query)
        top_k_docs = self.faiss_index.search(query_embedding, k=top_k)[1][0]

        metas = [self.meta[i] for i in top_k_docs]
        return [f"data/images/{meta['pdf']}/{meta['jpeg']}" for meta in metas]
    
    def add_to_index(self, pdf_path):
        pdf_name = pdf_path.split('/')[-1].replace('.pdf', '')
        image_folder = 'data/images/' + pdf_name

        for image in os.listdir(image_folder):
            summary = summarize_image(image_folder + '/' + image)

            inputs = self.tokenizer(
                summary, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = outputs.last_hidden_state[:, 0, :]
            self.faiss_index.add(embeddings.cpu())

            self.meta.append({'pdf':pdf_name, 'jpeg': image})

        with open("data/index_BGE/docs_meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False)

        faiss.write_index(self.faiss_index, 'data/index_BGE/faiss_index.bin')
        


class RetrievePipeline:
    def __init__(self, device='cpu'):
        self.bge_retriever = BGERetriever(device=device)
        self.colqwen_retriever = ColQwenRetriever(device=device)

    def retrieve(self, query: str, strategy: str = "ColQwen+SummaryEmb"):
        if strategy == "SummaryEmb":
            images = self.bge_retriever.retrieve(query)
        elif strategy == "ColQwen":
            images = self.colqwen_retriever.retrieve(query)
        #elif strategy == "Intersection":
        #    colqwen_images = self.colqwen_retriever.retrieve(query)
        #    bge_images = self.bge_retriever.retrieve(query)
        #    images = list(set(colqwen_images) & set(bge_images))
        elif strategy == 'ColQwen+SummaryEmb':
            colqwen_images = self.colqwen_retriever.retrieve(query)
            bge_images = self.bge_retriever.retrieve(query)
            top1colqwen = colqwen_images[0]
            top1bge = bge_images[0]
        
            images = list(set([top1colqwen, top1bge]))
        return images
    
    def add_to_index(self, pdf_path):
        self.bge_retriever.add_to_index(pdf_path)
        self.colqwen_retriever.add_to_index(pdf_path)