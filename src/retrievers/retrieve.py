import os
from typing import List
import glob
import json
from dotenv import load_dotenv
from omegaconf import OmegaConf
import faiss
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel
from colpali_engine.models import ColQwen2, ColQwen2Processor
from .base import BaseRetriever
from src.utils import summarize_image

load_dotenv()

bge_config = OmegaConf.load(os.getenv("BGE_CONFIG_PATH"))
colqwen_config = OmegaConf.load(os.getenv("COLQWEN_CONFIG_PATH"))

class BGERetriever(BaseRetriever):
	def __init__(self, device: str = "mps"):
		self.device = device
		self.tokenizer = AutoTokenizer.from_pretrained(bge_config.model_name)
		self.model = AutoModel.from_pretrained(bge_config.model_name).to(self.device)
		self.faiss_index = faiss.read_index(bge_config.faiss_path)
		with open(bge_config.metadata_path, "r", encoding="utf-8") as f:
			self.meta = json.load(f)

	def embed_queries(self, query: str | List[str]) -> torch.tensor:
		if isinstance(query, str):
			query = [query]
		inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(self.device)
		with torch.no_grad():
			outputs = self.model(**inputs)

		embeddings = outputs.last_hidden_state[:, 0, :]

		embeddings = embeddings.float().cpu()
		return embeddings

	def retrieve(self, query: str, top_k: int = 2) -> List[str]:
		query_embedding = self.embed_queries(query)
		top_k_docs = self.faiss_index.search(query_embedding, k=top_k)[1][0]

		metas = [self.meta[i] for i in top_k_docs]
		return [f"{bge_config.images_path}/{meta['pdf']}/{meta['jpeg']}" for meta in metas]

	def _add_image_to_index(self, image_path: str) -> None:
		summary = summarize_image(image_path)
		embedding = self.embed_queries(summary)
		
		self.faiss_index.add(embedding)
		faiss.write_index(self.faiss_index, bge_config.faiss_path)


		pdf_name = image_path.split('/')[-2]
		image = image_path.split('/')[-1]

		self.meta.append({'pdf': pdf_name, 'jpeg': image})

		with open(bge_config.metadata_path, "w", encoding="utf-8") as f:
			json.dump(self.meta, f, ensure_ascii=False)


class ColQwenRetriever:
	def __init__(self, device: str = "mps"):
		self.device = device
		self.model = ColQwen2.from_pretrained(
				colqwen_config.model_name,
				torch_dtype=torch.bfloat16,
				device_map=None,
			)
		self.chunk_size = 500
		self.processor = ColQwen2Processor.from_pretrained(colqwen_config.model_name)
		with open(colqwen_config.metadata_path, "r", encoding="utf-8") as f:
			self.meta = json.load(f)
		self.embeddings = []
		for file in sorted(glob.glob(colqwen_config.embeddings_path + "/*")):
			self.embeddings.extend(torch.load(file))

	def embed_queries(self, query: str | List[str]) -> torch.tensor:
		if isinstance(query, str):
			query = [query]
		batch_queries = self.processor.process_queries(query).to(self.model.device)
		with torch.no_grad():
			outputs = self.model(**batch_queries)
		return outputs.float().cpu()

	def embed_image(self, image: Image.Image) -> torch.tensor:
		batch_images = self.processor.process_images(image).to(self.model.device)
		with torch.no_grad():
			outputs = self.model(**batch_images)
		return outputs.float().cpu()
	
	def retrieve(self, query: str, top_k: int = 2) -> List[str]:
		query_embedding = self.embed_queries(query)
		scores = self.processor.score_multi_vector(query_embedding, self.embeddings)
		top_k_docs = scores.argsort(axis=1)[0][-top_k:][::-1].tolist()
		metas = [self.meta[i] for i in top_k_docs]

		return [f"{colqwen_config.images_path}/{meta['pdf']}/{meta['jpeg']}" for meta in metas]
	
	def _add_image_to_index(self, image_path):
		try:
			with open(image_path, "rb") as f:
				img = Image.open(f)
				embedding = self.embed_image(img)
				self.embeddings.append(embedding)
				self._save_embeddings()
				self.meta.append({'pdf': image_path.split('/')[-2], 'jpeg': image_path.split('/')[-1]})
				with open(colqwen_config.metadata_path, "w", encoding="utf-8") as f:
					json.dump(self.meta, f, ensure_ascii=False)
		except FileNotFoundError:
			print(f"Error: The file {image_path} was not found.")
	
	def _save_embeddings(self):
		for i in range(0, self.embeddings.shape[0], self.chunk_size):
			torch.save(self.embeddings[i:i+self.chunk_size], f"{colqwen_config.embeddings_path}/embeddings_{i}.pt")


class RetrievePipeline:
    def __init__(self, device: str = 'mps'):
        self.bge_retriever = BGERetriever(device=device)
        self.colqwen_retriever = ColQwenRetriever()

    def retrieve(self, query: str, strategy: str = "ColQwen+SummaryEmb"):
        if strategy == "SummaryEmb":
            images = self.bge_retriever.retrieve(query)
        elif strategy == "ColQwen":
            images = self.colqwen_retriever.retrieve(query)
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