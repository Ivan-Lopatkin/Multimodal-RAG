from abc import ABC, abstractmethod
from typing import List
import torch

class BaseRetriever(ABC):

	@abstractmethod
	def retrieve(self, query: str | List[str], k: int = 2):
		"Method for retrieval nearest items from index for each query"
		pass
	
	@abstractmethod
	def _add_image_to_index(self, image_path: str) -> None:
		"Add image emdedding to vector store"
		pass
	
	@abstractmethod
	def embed_queries(query: str | List[str]) -> torch.tensor:
		"Embeds query/queries"
		pass 