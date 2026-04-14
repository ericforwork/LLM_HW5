import os
import json
import functools
import numpy as np
from typing import Any, Type, List
from crewai.tools import BaseTool
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

class RobustJSONSearchToolSchema(BaseModel):
    class Config:
        extra = "allow"
    search_query: Any = Field(None, description="The search query string.")

class SimpleLocalSearchTool(BaseTool):
    """A truly local search tool that doesn't use buggy external RAG libraries."""
    name: str = "robust_search_tool"
    description: str = "Search tool for local data"
    data_path: str = ""
    docs: List[Any] = []
    embeddings: Any = None
    model: Any = None
    args_schema: Any = RobustJSONSearchToolSchema

    def __init__(self, data_path: str, name: str, description: str):
        super().__init__()
        self.data_path = data_path
        self.name = name
        self.description = description
        self._load_data()

    def _load_data(self):
        print(f"--- Loading data for {self.name} from {self.data_path} ---", flush=True)
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                if self.data_path.endswith('.jsonl') or "subset.json" in self.data_path:
                    # Handle NDJSON/JSONL if needed, but our files seem to be standard JSON lists
                    try:
                        self.docs = json.load(f)
                    except:
                        f.seek(0)
                        self.docs = [json.loads(line) for line in f if line.strip()]
                else:
                    self.docs = json.load(f)
            
            if not isinstance(self.docs, list):
                self.docs = [self.docs]
            
            print(f"--- Loaded {len(self.docs)} documents for {self.name} ---", flush=True)
            # Initialize model
            self.model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            # Pre-embed documents (simple stringification for search)
            # Only embed the first 500 docs for speed if the file is massive, but 6MB should be fine
            texts = [str(doc)[:500] for doc in self.docs]
            self.embeddings = self.model.encode(texts, convert_to_tensor=True)
            print(f"--- Indexed {len(self.docs)} documents for {self.name} ---", flush=True)
        except Exception as e:
            print(f"Error loading local search data: {e}", flush=True)
            self.docs = []

    def _run(self, **kwargs) -> str:
        query = kwargs.get("search_query")
        if not query:
            for v in kwargs.values():
                if isinstance(v, str):
                    query = v
                    break
        if not query: query = str(kwargs)

        print(f"DEBUG: Local Simple Search for query='{query}'", flush=True)
        if not self.docs: return "No data found."

        query_embedding = self.model.encode(query, convert_to_tensor=True)
        # Simple cosine similarity via dot product (normalized)
        from sentence_transformers import util
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=5)[0]
        
        results = []
        for hit in hits:
            idx = hit['corpus_id']
            results.append(self.docs[idx])
        
        res_str = json.dumps(results, indent=2, ensure_ascii=False)
        print(f"DEBUG: Found {len(results)} matches.", flush=True)
        return res_str

class ReviewPrediction(BaseModel):
    stars: float
    text: Any

@CrewBase
class YelpPredictionCrew():
    """YelpPredictionCrew crew"""

    @functools.lru_cache(maxsize=3)
    def _get_local_tool(self, name: str, path: str):
        return SimpleLocalSearchTool(
            data_path=path, 
            name=f"search_{name.lower()}", 
            description=f"Search {name} data by {name.lower()}_id or attributes."
        )

    def robust_user_search(self):
        path = os.path.join(os.path.dirname(__file__), '../../data/user_subset.json')
        return self._get_local_tool("User", path)

    def robust_item_search(self):
        path = os.path.join(os.path.dirname(__file__), '../../data/item_subset.json')
        return self._get_local_tool("Item", path)

    def robust_review_search(self):
        path = os.path.join(os.path.dirname(__file__), '../../data/review_subset.json')
        return self._get_local_tool("Review", path)

    @agent
    def yelp_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['yelp_analyst'],
            tools=[
                self.robust_user_search(),
                self.robust_item_search(),
                self.robust_review_search()
            ],
            verbose=True,
            llm='ollama/phi3'
        )

    @task
    def predict_review_task(self) -> Task:
        return Task(
            config=self.tasks_config['predict_review_task']
            # Removed output_pydantic=ReviewPrediction to allow robust external parsing in main.py
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.yelp_analyst()],
            tasks=[self.predict_review_task()],
            process=Process.sequential,
            verbose=True
        )
