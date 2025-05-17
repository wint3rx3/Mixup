# optimizer/reinforce_graph.py

import uuid
import json
from typing import List, Dict, Optional
from config import GRAPH_PATH

GRAPH_PATH = "data/prompt_graph.jsonl"

class PromptNode:
    def __init__(self, template: dict, parent_ids: Optional[List[str]] = None, generation: int = 0):
        self.id = template.get("id") or f"node_{uuid.uuid4().hex[:8]}"
        self.template = template["template"]
        self.type = template["type"]
        self.description = template.get("description", "")
        self.parent_ids = parent_ids or []
        self.generation = generation

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "template": self.template,
            "description": self.description,
            "parent_ids": self.parent_ids,
            "generation": self.generation
        }


class ReinforceGraph:
    def __init__(self):
        self.nodes: Dict[str, PromptNode] = {}

    def load_from_file(self, path=GRAPH_PATH):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    node_data = json.loads(line)
                    node = PromptNode(
                        template={"template": node_data["template"], "type": node_data["type"]},
                        parent_ids=node_data.get("parent_ids", []),
                        generation=node_data["generation"]
                    )
                    node.id = node_data["id"]
                    node.description = node_data.get("description", "")
                    self.nodes[node.id] = node
        except FileNotFoundError:
            pass

    def save_to_file(self, path=GRAPH_PATH):
        with open(path, "w", encoding="utf-8") as fout:
            for node in self.nodes.values():
                fout.write(json.dumps(node.to_dict(), ensure_ascii=False) + "\n")

    def add_generation(self, parent_nodes: List[PromptNode], children_templates: List[dict]):
        generation = max((n.generation for n in parent_nodes), default=0) + 1
        for template in children_templates:
            node = PromptNode(template, parent_ids=[p.id for p in parent_nodes], generation=generation)
            self.nodes[node.id] = node
        self.save_to_file()

    def top_k_from_memory(self, memory_path: str, k: int = 3) -> List[PromptNode]:
        with open(memory_path, encoding="utf-8") as f:
            memory = [json.loads(line) for line in f]

        memory.sort(key=lambda x: x["avg_recall"], reverse=True)
        top_ids = [m["template_id"] for m in memory[:k]]

        nodes = []
        for tid in top_ids:
            if tid in self.nodes:
                nodes.append(self.nodes[tid])
            else:
                print(f"⚠️ memory.jsonl에 존재하나 graph에 없는 template_id: {tid}")
        return nodes

