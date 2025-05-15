import uuid
from typing import List, Dict
from optimizer.rag_prompt_generator import generate_prompts_batch

def generate_child_prompts(parents: List[Dict], n_children: int, api_key: str) -> List[Dict]:
    children = []

    for parent in parents:
        try:
            new_templates = generate_prompts_batch(parent["template"], n=n_children)
            for template in new_templates:
                children.append({
                    "id": f"node_{uuid.uuid4().hex[:8]}",
                    "template": template,
                    "parent_ids": [parent["id"]],
                    "generation": parent["generation"] + 1
                })

        except Exception as e:
            print(f"[RAG Prompt Batch Generation Error] {e}")

    return children
