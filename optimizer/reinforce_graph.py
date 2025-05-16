import os
import json
from typing import List, Dict, Optional


class ReinforceGraph:
    def __init__(self, memory_path: str = "runs/memory.jsonl"):  # memory 저장 경로 지정
        self.memory_path = memory_path
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        if not os.path.exists(memory_path):
            with open(memory_path, "w", encoding="utf-8") as f:
                pass  # 빈 파일 생성

    def load_all(self) -> List[Dict]:  # 전체 노드 리스트 로드
        with open(self.memory_path, "r", encoding="utf-8") as f:
            return [json.loads(line.strip()) for line in f if line.strip()]

    def append_nodes(self, nodes: List[Dict]):  # 새 노드들 추가 기록
        with open(self.memory_path, "a", encoding="utf-8") as f:
            for node in nodes:
                f.write(json.dumps(node, ensure_ascii=False) + "\n")

    def get_topk(self, k: int = 5, by: str = "valid_recall.recall") -> List[Dict]:  # 성능 기준으로 상위 K개 노드 반환
        all_nodes = self.load_all()
        scored = []
        for node in all_nodes:
            try:
                keys = by.split(".")
                val = node
                for key in keys:
                    val = val[key]
                scored.append((val, node))
            except:
                continue
        scored.sort(reverse=True, key=lambda x: x[0])
        return [item[1] for item in scored[:k]]

    def find_by_id(self, node_id: str) -> Optional[Dict]:  # 특정 노드 ID 조회
        for node in self.load_all():
            if node.get("id") == node_id:
                return node
        return None

    def get_generation(self, generation: int) -> List[Dict]:  # 특정 세대의 노드들
        return [n for n in self.load_all() if n.get("generation") == generation]