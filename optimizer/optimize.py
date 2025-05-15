import os
import time
import random
from dotenv import load_dotenv
import pandas as pd
from submission.config import ExperimentConfig
from submission.utils.metrics import evaluate_correction
from optimizer.base_templates import BASE_TEMPLATES
from optimizer.async_runner import AsyncExperimentRunner
from optimizer.template_rewriter import generate_child_prompts
from optimizer.prompt_validator import validate_template, count_tokens
from optimizer.reinforce_graph import ReinforceGraph

def compute_reward(node, strategy="f1_penalized"):
    recall = node['valid_recall'].get('recall', 0)
    precision = node['valid_recall'].get('precision', 0)
    token_penalty = 0.01 * count_tokens(node['template'])

    if strategy == "recall_only":
        return recall - token_penalty

    elif strategy == "precision_only":
        return precision - token_penalty

    elif strategy == "f1_penalized":
        if recall + precision == 0:
            f1 = 0
        else:
            f1 = 2 * recall * precision / (recall + precision)
        return f1 - token_penalty

    elif strategy == "recall_precision_avg":
        return 0.5 * (recall + precision) - token_penalty

    else:
        raise ValueError(f"Unknown reward strategy: {strategy}")


def soft_topk_selection(nodes, k, epsilon=0.3):
    """Top-k 중 일부는 랜덤 탐색으로 교체 (epsilon-greedy 방식)"""
    sorted_nodes = sorted(nodes, key=compute_reward, reverse=True)
    selected = sorted_nodes[:k]
    if random.random() < epsilon:
        candidates = sorted_nodes[k:]
        if candidates:
            selected[-1] = random.choice(candidates)
    return selected

def deduplicate_templates(templates):
    seen = set()
    unique = []
    for t in templates:
        content = t["template"].strip()
        if content in seen or count_tokens(content) > 2000:
            continue
        seen.add(content)
        unique.append(t)
    return unique

def main_loop(max_generation=5, top_k=5, n_children=3, toy_size=100, reward_strategy="recall_only"):
    load_dotenv()
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        raise ValueError("API 키가 .env에 없습니다.")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    toy_data = train.sample(n=toy_size, random_state=42).reset_index(drop=True)
    train_data = toy_data.sample(frac=0.8, random_state=42)
    valid_data = toy_data.drop(train_data.index)

    graph = ReinforceGraph()
    runner_cls = AsyncExperimentRunner

    if len(graph.load_all()) == 0:
        print("[Init] 초기 템플릿 등록")
        graph.append_nodes(BASE_TEMPLATES)

    best_score_history = []
    patience = 3

    for gen in range(max_generation):
        print(f"\n=== [GENERATION {gen}] ===")
        current_gen = graph.get_generation(gen)
        evaluated_nodes = []

        for i, node in enumerate(current_gen):
            runner = runner_cls(ExperimentConfig(template_name=node["id"]), api_key, node["template"])
            result = runner.run_template_experiment(train_data, valid_data)

            node["train_recall"] = result["train_recall"]
            node["valid_recall"] = result["valid_recall"]

            recall = node["valid_recall"].get("recall", 0)
            precision = node["valid_recall"].get("precision", 0)
            reward = compute_reward(node, strategy=reward_strategy)

            if reward < 0.01:
                continue  

            print(f"[{i+1}/{len(current_gen)}] 템플릿 ID: {node['id']} | recall: {recall:.2f} | precision: {precision:.2f} | 보상: {reward:.2f}")

            evaluated_nodes.append(node)

        graph.append_nodes(evaluated_nodes)

        rewards = [compute_reward(n, strategy=reward_strategy) for n in evaluated_nodes]
        best = max(rewards)
        avg = sum(rewards) / len(rewards)
        print(f"[GEN {gen}] 평균 reward: {avg:.2f}, 최고 reward: {best:.2f}")
        best_score_history.append(best)

        if gen >= patience and all(
            best_score_history[-1] <= best_score_history[-1 - i]
            for i in range(1, patience + 1)
        ):
            print("🔚 성능 개선 정체로 조기 종료합니다.")
            break

        topk = soft_topk_selection(evaluated_nodes, top_k, epsilon=0.3)

        # 자식 템플릿 생성 확인
        print(f"\n == 자식 템플릿 생성 ==")
        children = generate_child_prompts(topk, n_children, api_key)

        print("\n 생성된 자식 템플릿:")
        for c in children:
            print(f"- {c['id']} | 부모: {c['parent_ids']}")
            print(f"  템플릿: {c['template']}\n")

        deduped = deduplicate_templates(children)

        # 유효성 검사 로그 추가
        print("\n == 유효성 검사 시작 ==")
        filtered = []
        for c in deduped:
            if validate_template(c["template"], verbose=True):
                filtered.append(c)

        print(f"[GEN {gen+1}] 유효한 자식 수: {len(filtered)}개")
        graph.append_nodes(filtered)

        if not filtered:
            print("⚠️ 유효한 자식 템플릿이 없어 종료합니다.")
            break

if __name__ == "__main__":
    main_loop(
        max_generation=3,       # 루프 3세대
        top_k=3,                # 상위 3개 선택
        n_children=3,           # 자식 3개씩 생성
        toy_size=10,            # 평가용 샘플 10개
        reward_strategy="f1_penalized"  # 복합 점수 활용
    )