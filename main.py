import asyncio
import os
import aiohttp
from dotenv import load_dotenv
from config import SAMPLE_SIZE, TOP_K, N_CHILDREN

from prompts.base_templates import BASE_TEMPLATES
from optimizer.async_runner import run_all
from optimizer.evaluator import evaluate, extract_error_patterns  
from optimizer.reinforce_graph import ReinforceGraph, PromptNode
from prompts.rag_prompt_generator import generate_child_prompts

load_dotenv()

def ensure_graph_exists(graph: ReinforceGraph):
    if not graph.nodes:
        print("📌 No graph found. Initializing with base templates...")
        for t in BASE_TEMPLATES:
            node = PromptNode(template=t, generation=0)
            graph.nodes[node.id] = node
        graph.save_to_file()

async def main_loop(
    n_children: int = N_CHILDREN,
    top_k: int = TOP_K,
    sample_size: int = SAMPLE_SIZE
):
    # Step 1. Load or initialize graph
    graph = ReinforceGraph()
    graph.load_from_file()
    ensure_graph_exists(graph)

    # Step 2. Run base templates on sample data
    print("\n🚀 Running LLM experiments...")

    latest_gen = max(node.generation for node in graph.nodes.values())

    templates = [
        {
            "id": node.id,
            "template": node.template,
            "type": node.type,
            "description": node.description
        }
        for node in graph.nodes.values()
        if node.generation == latest_gen
    ]

    await run_all(limit=sample_size, templates=templates)

    # Step 3. Evaluate predictions and score
    print("\n📊 Evaluating results...")
    memory = evaluate()

    print("\n📋 템플릿 평가 요약:")
    for m in memory:
        print(f" - {m['template_id']} | eval_count={m['eval_count']} | "
              f"recall={m['avg_recall']} | precision={m['avg_precision']} | lcs={m['avg_lcs']}")

    # Step 3.5. Extract failure patterns (RAG 문서용)
    print("\n📎 Extracting failure patterns...")
    extract_error_patterns()

    # Step 4. Select top-k templates
    print(f"\n🌟 Selecting top-{top_k} templates for next generation...")
    top_nodes = graph.top_k_from_memory("data/memory.jsonl", k=top_k)
    for node in top_nodes:
        print(f" - {node.id} (generation {node.generation})")

    # Step 5. Generate child prompts using RAG
    print("\n🧬 Generating child templates...")
    async with asyncio.Semaphore(4):
        async with aiohttp.ClientSession() as session:
            children = await generate_child_prompts(session, top_nodes, n_children=n_children)

    print(f"\n🌱 자녀 템플릿 생성: 총 {len(children)}개")
    if children:
        print("📌 부모 → 자식 목록:")
        for child in children:
            print(f" - {child['description']} → {child['template'][:60]}...")
    else:
        print("⚠️ 생성된 자녀 템플릿이 없습니다.")

    # Step 6. Add children to graph
    graph.add_generation(top_nodes, children)

    latest_generation = max(node.generation for node in graph.nodes.values())
    print(f"\n📈 현재 그래프 세대 수: {latest_generation}")
    print("✅ One generation complete.")

if __name__ == "__main__":
    asyncio.run(main_loop())
