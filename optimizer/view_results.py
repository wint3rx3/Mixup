# optimizer/view_results.py

from optimizer.reinforce_graph import ReinforceGraph

def display_all_generations(max_gen: int = 5):
    graph = ReinforceGraph()

    for gen in range(max_gen):
        nodes = graph.get_generation(gen)
        print(f"\n🔷 Generation {gen} | {len(nodes)} prompts")
        for node in nodes:
            valid = node.get('valid_recall', {})
            recall = valid.get('recall', 0)
            precision = valid.get('precision', 0)
            print(f"- {node['id']} | R: {recall:.2f} | P: {precision:.2f}")
            print(f"  → {node['template']}\n")


def display_topk(k: int = 5, by: str = "valid_recall.recall"):
    graph = ReinforceGraph()
    top_nodes = graph.get_topk(k=k, by=by)
    
    print(f"\n🏆 Top-{k} Prompts by {by}\n")
    for i, node in enumerate(top_nodes):
        recall = node['valid_recall'].get('recall', 0)
        precision = node['valid_recall'].get('precision', 0)
        print(f"{i+1}. {node['id']} | R: {recall:.2f} | P: {precision:.2f}")
        print(f"   {node['template']}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=5, help="상위 몇 개를 볼지")
    parser.add_argument("--by", type=str, default="valid_recall.recall", help="정렬 기준 필드")
    parser.add_argument("--gens", type=int, default=3, help="세대별 출력 최대 개수")

    args = parser.parse_args()

    display_all_generations(args.gens)
    display_topk(k=args.topk, by=args.by)
