import json
import os
from core.oracle import Oracle
from core.slm_manager import generate_discriminating_test
from core.codet import run_codet_consensus
from utils.execution import run_code

DATA_FILE = "data/candidates_cache.json"


def run_ticoder_logic(candidates, tests, canonical):
    """Lógica isolada do TiCoder: Filtra pelo Oráculo."""
    oracle = Oracle(canonical)
    current_candidates = candidates.copy()

    for test_input in tests:
        expected = oracle.evaluate(test_input)
        if expected == "UNDEFINED":
            continue

        pass_list = []
        for code in current_candidates:
            if run_code(code, test_input) == expected:
                pass_list.append(code)
        current_candidates = pass_list
        if len(current_candidates) <= 1:
            break
    return current_candidates


def main():
    print("Starting Comparative Benchmark...")

    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    results = []

    for task_id, problem_data in data.items():
        print(f"Benchmarking {task_id}...")
        prompt = problem_data.get("prompt", "")
        candidates = problem_data["candidates"] + [
            problem_data["canonical"]
        ]  # Injeta canônico

        # Geração de Testes (Compartilhada para ser justo)
        tests = generate_discriminating_test(prompt, candidates, n=5)
        if tests == ["0"] or not tests:
            tests = []

        # Roda TiCoder
        ticoder_survivors = run_ticoder_logic(
            candidates, tests, problem_data["canonical"]
        )

        # Roda CodeT
        codet_survivors = run_codet_consensus(candidates, tests)

        # Salva resultados
        results.append(
            {
                "task": task_id,
                "ti_len": len(ticoder_survivors),
                "co_len": len(codet_survivors),
                "ti_safe": problem_data["canonical"] in ticoder_survivors,
                "co_safe": problem_data["canonical"] in codet_survivors,
            }
        )

    # Tabela Final
    print("\n" + "=" * 60)
    print(f"{'Task':<15} | {'TiCoder (N)':<12} | {'CodeT (N)':<12} | {'Winner'}")
    print("-" * 60)

    for r in results:
        # Vencedor: Quem reduziu mais (menor N) mas manteve a segurança (Safe=True)
        t_score = r["ti_len"] if r["ti_safe"] else 999
        c_score = r["co_len"] if r["co_safe"] else 999

        if t_score < c_score:
            winner = "TiCoder"
        elif c_score < t_score:
            winner = "CodeT"
        else:
            winner = "Tie"

        # Adiciona um asterisco (*) se a solução falhou (matou o canônico)
        t_disp = f"{r['ti_len']}" + ("" if r["ti_safe"] else " (FAIL)")
        c_disp = f"{r['co_len']}" + ("" if r["co_safe"] else " (FAIL)")

        print(f"{r['task']:<15} | {t_disp:<12} | {c_disp:<12} | {winner}")
    print("=" * 60)


if __name__ == "__main__":
    main()
