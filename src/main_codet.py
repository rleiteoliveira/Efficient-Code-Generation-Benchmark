import json
import os
from core.slm_manager import generate_discriminating_test
from core.codet import run_codet_consensus
# Note que NÃO importamos o Oracle para uso na lógica, apenas a solução canônica para validação final.

DATA_FILE = "data/candidates_cache.json"


def main():
    print("Starting CodeT (Consensus) Main Loop...")

    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    total_problems = len(data)
    print(f"Loaded {total_problems} problems.")

    total_candidates_processed = 0
    total_candidates_kept = 0

    for task_id, problem_data in data.items():
        print(f"\n{'=' * 40}")
        print(f"Processing {task_id} (CodeT Strategy)...")

        problem_docstring = problem_data.get("prompt", "")
        candidates = problem_data["candidates"]
        canonical_solution = problem_data["canonical"]

        # Prepara o pool. Injetamos o canônico apenas para saber se o CodeT vai eliminá-lo (erro) ou mantê-lo (acerto).
        # Em um cenário real de CodeT, o canônico não existiria, mas precisamos dele para calcular sua métrica de acurácia.
        candidates_pool = candidates.copy()
        candidates_pool.append(canonical_solution)

        initial_count = len(candidates_pool)
        total_candidates_processed += initial_count

        # 1. Gerar Testes (O CodeT precisa de inputs para verificar o consenso)
        generated_tests = generate_discriminating_test(
            problem_docstring, candidates_pool, n=5
        )

        if generated_tests == ["0"] or not generated_tests:
            print("SLM failed to generate tests. Skipping pruning.")
            total_candidates_kept += initial_count
            continue

        print(f"  SLM Generated Tests: {generated_tests}")

        # 2. Executar Lógica de Consenso (CodeT)
        # AQUI ESTÁ A MÁGICA: Não olhamos o gabarito. A maioria vence.
        survivors = run_codet_consensus(candidates_pool, generated_tests)

        final_count = len(survivors)
        total_candidates_kept += final_count

        print(f"  CodeT Survivors: {final_count}/{initial_count}")

        # 3. Validação (Apenas informativo)
        if canonical_solution in survivors:
            print("  ✅ SUCCESS: Canonical Solution is in the consensus.")
        else:
            print("  ⚠️ FAILURE: Canonical Solution was rejected by the group.")

        # Análise extra: Se sobrou alguém mas o canônico morreu, o CodeT "alucinou" em grupo.
        if final_count > 0 and canonical_solution not in survivors:
            print("     -> The model hallucinated a wrong consensus!")

    # --- Resumo Estatístico ---
    print("\n" + "=" * 40)
    print("       CODET EXECUTION SUMMARY       ")
    print("=" * 40)
    print(f"Total Problems:     {total_problems}")
    print(f"Initial Candidates: {total_candidates_processed}")
    print(f"Final Candidates:   {total_candidates_kept}")

    reduction_rate = 0
    if total_candidates_processed > 0:
        reduction_rate = (
            (total_candidates_processed - total_candidates_kept)
            / total_candidates_processed
        ) * 100

    print(f"Reduction Rate:     {reduction_rate:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()
