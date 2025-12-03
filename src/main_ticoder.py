import json
import os
from core.oracle import Oracle
from core.slm_manager import generate_discriminating_test
from core.codet import run_codet_consensus
from utils.execution import run_code

DATA_FILE = "data/candidates_cache.json"

def main():
    print("Starting TiCoder-SLM Main Loop...")
    
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please run src/generate_candidates.py first.")
        return

    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    total_problems = len(data)
    print(f"Loaded {total_problems} problems.")

    total_candidates_processed = 0
    total_candidates_kept = 0

    results = []

    for task_id, problem_data in data.items():
        print(f"\nProcessing {task_id}...")
        
        problem_docstring = problem_data.get("prompt", "")

        candidates = problem_data['candidates']
        canonical_solution = problem_data['canonical']

        # INJECTION: Add Canonical Solution to Candidates for Validation (Strategy A)
        # This ensures that at least one candidate is correct. If this gets pruned, our logic is wrong.
        candidates_pool = candidates.copy()
        candidates_pool.append(canonical_solution)
        
        initial_count = len(candidates_pool)
        
        generated_tests = generate_discriminating_test(
            problem_docstring, candidates_pool, n=5
        )

        # Fallback se SLM falhar
        if generated_tests == ["0"] or not generated_tests:
            print("SLM failed to generate tests. Skipping.")
            continue

        print(f"SLM Generated Tests: {generated_tests}")
        
        # --- ESTRATÉGIA A: TiCoder (Oracle-Based) ---
        print("\n--- Running TiCoder Strategy ---")
        ticoder_survivors = []
        oracle = Oracle(canonical_solution)

        # Simulando poda sequencial com os testes gerados
        current_candidates = candidates_pool.copy()
        for test_input in generated_tests:
            expected = oracle.evaluate(test_input)
            if expected == "UNDEFINED":
                continue

            pass_list = []
            for code in current_candidates:
                if run_code(code, test_input) == expected:
                    pass_list.append(code)
            current_candidates = pass_list
            if len(current_candidates) <= 1:
                break  # Otimização TiCoder

        ticoder_survivors = current_candidates
        print(f"TiCoder Survivors: {len(ticoder_survivors)}/{initial_count}")
        if canonical_solution in ticoder_survivors:
            print("✅ Canonical Solution Survived (TiCoder)")
        else:
            print("❌ Canonical Solution Pruned (TiCoder Error)")

        # --- ESTRATÉGIA B: CodeT (Consensus-Based) ---
        print("\n--- Running CodeT Strategy ---")
        # CodeT não olha para o Oráculo. Ele olha para o grupo.
        codet_survivors = run_codet_consensus(candidates_pool, generated_tests)

        print(f"CodeT Survivors: {len(codet_survivors)}/{initial_count}")
        if canonical_solution in codet_survivors:
            print("✅ Canonical Solution is in Consensus (CodeT)")
        else:
            print("⚠️ Canonical Solution lost the vote (CodeT Weakness)")

        # --- Comparativo Final do Problema ---
        results.append(
            {
                "task": task_id,
                "ticoder_count": len(ticoder_survivors),
                "codet_count": len(codet_survivors),
                "ticoder_safe": canonical_solution in ticoder_survivors,
                "codet_safe": canonical_solution in codet_survivors,
            }
        )

    # --- Resumo Global ---
    print("\n" + "=" * 40)
    print("      COMPARATIVE RESULTS SUMMARY      ")
    print("=" * 40)
    print(f"{'Task':<15} | {'TiCoder Left':<12} | {'CodeT Left':<10} | {'Winner?':<10}")
    print("-" * 55)

    ticoder_wins = 0
    codet_wins = 0

    for r in results:
        # Critério de vitória simples: Quem filtrou mais (menor count) MANTENDO a segurança (safe)
        t_score = r["ticoder_count"] if r["ticoder_safe"] else 999
        c_score = r["codet_count"] if r["codet_safe"] else 999

        winner = "Tie"
        if t_score < c_score:
            winner = "TiCoder"
            ticoder_wins += 1
        elif c_score < t_score:
            winner = "CodeT"
            codet_wins += 1

        print(
            f"{r['task']:<15} | {r['ticoder_count']:<12} | {r['codet_count']:<10} | {winner}"
        )

    print("-" * 55)
    print(f"TiCoder Wins: {ticoder_wins} | CodeT Wins: {codet_wins}")
    print("=" * 40)


if __name__ == "__main__":
    main()
