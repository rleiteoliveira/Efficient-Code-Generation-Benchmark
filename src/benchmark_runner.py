import json
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from utils.data_loader import load_dataset
from core.slm_manager import generate_discriminating_test
from core.codet import run_codet_consensus
from core.oracle import Oracle
from utils.execution import run_code

HISTORY_FILE = "data_history/benchmark_history.json"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CACHE = os.path.join(BASE_DIR, "data", "candidates_cache.json")


# Função auxiliar de TiCoder (lógica de corte)
def run_ticoder_logic(candidates, tests, canonical):
    if not canonical:
        return candidates  # Sem gabarito não tem TiCoder
    oracle = Oracle(canonical)
    current = candidates.copy()
    for t in tests:
        exp = oracle.evaluate(t)
        if exp == "UNDEFINED":
            continue
        current = [c for c in current if run_code(c, t) == exp]
        if len(current) <= 1:
            break
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        required=True,
        help="Nome do modelo usado (ex: gpt-4o-mini, local-20b)",
    )
    parser.add_argument("--dataset", default="humaneval", help="humaneval ou mbpp")
    parser.add_argument(
        "--limit", type=int, default=10, help="Quantos problemas testar"
    )
    parser.add_argument(
        "--candidates_file",
        default=DEFAULT_CACHE,
        help="Arquivo JSON com os candidatos gerados (Cache)",
    )
    args = parser.parse_args()

    # 1. Carregar Candidatos (Cache Gerado)
    with open(args.candidates_file, "r") as f:
        cache_data = json.load(f)

    # 2. Loop de Teste
    results_summary = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "dataset": args.dataset,
        "total_problems": 0,
        # Médias serão recalculadas no final
        "ticoder_avg_reduction": 0.0,
        "codet_avg_reduction": 0.0,
        "details": [],
    }

    ticoder_reductions = []
    codet_reductions = []

    print(f"\n--- Iniciando Benchmark: {args.model_name} em {args.dataset} ---")

    # Cabeçalho da Tabela no Terminal
    header = (
        f"| {'Task ID':<15} | {'Init':<5} | {'TiCoder Left':<12} | {'CodeT Left':<10} |"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    # Processar apenas os que temos no cache E no dataset
    count = 0
    # Usamos tqdm aqui para a barra de progresso
    loop = tqdm(
        cache_data.items(), total=min(args.limit, len(cache_data)), desc="Progresso"
    )

    for task_id, data in loop:
        if count >= args.limit:
            break
        count += 1

        prompt = data.get("prompt", "")
        candidates = data["candidates"]
        canonical = data["canonical"]
        pool = candidates + [canonical]
        initial_len = len(pool)

        # A. Gerar Testes (O SLM trabalha aqui)
        # Otimização: Se já tiver testes no cache (de uma rodada anterior), poderia usar.
        # Por enquanto, geramos sempre para garantir que é o modelo atual rodando.
        tests = generate_discriminating_test(prompt, pool, n=5)
        if tests == ["0"] or not tests:
            tests = []

        # B. Rodar Estratégias
        ticoder_surv = run_ticoder_logic(pool, tests, canonical)
        codet_surv = run_codet_consensus(pool, tests)

        # C. Calcular Métricas
        ti_final = len(ticoder_surv)
        co_final = len(codet_surv)

        # Redução %
        ti_red = (initial_len - ti_final) / initial_len
        co_red = (initial_len - co_final) / initial_len

        ticoder_reductions.append(ti_red)
        codet_reductions.append(co_red)

        # Validação de Segurança (Gabarito sobreviveu?)
        ti_safe = canonical in ticoder_surv if canonical else False
        co_safe = canonical in codet_surv if canonical else False

        results_summary["details"].append(
            {
                "task_id": task_id,
                "initial_count": initial_len,
                "ticoder_final": ti_final,
                "codet_final": co_final,
                "ticoder_safe": ti_safe,  # NOVO: Salva se é seguro
                "codet_safe": co_safe,  # NOVO: Salva se é seguro
                "tests_generated": tests,
            }
        )

        # IMPRESSÃO NA TELA (Usando tqdm.write para não quebrar a barra)
        # Adiciona um asterisco (*) se o modelo foi inseguro (matou o gabarito)
        ti_disp = f"{ti_final}" + ("" if ti_safe else "*")
        co_disp = f"{co_final}" + ("" if co_safe else "*")

        status_line = (
            f"| {task_id:<15} | {initial_len:<5} | {ti_disp:<12} | {co_disp:<10} |"
        )
        tqdm.write(status_line)

    loop.close()  # Fecha a barra de progresso corretamente

    # 3. Salvar no Banco de Dados (Histórico)
    if ticoder_reductions:
        results_summary["total_problems"] = count
        results_summary["ticoder_avg_reduction"] = sum(ticoder_reductions) / len(
            ticoder_reductions
        )
        results_summary["codet_avg_reduction"] = sum(codet_reductions) / len(
            codet_reductions
        )

    os.makedirs("data_history", exist_ok=True)

    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except:
            pass

    history.append(results_summary)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

    print(separator)
    print(
        f"✅ Benchmark Salvo! Média de Redução -> TiCoder: {results_summary['ticoder_avg_reduction']:.2%} | CodeT: {results_summary['codet_avg_reduction']:.2%}"
    )
    print("(* = Alerta: Solução canônica foi eliminada)")


if __name__ == "__main__":
    main()
