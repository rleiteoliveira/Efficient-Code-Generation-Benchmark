from collections import Counter
from utils.execution import run_code


def run_codet_consensus(candidates: list, tests: list) -> list:
    """
    Executa a lógica do CodeT (Code Termination) baseada em consenso.
    Retorna a lista de candidatos que concordaram com a maioria.
    """
    if not candidates or not tests:
        return candidates

    # Pontuação: Cada vez que um candidato concorda com a maioria, ganha 1 ponto.
    candidate_scores = [0] * len(candidates)

    for test_input in tests:
        outputs = []

        # 1. Executar o teste em TODOS os candidatos
        for code in candidates:
            output = run_code(code, test_input)
            outputs.append(output)

        # 2. Achar o Consenso (Resposta mais comum)
        counts = Counter(outputs)
        if not counts:
            continue

        most_common_output, _ = counts.most_common(1)[0]

        # Se a maioria deu erro (UNDEFINED), ignoramos esse teste (consenso fraco)
        if most_common_output == "UNDEFINED":
            continue

        # 3. Pontuar quem concordou
        for i, output in enumerate(outputs):
            if output == most_common_output:
                candidate_scores[i] += 1

    # 4. Selecionar os vencedores (Quem teve a pontuação máxima)
    max_score = max(candidate_scores) if candidate_scores else 0

    # Se max_score for 0, ninguém passou em nada, retorna tudo (fallback)
    if max_score == 0:
        return candidates

    survivors = []
    for i, code in enumerate(candidates):
        if candidate_scores[i] == max_score:
            survivors.append(code)

    return survivors
