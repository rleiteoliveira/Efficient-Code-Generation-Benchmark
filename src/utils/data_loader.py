import json
import gzip
import os


def load_dataset(dataset_name, limit=None):
    """
    Carrega HumanEval ou MBPP e retorna formato unificado.
    """
    problems = []

    # Pega o diretório base do projeto (duas pastas acima deste arquivo)
    # src/utils/data_loader.py -> sobe para src/ -> sobe para Raiz
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    # Caminhos
    paths = {
        # HumanEval: O script baixa o .gz na pasta 'data' da raiz
        "humaneval": os.path.join(base_dir, "data", "HumanEval.jsonl.gz"),
        
        # MBPP: Está na pasta 'datasets' que você copiou
        "mbpp": os.path.join(base_dir, "datasets", "mbpp", "sanitized-mbpp.json")
    }

    # Fallback para MBPP normal se não tiver o sanitized
    if dataset_name == "mbpp" and not os.path.exists(paths["mbpp"]):
        paths["mbpp"] = "datasets/mbpp/mbpp.jsonl"

    file_path = paths.get(dataset_name.lower())
    if not file_path or not os.path.exists(file_path):
        # Se for humaneval, avisa para rodar o generate
        if dataset_name == "humaneval":
            print(
                "HumanEval not found. Running generate_candidates.py will download it."
            )
            return []
        raise FileNotFoundError(
            f"Dataset {dataset_name} not found at {file_path}. Please copy the 'datasets' folder from the team repository."
        )

    print(f"Loading {dataset_name} from {file_path}...")

    open_func = gzip.open if file_path.endswith(".gz") else open

    with open_func(
        file_path,
        "rb" if file_path.endswith(".gz") else "r",
        encoding="utf-8" if not file_path.endswith(".gz") else None,
    ) as f:
        # Se for JSON puro (lista de objetos), carrega tudo de uma vez
        if file_path.endswith(".json"):
            data_items = json.load(f)
        else:
            # Se for JSONL (linhas), carrega linha a linha
            data_items = [json.loads(line) for line in f]

        for i, raw in enumerate(data_items):
            if limit and i >= limit:
                break

            normalized = {}

            if "HumanEval" in dataset_name:  # Lógica HumanEval
                normalized = {
                    "id": raw["task_id"],
                    "prompt": raw["prompt"],
                    "canonical": raw["prompt"] + raw["canonical_solution"],
                    "entry_point": raw["entry_point"],
                }
            else:  # Lógica MBPP (Sanitized ou Normal)
                # MBPP sanitized tem 'source_file', 'task_id', 'prompt', 'code', 'test_list'
                # MBPP normal tem 'text', 'code', 'test_list'

                prompt_text = raw.get("prompt") or raw.get("text")
                code_sol = raw.get("code")

                # O MBPP as vezes não tem a função completa, mas vamos tentar
                normalized = {
                    "id": f"Mbpp/{raw['task_id']}",
                    "prompt": f'"""{prompt_text}"""\n',  # Transforma texto em docstring
                    "canonical": code_sol,
                    "entry_point": None,  # MBPP exige parsing mais chato para achar entry point,
                    # mas para o nosso TiCoder que usa "run_code" genérico, talvez passe.
                }

            problems.append(normalized)

    return problems
