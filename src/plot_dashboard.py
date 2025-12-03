import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Configuração de estilo para gráficos bonitos
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

HISTORY_FILE = "data_history/benchmark_history.json"
OUTPUT_DIR = "results_charts"


def load_detailed_data():
    if not os.path.exists(HISTORY_FILE):
        print("Nenhum histórico encontrado.")
        return pd.DataFrame()

    with open(HISTORY_FILE, "r") as f:
        data = json.load(f)

    # Explodir os detalhes para ter uma linha por problema por modelo
    detailed_rows = []
    for run in data:
        model = run["model_name"]
        dataset = run["dataset"]
        for detail in run["details"]:
            # Calcular redução para este problema específico
            init = detail["initial_count"]
            ti_red = (init - detail["ticoder_final"]) / init * 100
            co_red = (init - detail["codet_final"]) / init * 100

            detailed_rows.append(
                {
                    "Model": model,
                    "Dataset": dataset,
                    "TaskID": detail["task_id"],
                    "Strategy": "TiCoder",
                    "Remaining Candidates": detail["ticoder_final"],
                    "Reduction Rate (%)": ti_red,
                    "Canonical Survived": detail.get(
                        "ticoder_safe", True
                    ),  # Default True para compatibilidade
                }
            )
            detailed_rows.append(
                {
                    "Model": model,
                    "Dataset": dataset,
                    "TaskID": detail["task_id"],
                    "Strategy": "CodeT",
                    "Remaining Candidates": detail["codet_final"],
                    "Reduction Rate (%)": co_red,
                    "Canonical Survived": detail.get("codet_safe", True),
                }
            )

    return pd.DataFrame(detailed_rows)


def plot_charts():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_detailed_data()
    if df.empty:
        return

    # Filtrar apenas HumanEval por enquanto (pode mudar depois)
    df_he = df[df["Dataset"] == "humaneval"]
    if df_he.empty:
        print("Sem dados de HumanEval para plotar.")
        return

    print(f"Gerando gráficos para {df_he['Model'].nunique()} modelos...")

    # --- GRÁFICO 1: Eficiência Média de Redução (Barras) ---
    plt.figure()
    # Calcula a média por modelo e estratégia
    avg_reduction = (
        df_he.groupby(["Model", "Strategy"])["Reduction Rate (%)"].mean().reset_index()
    )

    chart1 = sns.barplot(
        data=avg_reduction,
        x="Model",
        y="Reduction Rate (%)",
        hue="Strategy",
        palette=["#1f77b4", "#ff7f0e"],  # Azul TiCoder, Laranja CodeT
    )
    plt.title("1. Eficiência Média de Filtragem (Maior é Melhor)")
    plt.ylabel("Taxa Média de Redução (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart1_avg_reduction.png"))
    print("✅ Gráfico 1 salvo: Eficiência Média")

    # --- GRÁFICO 2: Taxa de Segurança (Sobrevivência do Gabarito) ---
    plt.figure()
    # Calcula % de vezes que 'Canonical Survived' foi True
    safety_rate = (
        df_he.groupby(["Model", "Strategy"])["Canonical Survived"].mean() * 100
    )
    safety_rate = safety_rate.reset_index()
    safety_rate.rename(columns={"Canonical Survived": "Safety Rate (%)"}, inplace=True)

    chart2 = sns.barplot(
        data=safety_rate,
        x="Model",
        y="Safety Rate (%)",
        hue="Strategy",
        palette=["#1f77b4", "#ff7f0e"],
    )
    plt.title("2. Taxa de Segurança (Gabarito Sobreviveu?)")
    plt.ylabel("Porcentagem de Segurança (%)")
    plt.ylim(80, 105)  # Zoom no topo, pois esperamos valores altos
    plt.axhline(100, color="green", linestyle="--", alpha=0.5, label="Ideal (100%)")
    plt.xticks(rotation=45)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart2_safety_rate.png"))
    print("✅ Gráfico 2 salvo: Taxa de Segurança")

    # --- GRÁFICO 3: Distribuição de Candidatos Restantes (Boxplot) ---
    plt.figure()
    # Boxplot mostra a mediana, quartis e outliers. Ótimo para ver consistência.
    chart3 = sns.boxplot(
        data=df_he,
        x="Model",
        y="Remaining Candidates",
        hue="Strategy",
        palette=["#1f77b4", "#ff7f0e"],
        showfliers=True,  # Mostra pontos fora da curva (outliers)
    )
    plt.title(
        "3. Distribuição de Candidatos Restantes (Menor e Mais Compacto é Melhor)"
    )
    plt.ylabel("Número Absoluto de Candidatos Restantes")
    # Adiciona linha do ideal (apenas 1 restante)
    plt.axhline(1, color="red", linestyle=":", label="Ideal (Apenas Gabarito)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "chart3_distribution.png"))
    print("✅ Gráfico 3 salvo: Distribuição (Boxplot)")

    print(f"\nTodos os gráficos salvos na pasta '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    plot_charts()
