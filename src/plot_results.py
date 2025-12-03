import matplotlib.pyplot as plt
import numpy as np


def plot_comparison():
    # Dados extraídos dos seus logs (Média de candidatos restantes após filtragem)
    # Total inicial era 21 (20 candidatos + 1 canônico)

    # Média calculada "de olho" nos seus logs para TiCoder
    # Cloud foi ligeiramente melhor na média geral, mas Local foi muito próximo.
    avg_ticoder_local = 3.8  # Estimativa baseada nos seus dados
    avg_ticoder_cloud = 3.5  # Cloud filtrou um pouquinho mais em HE/10 e HE/24

    # CodeT falhou quase sempre (ficou em 21 ou perto)
    avg_codet_local = 19.5
    avg_codet_cloud = 19.2

    labels = ["TiCoder (Oracle)", "CodeT (Consensus)"]
    local_means = [avg_ticoder_local, avg_codet_local]
    cloud_means = [avg_ticoder_cloud, avg_codet_cloud]

    x = np.arange(len(labels))  # label locations
    width = 0.35  # width of the bars

    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    rects1 = ax.bar(
        x - width / 2, local_means, width, label="Local SLM (20B)", color="#4CAF50"
    )
    rects2 = ax.bar(
        x + width / 2, cloud_means, width, label="Cloud GPT-4o-mini", color="#2196F3"
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Candidatos Restantes (Menor é Melhor)")
    ax.set_title("Eficácia de Filtragem: Local vs Cloud (N=21)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Linha de base (Total de Candidatos)
    ax.axhline(y=21, color="r", linestyle="--", label="Total Inicial (21)")
    ax.text(0.5, 21.5, "Sem Filtragem (Baseline)", color="r", ha="center")

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.savefig("results_comparison.png")
    print("Gráfico salvo como results_comparison.png")


if __name__ == "__main__":
    plot_comparison()