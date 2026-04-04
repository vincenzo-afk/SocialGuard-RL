import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('dark_background')

def generate_curve():
    df = pd.read_csv("training_log.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#1a1d27')
        ax.tick_params(colors='#ccc')
        ax.xaxis.label.set_color('#90caf9')
        ax.yaxis.label.set_color('#90caf9')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2d3044')

    # Plot 1: TP vs FP
    ax1.plot(df["episode"], df["tp_rate"], 'o-', color="#4caf50", linewidth=3, markersize=8, label="True Positive Rate (↑)")
    ax1.plot(df["episode"], df["fp_rate"], 'o-', color="#ef5350", linewidth=3, markersize=8, label="False Positive Rate (↓)")
    ax1.set_title("NEMESIS-RL: Detection Performance", fontsize=16, fontweight='bold', pad=15)
    ax1.set_xlabel("Training Episodes", fontweight='bold')
    ax1.set_ylabel("Rate", fontweight='bold')
    ax1.legend(loc="center right", facecolor="#1a1d27", edgecolor="#2d3044", fontsize=12)
    ax1.set_ylim(0, 1.05)
    
    # Plot 2: Entropy
    ax2.plot(df["episode"], df["policy_entropy"], 'o-', color="#ce93d8", linewidth=3, markersize=8, label="Policy Entropy (↓)")
    ax2.set_title("NEMESIS-RL: Policy Convergence", fontsize=16, fontweight='bold', pad=15)
    ax2.set_xlabel("Training Episodes", fontweight='bold')
    ax2.set_ylabel("Entropy", fontweight='bold')
    ax2.legend(loc="upper right", facecolor="#1a1d27", edgecolor="#2d3044", fontsize=12)
    
    plt.tight_layout(pad=3.0)
    plt.savefig("learning_curve.png", dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print("Generated learning_curve.png successfully.")

if __name__ == "__main__":
    generate_curve()
