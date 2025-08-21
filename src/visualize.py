from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def _prepare(out_path: str):
    """Ensure the output directory exists."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

def bar_missing(counts, title, out_path: str):
    """Plot a bar chart of missing value counts, or 'No Missing Values' message."""
    counts_nonzero = counts[counts > 0]
    plt.figure()

    if counts_nonzero.empty:
        # Draw a message instead of bar chart
        plt.text(0.5, 0.5, "No Missing Values", fontsize=14, ha='center', va='center')
        plt.title(title)
        plt.axis('off')
        _prepare(out_path)
        plt.savefig(out_path)
        plt.close()
        print(f"[INFO] Saved '{title}' chart with 'No Missing Values' message: {out_path}")
        return

    # Plot normal bar chart
    counts_nonzero.plot(kind='bar')
    plt.title(title)
    plt.ylabel('Missing values')
    plt.tight_layout()
    _prepare(out_path)
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved missing values plot: {out_path}")

def plot_survival_by_gender(df, out_path: str):
    """Plot survival rate by gender."""
    if 'Survived' not in df.columns or 'Sex' not in df.columns:
        print("[WARNING] Columns 'Survived' or 'Sex' not found. Skipping survival plot.")
        return
    plt.figure()
    df.groupby('Sex')['Survived'].mean().plot(kind='bar')
    plt.title('Survival Rate by Sex')
    plt.ylabel('Rate')
    plt.tight_layout()
    _prepare(out_path)
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved survival by gender plot: {out_path}")

def plot_confusion_matrix(cm, classes, out_path: str):
    """Plot a confusion matrix."""
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes)
    plt.yticks(ticks, classes)

    # Add text labels
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", color="black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    _prepare(out_path)
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved confusion matrix plot: {out_path}")