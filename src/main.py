from src.utils import load_csv, save_csv, missing_counts
from src.cleaner import fill_missing
from src.visualize import bar_missing, plot_survival_by_gender, plot_confusion_matrix
from src.model import train_eval

def main():
    print("=" * 50)
    print("   TITANIC MISSING DATA CLEANER - GUVI PROJECT  ")
    print("=" * 50)

    # 1. Load dataset
    csv_path = input("\nEnter CSV file path (e.g., data/train.csv): ").strip().strip('"').strip("'")
    df, p = load_csv(csv_path)
    print(f"\n[INFO] Dataset loaded from: {p}")
    print(f"[INFO] Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # 2. Missing values BEFORE cleaning
    before = missing_counts(df)
    print("[INFO] Missing values BEFORE cleaning:\n", before)
    bar_missing(before, "Missing Values BEFORE Cleaning", "outputs/figures/missing_before.png")

    # 3. Ask user for fill method
    method = input("\nChoose fill method for numeric columns [mean/median/mode]: ").lower().strip()

    # 4. Clean dataset
    df_clean = fill_missing(df.copy(), method, fill_categorical=True)

    # 5. Missing values AFTER cleaning
    after = missing_counts(df_clean)
    print("\n[INFO] Missing values AFTER cleaning:\n", after)
    bar_missing(after, "Missing Values AFTER Cleaning", "outputs/figures/missing_after.png")

    # 6. Save cleaned dataset
    out_path = save_csv(df_clean, p, out_dir="outputs")
    print(f"\n[INFO] Cleaned dataset saved to: {out_path}")

    # 7. Extra visualization for rubric marks
    plot_survival_by_gender(df_clean, "outputs/figures/survival_by_sex.png")

    # 8. Optional ML step
    run_ml = input("\nRun quick Logistic Regression on 'Survived'? [y/n]: ").lower().strip()
    if run_ml == 'y':
        acc, cm, report = train_eval(df_clean)
        print(f"\n[RESULT] Model Accuracy: {acc:.3f}")
        print("\nClassification Report:\n", report)

        plot_confusion_matrix(cm, classes=['Died', 'Survived'], out_path="outputs/figures/confusion_matrix.png")

        with open("outputs/model_report.txt", "w") as f:
            f.write(f"Accuracy: {acc:.3f}\n\n{report}\n")

        print("[INFO] Model report saved to outputs/model_report.txt")
        print("[INFO] Confusion matrix plot saved to outputs/figures/confusion_matrix.png")

    print("\n[✔] PROCESS COMPLETE — Check the 'outputs/' folder for CSV, charts, and model report.")

if __name__ == "__main__":
    main()