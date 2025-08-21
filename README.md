# 🚢 Titanic Missing Data Cleaner  

A Python-based tool to **detect and clean missing data** in the Titanic dataset (or any dataset) and quickly train a **Logistic Regression model** for survival prediction.  

This project was built as a **data wrangling and preprocessing pipeline**:  
- Identifies missing values  
- Cleans them using **mean, median, or mode**  
- Saves the cleaned dataset  
- Runs a **quick Logistic Regression model** for accuracy evaluation  
- Generates **plots and reports** for analysis  

---

## 📂 Project Structure

titanic-missing-data-cleaner/
- └── README.md
- ├── data/ # Raw dataset (train.csv)
- ├── outputs/ # All generated outputs
- │ ├── figures/ # Plots (missing values, confusion matrix, etc.)
- │ ├── train_cleaned.csv
- │ └── model_report.txt
- ├── src/ # Source code
- │ ├── init.py
- │ ├── cleaner.py # Missing data handling
- │ ├── main.py # Entry point
- │ ├── model.py # Logistic Regression training
- │ ├── utils.py # Helper functions
- │ └── visualize.py # Visualization functions
- ├── requirements.txt # Project dependencies
- ├── .gitignore



---

## 🚀 Features  

- Clean raw datasets (handle missing values, outliers, duplicates, etc.)  
- Visualize data before & after cleaning  
- Train machine learning models on cleaned data  
- Generate evaluation metrics (confusion matrix, accuracy, etc.)  
- Save outputs automatically in `outputs/`  

---

## 🛠 Installation  

1. Clone the repository:  
   ```bash
   git clone https://github.com/mdTanveer243/Titanic-Missing-Data.git
   cd https://github.com/mdTanveer243/Titanic-Missing-Data


# ▶️ Usage

Run the pipeline with:

`python src/main.py`


**This will:**

- Clean `data/train.csv`

- Save `cleaned_train.csv` in `outputs/`

- Train the ML model and generate performance visualizations

# 📊 Example Outputs

- EDA before/after cleaning: `outputs/eda_before_after.png`

- Cleaned dataset: `outputs/cleaned_train.csv`

- Confusion matrix: `outputs/confusion_matrix.png`



# 👨‍💻 Author

Your Name

Email: `tanveersiddqui243@gmail.com`

GitHub: `mdtanveer243`
