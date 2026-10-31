# 🔍 Fake Job Posting Detection System

An end-to-end Machine Learning project designed to detect fraudulent or scam job listings using Natural Language Processing (NLP) and modern classification techniques. Built with Python and Scikit-Learn, and deployed as an interactive web application via Streamlit.

---

## 📌 Project Overview

Online job scams are increasingly common, targeting job seekers with fake offers to steal personal information or solicit upfront money under the guise of "training kits" or "processing fees." 

This project addresses the problem by analyzing the textual content of job posts (titles, descriptions, requirements, and benefits) to classify them as either **Legitimate (Real)** or **Fraudulent (Fake)** with a high confidence score.

---

## 📊 Dataset Used

- **Dataset Name:** Real or Fake Job Posting Prediction (EMSAD - Employment Scam Aegean Dataset)
- **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Total Records:** ~17,880 job postings
- **Target Variable:** `fraudulent` (`0` for Legitimate, `1` for Fraudulent)
- **Key Features Used:**
  - `title`: The job position title.
  - `company_profile`: Overview of the hiring organization.
  - `description`: Detailed responsibilities and tasks.
  - `requirements`: Qualifications and experience needed.
  - `benefits`: Perks and compensations offered.

---

## 🛠️ Machine Learning Model & Pipeline

1. **Preprocessing & Feature Engineering:**
   - Missing text values (`NaN`) are filled with empty strings.
   - Textual attributes (`title`, `company_profile`, `description`, `requirements`, and `benefits`) are merged into a single comprehensive feature text column.

2. **Feature Extraction (TF-IDF Vectorization):**
   - Utilizes **Term Frequency-Inverse Document Frequency (TF-IDF)** to convert unstructured text into numerical features.
   - Includes unigrams and bigrams (`ngram_range=(1, 2)`) with a limit of 5,000 top features and standard English stop-words removed.

3. **Classification Model (Random Forest Classifier):**
   - **Algorithm:** `RandomForestClassifier` with 100 decision trees (`n_estimators=100`).
   - **Class Imbalance Handling:** Set `class_weight='balanced'` to penalize misclassifications of the rare minority class (fraudulent postings represent $< 5\%$ of the total dataset).

---

## 📦 Tech Stack & Libraries Used

- **Language:** Python 3.8+
- **Machine Learning & NLP:** `scikit-learn`
- **Data Manipulation:** `pandas`, `numpy`
- **Model Serialization:** `joblib`
- **User Interface / Web App:** `streamlit`

---

## 🚀 Getting Started Locally

### 1. Prerequisites & Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/fake-job-posting-detection.git
cd fake-job-posting-detection
pip install -r requirements.txt
```

### 2. Add Dataset
Download `fake_job_postings.csv` from Kaggle and place it in the project root directory.

### 3. Run the Streamlit Web Application

```bash
streamlit run streamlit_app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
├── fake_job_postings.csv      # Kaggle Dataset (Not tracked by git)
├── streamlit_app.py           # Main Streamlit UI & ML Pipeline logic
├── fake_job_model.pkl         # Saved Random Forest model (Generated on first run)
├── tfidf_vectorizer.pkl       # Saved TF-IDF Vectorizer (Generated on first run)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 💡 Key Features of the App

- **Interactive GUI:** Easy-to-use inputs for job title and job description.
- **Real-Time Classification:** Instant analysis flagging postings as **Legitimate** or **Fraudulent**.
- **Confidence Scoring:** Shows the probability percentage behind each prediction.
- **Auto-Training System:** Automatically trains and caches model assets on the first run if pre-trained `.pkl` files are absent.