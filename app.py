import os
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = 'fake_job_model.pkl'
VECTORIZER_FILE = 'tfidf_vectorizer.pkl'
DATASET_FILE = 'fake_job_postings.csv'

# Page Configuration
st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def load_or_train_model():
    """Loads the model if it exists, otherwise trains it using the dataset."""
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer

    # If model files don't exist, check for dataset and train
    if not os.path.exists(DATASET_FILE):
        return None, None

    df = pd.read_csv(DATASET_FILE)
    text_fields = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    for col in text_fields:
        df[col] = df[col].fillna('')

    df['combined_text'] = (
        df['title'] + " " + 
        df['company_profile'] + " " + 
        df['description'] + " " + 
        df['requirements'] + " " + 
        df['benefits']
    )

    X = df['combined_text']
    y = df['fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train_tfidf, y_train)

    # Save for future use
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    return model, vectorizer

# App Header
st.title("🔍 Fake Job Posting Detection System")
st.markdown("Paste the job title and description below to check whether the listing is **Legitimate** or **Fraudulent**.")

# Load model and vectorizer
with st.spinner("Loading model and assets... Please wait."):
    model, vectorizer = load_or_train_model()

if model is None or vectorizer is None:
    st.error(f"⚠️ Dataset file ('{DATASET_FILE}') not found in the directory!")
    st.info("Please download the 'Real or Fake Job Posting Prediction' dataset from Kaggle and place it in the same folder as this script, then restart the app.")
else:
    # User Inputs
    st.markdown("---")
    job_title = st.text_input("Job Title", placeholder="e.g., Data Entry Clerk / Python Developer")
    job_description = st.text_area("Job Description / Details", height=200, placeholder="Paste the full job description here...")

    # Detect Button
    if st.button("Detect Job", type="primary", use_container_width=True):
        if not job_title.strip() or not job_description.strip():
            st.warning("⚠️ Please fill in both the Job Title and Job Description before analyzing.")
        else:
            # Combine input to match training format
            sample_text = [job_title + " " + job_description]
            sample_tfidf = vectorizer.transform(sample_text)
            
            prediction = model.predict(sample_tfidf)
            probability = model.predict_proba(sample_tfidf)
            confidence = probability[0][prediction[0]] * 100

            st.markdown("### Analysis Results")
            if prediction[0] == 1:
                st.error(f"🚨 **Fraudulent (Fake) Job Posting**\n\nConfidence: **{confidence:.2f}%**")
                st.write("⚠️ *Caution: This posting exhibits patterns commonly associated with job scams (such as unrealistic salary promises, upfront fees, or vague requirements).*")
            else:
                st.success(f"✅ **Legitimate (Real) Job Posting**\n\nConfidence: **{confidence:.2f}%**")
                st.write("✔️ *This posting appears consistent with standard, legitimate employment listings.*")