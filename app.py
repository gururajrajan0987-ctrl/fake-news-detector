import nltk
import os
nltk.data.path.append(
    os.path.join(os.path.expanduser("~"), "nltk_data")
)
import streamlit as st
import pandas as pd
import string
import nltk
import io
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# ================= CONFIG =================
CREDENTIALS = {"guru": "1234"}

SENSATIONALIST_WORDS = [
    'shocking','unbelievable','miracle','never','secret',
    'bombshell','breaking','fake','hoax','lie',
    'exposed','conspiracy','spy','secretly','uncovered'
]

st.set_page_config(page_title="Fake News Detector", layout="wide")

# ================= NLTK =================
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('vader_lexicon')

# ================= HELPERS =================
def clean_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    sw = set(stopwords.words('english'))
    return ' '.join(w for w in text.split() if w not in sw)

# ================= TRAIN MODEL =================
@st.cache_resource
def train_model():
    true_df = pd.read_csv("True.csv", on_bad_lines="skip")
    fake_df = pd.read_csv("Fake.csv", on_bad_lines="skip")

    true_df["label"] = "Real"
    fake_df["label"] = "Fake"

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df["clean_text"] = df["title"].apply(clean_text)

    df["word_count"] = df["title"].apply(lambda x: len(x.split()))
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["title"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    df["sensationalism_score"] = df["title"].apply(
        lambda x: sum(1 for w in x.lower().split() if w in SENSATIONALIST_WORDS)
    )

    scaler = StandardScaler()
    num = scaler.fit_transform(df[["word_count","sentiment_score","sensationalism_score"]])

    vectorizer = TfidfVectorizer(max_features=5000)
    X_text = vectorizer.fit_transform(df["clean_text"])
    X = hstack([X_text, num])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    cm = confusion_matrix(y_test, model.predict(X_test), labels=["Real","Fake"])

    st.session_state["model_accuracy"] = acc
    st.session_state["confusion_matrix"] = cm

    return model, vectorizer, scaler, acc, df

# ================= LOGIN =================
def login_page():
    st.title("📰 Fake News Detector - Login")
    with st.form("login"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if CREDENTIALS.get(u) == p:
                st.session_state.logged = True
                st.session_state.history = []
                st.rerun()
            else:
                st.error("Invalid credentials")

# ================= DASHBOARD =================
def dashboard_page(df):
    st.title("📊 Dashboard")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{st.session_state['model_accuracy']*100:.2f}%")
    with col2:
        st.metric("Model", "Random Forest")
    with col3:
        st.metric("Total Headlines", len(df))

    st.subheader("📁 Dataset Distribution")
    chart_df = df["label"].value_counts().reset_index()
    chart_df.columns = ["Type","Count"]
    st.bar_chart(chart_df.set_index("Type"))

    st.subheader("📉 Confusion Matrix")
    cm = st.session_state["confusion_matrix"]
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(["Real","Fake"])
    ax.set_yticklabels(["Real","Fake"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)

# ================= SINGLE PREDICTION =================
def single_prediction(model, vectorizer, scaler):
    st.subheader("🔍 Single Prediction")
    text = st.text_area("Enter headline or paragraph")

    if st.button("Predict"):
        analyzer = SentimentIntensityAnalyzer()
        cleaned = clean_text(text)
        wc = len(text.split())
        sent = analyzer.polarity_scores(text)["compound"]
        sens = sum(1 for w in text.lower().split() if w in SENSATIONALIST_WORDS)

        scaled = scaler.transform([[wc, sent, sens]])
        vec = vectorizer.transform([cleaned])
        X = hstack([vec, scaled])

        pred = model.predict(X)[0]
        prob = model.predict_proba(X).max()

        st.session_state.history.append((text, pred, prob))

        if pred == "Fake":
            st.error(f"🧨 FAKE NEWS ({prob:.2%})")
        else:
            st.success(f"✅ REAL NEWS ({prob:.2%})")

# ================= BATCH PREDICTION =================
def batch_prediction(model, vectorizer, scaler):
    st.subheader("📂 Batch Prediction")
    st.info("Upload a CSV or TXT file. Each line / row will be treated as a headline.")

    uploaded_file = st.file_uploader("Upload file", type=["csv", "txt"])

    if uploaded_file is not None:
        # Read file
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            headlines = df.iloc[:, 0].astype(str).tolist()
        else:
            text = uploaded_file.read().decode("utf-8")
            headlines = [line for line in text.splitlines() if line.strip()]

        total = len(headlines)
        st.success(f"✅ {total} headlines loaded")

        if st.button("Run Batch Prediction"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            results = []
            analyzer = SentimentIntensityAnalyzer()

            for i, text in enumerate(headlines):
                # Update progress
                progress = int(((i + 1) / total) * 100)
                progress_bar.progress(progress)
                status_text.text(f"Processing headline {i+1} of {total}")

                cleaned = clean_text(text)
                wc = len(text.split())
                sent = analyzer.polarity_scores(text)["compound"]

                # Disable sensationalism for long text
                if wc > 25:
                    sens = 0
                else:
                    sens = sum(1 for w in text.lower().split() if w in SENSATIONALIST_WORDS)

                scaled = scaler.transform([[wc, sent, sens]])
                vec = vectorizer.transform([cleaned])
                X = hstack([vec, scaled])

                pred = model.predict(X)[0]
                prob = model.predict_proba(X).max()

                results.append({
                    "Headline": text,
                    "Prediction": pred,
                    "Confidence": f"{prob:.2%}"
                })

            progress_bar.progress(100)
            status_text.text("✅ Batch prediction completed")

            result_df = pd.DataFrame(results)
            st.subheader("📊 Batch Prediction Results")
            st.dataframe(result_df, use_container_width=True)

            # Download option
            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Results as CSV",
                csv_data,
                "batch_predictions.csv",
                "text/csv"
            )

# ================= HISTORY =================
def history_page():
    st.subheader("🕒 Prediction History")
    if st.session_state.history:
        hist_df = pd.DataFrame(
            st.session_state.history,
            columns=["Headline","Prediction","Confidence"]
        )
        st.dataframe(hist_df)
    else:
        st.info("No predictions yet")

# ================= PDF REPORT =================
def download_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    c.drawString(50, 800, "Fake News Detector Report")
    y = 760
    for h, p, conf in st.session_state.history:
        c.drawString(50, y, f"{p} ({conf:.2%}) - {h[:80]}")
        y -= 20
        if y < 50:
            c.showPage()
            y = 800
    c.save()
    buffer.seek(0)
    st.download_button("📄 Download PDF Report", buffer, "report.pdf")

# ================= ABOUT =================
def about_page():
    st.title("ℹ️ About Project")
    st.write("""
    This Fake News Detector uses **Machine Learning + NLP**
    to classify news as Real or Fake.
    
    **Features**
    - TF-IDF
    - Sentiment Analysis
    - Random Forest
    - Batch Prediction
    - PDF Reporting
    
    **Developer:** Guru
    """)

# ================= MAIN =================
def main():
    if "logged" not in st.session_state:
        st.session_state.logged = False

    if not st.session_state.logged:
        login_page()
        return

    model, vectorizer, scaler, acc, df = train_model()

    menu = st.sidebar.radio(
        "Menu",
        ["Dashboard","Single Prediction","Batch Prediction","History","About","Logout"]
    )

    if menu == "Dashboard":
        dashboard_page(df)
    elif menu == "Single Prediction":
        single_prediction(model, vectorizer, scaler)
    elif menu == "Batch Prediction":
        batch_prediction(model, vectorizer, scaler)
    elif menu == "History":
        history_page()
        if st.session_state.history:
            download_pdf()
    elif menu == "About":
        about_page()
    elif menu == "Logout":
        st.session_state.logged = False
        st.rerun()

if __name__ == "__main__":
    main()
