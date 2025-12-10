# �️ FraudGuard AI: ETL + RAG + ML Fraud Detection

**Advanced Healthcare Fraud Detection System powered by Generative AI.**  
Transforming raw claims data into actionable intelligence using ETL pipelines, Machine Learning, and Retrieval-Augmented Generation (RAG).

![FraudGuard AI Banner](https://img.shields.io/badge/AI--Powered-Fraud%20Detection-blueviolet?style=for-the-badge)  
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red?logo=streamlit&style=flat-square)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?logo=langchain&style=flat-square)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-orange?logo=google&style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-blue?logo=scikit-learn&style=flat-square)

<br>

## 🚀 Features

- �️‍♂️ **Hybrid Fraud Detection:** Combines rule-based logic with ML models to flag suspicious claims.
- 💬 **AI Fraud Investigator:** Chat naturally with your data using Google Gemini-powered agents ("Show me suspicious cardiology claims").
- 🧠 **Semantic Search (RAG):** Find relevant historical fraud cases by meaning, not just keywords (e.g., "duplicate billing patterns").
- � **Interactive Dashboard:** Real-time analytics with Plotly and Streamlit.
- � **Automated ETL Pipeline:** Seamlessly processes raw CSV data (`claims.csv`, `providers.csv`) into analysis-ready formats.

## 🛠️ Built With

- **Framework**: [Streamlit](https://streamlit.io/)
- **LLM & Agents**: [LangChain](https://www.langchain.com/), [Google Gemini API](https://ai.google.dev/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/) (Isolation Forest / Random Forest)
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss), [Sentence Transformers](https://www.sbert.net/)
- **Data Engineering**: Pandas, NumPy

## 📸 Interface

> *Visualize your fraud detection workflow with a modern, dark-themed UI.*

*(Run the app to see the interactive dashboard)*

## 🚚 Getting Started

### Prerequisites

- Python 3.8+
- [Google Gemini API Key](https://aistudio.google.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-ur>
   cd Fraud_Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Run the App

```bash
streamlit run Frontend/interface_ultimate.py
```

- The application will open in your browser at `http://localhost:8501`.

## 🎯 Usage

1. **Enter API Key:** Input your Google Gemini API key in the sidebar to enable AI features.
2. **Dashboard Overview:** View top-level metrics, fraud rates by specialty, and recent alerts.
3. **AI Chat:** Ask questions like *"Identify providers with high fraud risk assessments"* to get instant answers.
4. **Semantic Search:** Use the RAG tab to find similar past cases (e.g., *"upcoding in radiology"*).

## 🤝 Contributing

Contributions are welcome! Please feel free to check the [issues page](https://github.com/your-username/repo-name/issues).

## 👨‍💻 Developer

**Kumar Saurav**

## 🙏 Acknowledgements

- [Streamlit Community](https://discuss.streamlit.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [Google AI Studio](https://aistudio.google.com/)
