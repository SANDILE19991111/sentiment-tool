# 🧠 SentimentIQ — AI Sentiment Analysis Tool

> An interactive AI-powered web application that classifies customer reviews and social media comments as **Positive**, **Negative**, or **Neutral** using Natural Language Processing.

**Built by Bongimusa Khoza** 

---

## 🌐 Live Demo

👉 **[View Live App](https://sentiment-tool-hd6pd9vitvfbfjn89jjakf.streamlit.app/)** *(update after deployment)*

---

## 📸 Features

- ⚡ **Real-time sentiment analysis** — instant results as you type or upload
- 📊 **Interactive charts** — bar, pie, and donut chart visualisations
- ☁️ **Word clouds** — separate clouds for positive and negative text
- 📈 **Polarity distribution** — histogram showing sentiment score spread
- 💡 **AI insights** — automatic recommendations based on results
- 📥 **Export options** — download results as CSV or Markdown report
- 📂 **Multiple input types** — sample data, CSV upload, or custom text

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.13 | Core language |
| Streamlit | Web dashboard framework |
| TextBlob | NLP sentiment analysis |
| Plotly | Interactive charts |
| Matplotlib | Static charts & word clouds |
| WordCloud | Text visualisation |
| Pandas | Data processing |

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/SANDILE19991111/sentiment-tool.git
cd sentiment-tool
```

**2. Install dependencies**
```bash
python -m pip install streamlit textblob nltk pandas numpy matplotlib seaborn plotly wordcloud requests
```

**3. Run the app**
```bash
python -m streamlit run app.py
```

**4. Open your browser at** `http://localhost:8501`

---

## 📂 Project Structure

```
sentiment-tool/
├── app.py                  # Main Streamlit dashboard
├── sentiment_analyzer.py   # Core AI sentiment logic
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🧠 How Sentiment Analysis Works

TextBlob analyses each piece of text and assigns a **polarity score** from **-1.0 to +1.0**:

| Score | Classification | Example |
|-------|---------------|---------|
| Above +0.05 | ✅ POSITIVE | *"Amazing product! Highly recommend!"* |
| -0.05 to +0.05 | 😐 NEUTRAL | *"It's okay, does the job."* |
| Below -0.05 | ❌ NEGATIVE | *"Terrible quality, waste of money."* |

---

## 📊 Week 3 Outcomes Achieved

- ✅ Working sentiment analysis solution
- ✅ Dashboard analysing sentiment with visualisations
- ✅ Insights report generated automatically
- ✅ Ability to interpret data using AI
- ✅ Documented individual project
- ✅ Deployed as live web application

---

## 👤 Author

**Bongimusa Khoza**


---

## 📄 License

MIT License — free to use and modify.
