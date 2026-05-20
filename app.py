"""
SentimentIQ — AI Sentiment Analysis Tool
Individual Project — Bongimusa Khoza
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sentiment_analyzer import SentimentAnalyzer, get_sample_reviews, get_sample_social, get_sample_ecommerce

st.set_page_config(page_title="SentimentIQ", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 0rem; padding-bottom: 2rem; }
.hero { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2.5rem 2.5rem; border-radius: 20px; margin-bottom: 2rem; text-align: center; color: white; }
.hero h1 { font-size: 2.8rem; font-weight: 700; margin: 0; letter-spacing: -1px; }
.hero p { font-size: 1.1rem; opacity: 0.88; margin-top: 0.6rem; font-weight: 300; }
.hero-badges { margin-top: 1.2rem; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
.badge { background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.35); padding: 5px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 500; }
.metric-card { background: white; border-radius: 16px; padding: 1.4rem 1.2rem; text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.07); border: 1px solid #f0f0f0; }
.metric-num { font-size: 2.2rem; font-weight: 700; margin: 0; }
.metric-label { font-size: 0.78rem; color: #888; margin: 4px 0 0; text-transform: uppercase; letter-spacing: 0.5px; }
.card-total .metric-num { color: #4a4a8a; }
.card-pos .metric-num { color: #22c55e; }
.card-neg .metric-num { color: #ef4444; }
.card-neu .metric-num { color: #f59e0b; }
.card-pol .metric-num { color: #6366f1; }
.section-title { font-size: 1.1rem; font-weight: 600; color: #1a1a2e; margin: 1.5rem 0 0.8rem; }
.insight-card { background: linear-gradient(135deg, #f8f9ff 0%, #eef2ff 100%); border-left: 4px solid #6366f1; border-radius: 0 12px 12px 0; padding: 1rem 1.2rem; margin: 0.5rem 0; font-size: 0.9rem; color: #333; }
.insight-good { border-left-color: #22c55e; background: linear-gradient(135deg, #f0fdf4, #dcfce7); }
.insight-bad { border-left-color: #ef4444; background: linear-gradient(135deg, #fef2f2, #fee2e2); }
.insight-warn { border-left-color: #f59e0b; background: linear-gradient(135deg, #fffbeb, #fef3c7); }
.challenge-card { background: #1e1e2e; border-left: 4px solid #667eea; border-radius: 0 12px 12px 0; padding: 1rem 1.2rem; margin: 0.5rem 0; font-size: 0.9rem; color: #ccc; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); }
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }
.stButton > button { background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; border-radius: 10px; padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem; width: 100%; }
#MainMenu { visibility: hidden; } header { visibility: hidden; } footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'original_col' not in st.session_state:
    st.session_state.original_col = None

with st.sidebar:
    st.markdown("## 🧠 SentimentIQ")
    st.markdown("<p style='opacity:0.6; font-size:0.82rem;'>AI-Powered Text Analysis</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 📁 Data Source")
    data_option = st.radio("", [
        "🛍️ Sample Reviews",
        "💬 Sample Social Media",
        "🛒 Sample E-Commerce",
        "📂 Upload CSV",
        "✏️ Custom Text"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 🎨 Chart Style")
    chart_type = st.selectbox("", ["Bar Chart", "Pie Chart", "Donut Chart"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div style='opacity:0.6; font-size:0.78rem; line-height:1.8;'><b>How it works</b><br>📌 Polarity: -1 to +1<br>🟢 Above +0.05 → Positive<br>🔴 Below -0.05 → Negative<br>🟡 In between → Neutral</div>", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 SentimentIQ</h1>
    <p>Understand what people really feel — powered by AI sentiment analysis</p>
    <div class="hero-badges">
        <span class="badge">⚡ Real-time Analysis</span>
        <span class="badge">📊 Interactive Charts</span>
        <span class="badge">☁️ Word Clouds</span>
        <span class="badge">🛒 E-Commerce Ready</span>
        <span class="badge">📄 Export Reports</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Analyse", "⚠️ Challenges"])

with tab1:
    df = None
    text_col = None

    if data_option == "🛍️ Sample Reviews":
        df = get_sample_reviews()
        text_col = "review"
        st.info("📦 **15 sample product reviews** loaded — click **Run Analysis** below!")
    elif data_option == "💬 Sample Social Media":
        df = get_sample_social()
        text_col = "comment"
        st.info("💬 **13 sample social media comments** loaded — click **Run Analysis** below!")
    elif data_option == "🛒 Sample E-Commerce":
        df = get_sample_ecommerce()
        text_col = "review"
        st.info("🛒 **20 sample e-commerce reviews** loaded — click **Run Analysis** below!")
    elif data_option == "📂 Upload CSV":
        uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            text_col = st.selectbox("Which column contains the text?", df.columns.tolist())
            st.dataframe(df.head(3), use_container_width=True)
        else:
            st.warning("Please upload a CSV file to continue.")
    elif data_option == "✏️ Custom Text":
        st.markdown("#### ✏️ Enter your texts (one per line)")
        custom = st.text_area("", height=160,
            placeholder="I love this product!\nThis is terrible.\nIt's okay I guess.",
            label_visibility="collapsed")
        if custom.strip():
            lines = [l.strip() for l in custom.strip().split('\n') if l.strip()]
            df = pd.DataFrame({"text": lines})
            text_col = "text"

    if df is not None and text_col:
        col_btn, col_reset = st.columns([3, 1])
        with col_btn:
            run = st.button("🚀 Run Sentiment Analysis")
        with col_reset:
            if st.button("🔄 Reset"):
                st.session_state.results = None
                st.session_state.original_col = None
                st.rerun()

        if run:
            with st.spinner("🔍 Analysing sentiment..."):
                st.session_state.original_col = text_col
                st.session_state.results = st.session_state.analyzer.analyze_dataframe(df, text_col)
            st.success(f"✅ Done! Analysed **{len(st.session_state.results)}** texts.")

    if st.session_state.results is not None and st.session_state.original_col is not None:
        res = st.session_state.results
        counts = res['sentiment'].value_counts()
        total = len(res)
        pos = counts.get('POSITIVE', 0)
        neg = counts.get('NEGATIVE', 0)
        neu = counts.get('NEUTRAL', 0)
        avg_pol = res['polarity'].mean()

        st.markdown("---")
        st.markdown('<div class="section-title">📊 Summary</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.markdown(f'<div class="metric-card card-total"><p class="metric-num">{total}</p><p class="metric-label">Total Analysed</p></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card card-pos"><p class="metric-num">{pos}</p><p class="metric-label">✅ Positive</p></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card card-neg"><p class="metric-num">{neg}</p><p class="metric-label">❌ Negative</p></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card card-neu"><p class="metric-num">{neu}</p><p class="metric-label">😐 Neutral</p></div>', unsafe_allow_html=True)
        with c5: st.markdown(f'<div class="metric-card card-pol"><p class="metric-num">{avg_pol:+.2f}</p><p class="metric-label">Avg Polarity</p></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Visualisations</div>', unsafe_allow_html=True)
        colors_map = {'POSITIVE': '#22c55e', 'NEUTRAL': '#f59e0b', 'NEGATIVE': '#ef4444'}
        c_list = [colors_map.get(s, '#888') for s in counts.index]

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Sentiment Distribution**")
            if chart_type == "Bar Chart":
                fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                    color_discrete_map=colors_map, text=counts.values, labels={'x': '', 'y': 'Count'})
                fig.update_traces(textposition='outside', marker_line_width=0)
                fig.update_layout(showlegend=False, height=320, margin=dict(t=10,b=10,l=10,r=10),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(gridcolor='#f0f0f0'))
                st.plotly_chart(fig, use_container_width=True)
            elif chart_type == "Pie Chart":
                fig = px.pie(values=counts.values, names=counts.index,
                    color=counts.index, color_discrete_map=colors_map)
                fig.update_layout(height=320, margin=dict(t=10,b=10), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = go.Figure(go.Pie(labels=counts.index, values=counts.values,
                    hole=0.55, marker_colors=c_list, textinfo='label+percent'))
                fig.update_layout(height=320, margin=dict(t=10,b=10), paper_bgcolor='rgba(0,0,0,0)',
                    annotations=[dict(text=f'{total}<br>texts', x=0.5, y=0.5, font_size=14, showarrow=False)])
                st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown("**Polarity Score Distribution**")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            fig2.patch.set_facecolor('#fafafa')
            ax2.set_facecolor('#fafafa')
            n, bins, patches = ax2.hist(res['polarity'], bins=12, edgecolor='white', linewidth=0.8)
            for patch, left in zip(patches, bins):
                patch.set_facecolor('#22c55e' if left > 0.05 else ('#ef4444' if left < -0.05 else '#f59e0b'))
            ax2.axvline(0, color='#6366f1', linestyle='--', linewidth=1.5, label='Neutral (0)')
            ax2.axvline(avg_pol, color='#1a1a2e', linestyle='-', linewidth=2, label=f'Average ({avg_pol:+.2f})')
            ax2.set_xlabel("Polarity Score", fontsize=10, color='#555')
            ax2.set_ylabel("Count", fontsize=10, color='#555')
            ax2.legend(fontsize=9)
            ax2.spines[['top','right']].set_visible(False)
            ax2.spines[['left','bottom']].set_color('#ddd')
            fig2.tight_layout()
            st.pyplot(fig2)

        # E-commerce extra: rating vs sentiment chart
        if data_option == "🛒 Sample E-Commerce" and 'rating' in res.columns:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">🛒 E-Commerce Insights</div>', unsafe_allow_html=True)
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown("**Rating vs Sentiment**")
                rating_sentiment = res.groupby(['rating', 'sentiment']).size().reset_index(name='count')
                fig_ec = px.bar(rating_sentiment, x='rating', y='count', color='sentiment',
                    color_discrete_map=colors_map, barmode='group',
                    labels={'rating': 'Star Rating', 'count': 'Count'})
                fig_ec.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_ec, use_container_width=True)
            with ec2:
                st.markdown("**Category Breakdown**")
                if 'category' in res.columns:
                    cat_counts = res.groupby(['category', 'sentiment']).size().reset_index(name='count')
                    fig_cat = px.bar(cat_counts, x='category', y='count', color='sentiment',
                        color_discrete_map=colors_map, barmode='stack',
                        labels={'category': 'Category', 'count': 'Count'})
                    fig_cat.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10),
                        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_tickangle=-30)
                    st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">☁️ Word Clouds</div>', unsafe_allow_html=True)
        pos_text = " ".join(res[res['sentiment'] == 'POSITIVE']['text'].astype(str).tolist())
        neg_text = " ".join(res[res['sentiment'] == 'NEGATIVE']['text'].astype(str).tolist())

        wc1, wc2 = st.columns(2)
        with wc1:
            st.markdown("**Positive Reviews**")
            if pos_text.strip():
                wc = WordCloud(width=600, height=300, background_color='#f0fdf4',
                    colormap='Greens', max_words=40).generate(pos_text)
                fig3, ax3 = plt.subplots(figsize=(6, 3))
                fig3.patch.set_facecolor('#f0fdf4')
                ax3.imshow(wc, interpolation='bilinear'); ax3.axis('off')
                fig3.tight_layout(pad=0); st.pyplot(fig3)
            else:
                st.info("No positive texts to display.")
        with wc2:
            st.markdown("**Negative Reviews**")
            if neg_text.strip():
                wc = WordCloud(width=600, height=300, background_color='#fef2f2',
                    colormap='Reds', max_words=40).generate(neg_text)
                fig4, ax4 = plt.subplots(figsize=(6, 3))
                fig4.patch.set_facecolor('#fef2f2')
                ax4.imshow(wc, interpolation='bilinear'); ax4.axis('off')
                fig4.tight_layout(pad=0); st.pyplot(fig4)
            else:
                st.info("No negative texts to display.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">💡 AI Insights & Recommendations</div>', unsafe_allow_html=True)
        overall = "POSITIVE" if avg_pol > 0.1 else ("NEGATIVE" if avg_pol < -0.1 else "NEUTRAL")
        recs = []
        if neg/total > 0.3: recs.append(("bad", "⚠️ High negative rate — investigate common complaints"))
        if pos/total > 0.6: recs.append(("good", "🌟 Strong positive sentiment — use this in your marketing"))
        if avg_pol < -0.1: recs.append(("bad", "🔴 Overall tone is negative — prioritise improvements"))
        elif avg_pol > 0.1: recs.append(("good", "🟢 Overall tone is positive — keep maintaining quality"))
        if not recs: recs.append(("warn", "📊 Mixed sentiment — look deeper for specific themes"))
        recs.append(("warn", "📬 Always respond promptly to negative feedback"))
        if data_option == "🛒 Sample E-Commerce":
            recs.append(("good", "🛒 E-commerce tip: Focus on delivery speed and packaging quality"))
            recs.append(("bad", "🛒 E-commerce tip: Address product description accuracy issues"))

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown(f'<div class="insight-card"><b>Overall Sentiment: {overall}</b><br>Average polarity: <b>{avg_pol:+.3f}</b><br>Confidence: <b>{res["confidence"].mean():.0%}</b></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="insight-card"><b>Breakdown</b><br>✅ {pos} positive ({pos/total*100:.1f}%)<br>❌ {neg} negative ({neg/total*100:.1f}%)<br>😐 {neu} neutral ({neu/total*100:.1f}%)</div>', unsafe_allow_html=True)
        with col_i2:
            for style, text in recs:
                st.markdown(f'<div class="insight-card insight-{style}">{text}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Detailed Results</div>', unsafe_allow_html=True)
        display = res[['text', 'sentiment', 'confidence', 'polarity', 'subjectivity']].copy()
        display.columns = ['Text', 'Sentiment', 'Confidence', 'Polarity', 'Subjectivity']

        def colour_sentiment(val):
            if val == 'POSITIVE': return 'background-color:#dcfce7;color:#166534;font-weight:600'
            if val == 'NEGATIVE': return 'background-color:#fee2e2;color:#991b1b;font-weight:600'
            return 'background-color:#fef3c7;color:#92400e;font-weight:600'

        st.dataframe(display.style.map(colour_sentiment, subset=['Sentiment']), use_container_width=True, height=300)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">💾 Export Your Work</div>', unsafe_allow_html=True)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("📥 Download CSV Results", data=res.to_csv(index=False),
                file_name="sentiment_results.csv", mime="text/csv", use_container_width=True)
        with dl2:
            report = f"# Sentiment Analysis Report\n\n**Overall:** {overall} | **Polarity:** {avg_pol:+.3f}\n\n| Metric | Value |\n|--------|-------|\n| Total | {total} |\n| Positive | {pos} ({pos/total*100:.1f}%) |\n| Negative | {neg} ({neg/total*100:.1f}%) |\n| Neutral | {neu} ({neu/total*100:.1f}%) |\n\n*Generated by SentimentIQ — Bongimusa Khoza*"
            st.download_button("📄 Download Insights Report", data=report,
                file_name="sentiment_report.md", mime="text/markdown", use_container_width=True)

# ── Challenges Tab ────────────────────────────────────────────────
with tab2:
    st.markdown("## ⚠️ Challenges I Faced Building SentimentIQ")
    st.markdown("*A honest reflection on the real problems encountered during development and deployment.*")
    st.markdown("---")

    challenges = [
        {
            "num": "1",
            "title": "streamlit command not recognised in terminal",
            "problem": "After installing Streamlit, running `streamlit run app.py` gave a 'command not found' error in PowerShell on Windows.",
            "solution": "Used `python -m streamlit run app.py` instead, which runs Streamlit through the correct Python installation.",
            "lesson": "Always use `python -m` prefix when running Python tools on Windows to avoid PATH issues.",
            "icon": "💻"
        },
        {
            "num": "2",
            "title": "ModuleNotFoundError: wordcloud",
            "problem": "Even after running `pip install wordcloud`, the app still threw a ModuleNotFoundError when deployed.",
            "solution": "Realised the system had multiple Python versions. Used `python -m pip install wordcloud` to install for the correct version.",
            "lesson": "Always match the pip installation to the specific Python version you are running.",
            "icon": "📦"
        },
        {
            "num": "3",
            "title": "Deployment failing due to pinned package versions",
            "problem": "Streamlit Cloud uses Python 3.14. My requirements.txt had old specific version numbers like pandas==2.1.4 which were incompatible.",
            "solution": "Removed all version numbers from requirements.txt and let Streamlit Cloud install the latest compatible versions automatically.",
            "lesson": "For cloud deployment, avoid pinning specific versions unless absolutely necessary.",
            "icon": "☁️"
        },
        {
            "num": "4",
            "title": "applymap deprecated in newer pandas",
            "problem": "The app crashed on Streamlit Cloud with an AttributeError because pandas renamed applymap to map in newer versions.",
            "solution": "Updated the code to use `.style.map()` instead of `.style.applymap()` which works across all pandas versions.",
            "lesson": "Always check for deprecated methods when deploying to environments with newer library versions.",
            "icon": "🐼"
        },
        {
            "num": "5",
            "title": "KeyError on text column after analysis",
            "problem": "After running analysis the app crashed with a KeyError because the original text column name was lost when the results dataframe was created.",
            "solution": "Saved the original column name in Streamlit session state before running the analysis so it could be retrieved afterwards.",
            "lesson": "Use session state to persist important variables across Streamlit reruns.",
            "icon": "🔑"
        },
        {
            "num": "6",
            "title": "CSV upload analysing numeric data as neutral",
            "problem": "When uploading a CSV with numeric columns (like daily_social_media_hours), the tool analysed the numbers as text and classified everything as NEUTRAL.",
            "solution": "Added a column selector dropdown so users can choose which column contains the actual text to analyse.",
            "lesson": "Always validate user input — not all CSV columns contain text suitable for sentiment analysis.",
            "icon": "📂"
        },
        {
            "num": "7",
            "title": "App showed blank page on first load",
            "problem": "After updating code on GitHub, the Streamlit Cloud app showed a completely blank white page with no error message visible.",
            "solution": "Checked the Manage App logs in Streamlit Cloud which revealed the exact error — a missing module that needed to be installed.",
            "lesson": "Always check the deployment logs rather than guessing — the answer is always in the logs.",
            "icon": "📄"
        },
        {
            "num": "8",
            "title": "String slicing error in Python 3.14",
            "problem": "The analyze method crashed with a TypeError when slicing text strings on Python 3.14 due to stricter type handling.",
            "solution": "Added explicit str() conversion before any string operations to ensure the text is always a plain Python string.",
            "lesson": "Python 3.14 is stricter about types — always convert input to the expected type explicitly.",
            "icon": "🐍"
        },
    ]

    for c in challenges:
        with st.expander(f"#{c['num']}  {c['icon']}  {c['title']}"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="insight-card insight-bad">
                <b>❌ The Problem</b><br>{c['problem']}
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="insight-card insight-good">
                <b>✅ The Solution</b><br>{c['solution']}
                </div>
                """, unsafe_allow_html=True)
            st.markdown(f"""
            <div class="insight-card">
            <b>💡 Lesson Learned:</b> {c['lesson']}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea20, #764ba220); border-radius: 12px; padding: 1.2rem; text-align: center; color: #555;'>
    <b>💬 Reflection</b><br>
    Every challenge above made the final product stronger. Debugging real errors in a live deployment environment
    taught me more about Python, cloud deployment, and library compatibility than any tutorial could.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;padding:1.5rem;background:linear-gradient(135deg,#667eea20,#764ba220);border-radius:16px;color:#555;font-size:0.85rem;'>🧠 <b>SentimentIQ</b> · Built with Python, TextBlob & Streamlit · Bongimusa Khoza · 2026</div>", unsafe_allow_html=True)
