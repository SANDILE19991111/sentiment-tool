"""
Sentiment Analysis Dashboard
Individual Project — Week 3
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import io
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sentiment_analyzer import SentimentAnalyzer, get_sample_reviews, get_sample_social

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-box { background: #f8f9fa; border-radius: 10px; padding: 1rem; text-align: center; }
    .pos-badge { background: #d4edda; color: #155724; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .neg-badge { background: #f8d7da; color: #721c24; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .neu-badge { background: #fff3cd; color: #856404; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .insight-box { background: #e8f4f8; border-left: 4px solid #1E88E5; padding: 1rem; border-radius: 0 8px 8px 0; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = SentimentAnalyzer()
if 'results' not in st.session_state:
    st.session_state.results = None
if 'df_source' not in st.session_state:
    st.session_state.df_source = None

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Sentiment Analyser")
    st.markdown("*My Individual Project*")
    st.divider()

    st.markdown("### 📁 Data Source")
    data_option = st.radio(
        "Choose your input",
        ["Sample Reviews", "Sample Social Media", "Upload CSV", "Custom Text"]
    )

    st.divider()
    st.markdown("### 🎨 Chart Style")
    chart_type = st.selectbox("Visualisation", ["Bar Chart", "Pie Chart", "Donut Chart"])

    st.divider()
    st.markdown("### ℹ️ About")
    st.info("Built with Python, TextBlob, Streamlit & Plotly.\n\nMy Week 3 individual project.")

# ── Header ────────────────────────────────────────────────────────
st.markdown("# 📊 Sentiment Analysis Tool")
st.markdown("*Analyse customer reviews and social media comments using AI*")
st.divider()

# ── Data Loading ──────────────────────────────────────────────────
df = None
text_col = None

if data_option == "Sample Reviews":
    df = get_sample_reviews()
    text_col = "review"
    st.info("📦 Using 15 sample product reviews — click **Run Analysis** to start.")

elif data_option == "Sample Social Media":
    df = get_sample_social()
    text_col = "comment"
    st.info("💬 Using 13 sample social media comments — click **Run Analysis** to start.")

elif data_option == "Upload CSV":
    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        text_col = st.selectbox("Which column contains the text?", df.columns.tolist())
        st.dataframe(df.head(3), use_container_width=True)
    else:
        st.warning("Please upload a CSV file to continue.")

elif data_option == "Custom Text":
    st.markdown("#### Enter your texts (one per line)")
    custom = st.text_area(
        "Paste your reviews or comments here:",
        height=180,
        placeholder="I love this product!\nThis is terrible quality.\nIt's okay I guess.\nBest purchase ever made!"
    )
    if custom.strip():
        lines = [l.strip() for l in custom.strip().split('\n') if l.strip()]
        df = pd.DataFrame({"text": lines})
        text_col = "text"

# ── Run Analysis ──────────────────────────────────────────────────
if df is not None and text_col:
    col_btn, col_reset = st.columns([2, 1])
    with col_btn:
        run = st.button("🚀 Run Sentiment Analysis", type="primary", use_container_width=True)
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.results = None
            st.rerun()

    if run:
        with st.spinner("Analysing sentiment..."):
            results = st.session_state.analyzer.analyze_dataframe(df, text_col)
            st.session_state.results = results
        st.success(f"✅ Analysis complete! Processed {len(results)} texts.")

# ── Results ───────────────────────────────────────────────────────
if st.session_state.results is not None:
    res = st.session_state.results
    counts = res['sentiment'].value_counts()
    total = len(res)
    pos = counts.get('POSITIVE', 0)
    neg = counts.get('NEGATIVE', 0)
    neu = counts.get('NEUTRAL', 0)
    avg_pol = res['polarity'].mean()

    st.divider()

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📝 Total", total)
    c2.metric("✅ Positive", f"{pos} ({pos/total*100:.0f}%)")
    c3.metric("❌ Negative", f"{neg} ({neg/total*100:.0f}%)")
    c4.metric("😐 Neutral", f"{neu} ({neu/total*100:.0f}%)")
    c5.metric("📈 Avg Polarity", f"{avg_pol:+.2f}")

    st.divider()

    # Charts
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Sentiment Distribution")
        colors_map = {'POSITIVE': '#4CAF50', 'NEUTRAL': '#FFC107', 'NEGATIVE': '#f44336'}
        c_list = [colors_map.get(s, '#888') for s in counts.index]

        if chart_type == "Bar Chart":
            fig = px.bar(
                x=counts.index, y=counts.values,
                color=counts.index,
                color_discrete_map=colors_map,
                labels={'x': 'Sentiment', 'y': 'Count'},
                text=counts.values
            )
            fig.update_traces(textposition='outside')
            fig.update_layout(showlegend=False, height=320, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Pie Chart":
            fig = px.pie(
                values=counts.values, names=counts.index,
                color=counts.index, color_discrete_map=colors_map
            )
            fig.update_layout(height=320, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        else:
            fig = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                hole=0.5, marker_colors=c_list
            ))
            fig.update_layout(height=320, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("#### Polarity Score Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(res['polarity'], bins=12, color='#1E88E5', edgecolor='white', alpha=0.85)
        ax2.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Neutral line (0)')
        ax2.axvline(avg_pol, color='orange', linestyle='-', linewidth=1.5, label=f'Average ({avg_pol:+.2f})')
        ax2.set_xlabel("Polarity  (−1 = very negative   →   +1 = very positive)", fontsize=10)
        ax2.set_ylabel("Number of texts", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.spines[['top', 'right']].set_visible(False)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.divider()

    # Word Clouds
    st.markdown("#### ☁️ Word Clouds")
    pos_text = " ".join(res[res['sentiment'] == 'POSITIVE'][text_col].astype(str))
    neg_text = " ".join(res[res['sentiment'] == 'NEGATIVE'][text_col].astype(str))

    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown("**Positive reviews**")
        if pos_text.strip():
            wc = WordCloud(width=500, height=280, background_color='white',
                           colormap='Greens', max_words=50).generate(pos_text)
            fig3, ax3 = plt.subplots(figsize=(5, 2.8))
            ax3.imshow(wc, interpolation='bilinear')
            ax3.axis('off')
            fig3.tight_layout()
            st.pyplot(fig3)
        else:
            st.info("No positive texts found.")

    with wc2:
        st.markdown("**Negative reviews**")
        if neg_text.strip():
            wc = WordCloud(width=500, height=280, background_color='white',
                           colormap='Reds', max_words=50).generate(neg_text)
            fig4, ax4 = plt.subplots(figsize=(5, 2.8))
            ax4.imshow(wc, interpolation='bilinear')
            ax4.axis('off')
            fig4.tight_layout()
            st.pyplot(fig4)
        else:
            st.info("No negative texts found.")

    st.divider()

    # Insights Report
    st.markdown("#### 📋 AI Insights Report")
    overall = "POSITIVE" if avg_pol > 0.1 else ("NEGATIVE" if avg_pol < -0.1 else "NEUTRAL")
    col_ins1, col_ins2 = st.columns(2)

    with col_ins1:
        st.markdown(f"""
<div class="insight-box">
<b>Overall Sentiment:</b> {overall}<br>
Average polarity score: <b>{avg_pol:+.3f}</b> out of ±1.0<br>
Average confidence: <b>{res['confidence'].mean():.0%}</b>
</div>
""", unsafe_allow_html=True)
        st.markdown(f"""
<div class="insight-box">
<b>Breakdown:</b><br>
✅ {pos} positive ({pos/total*100:.1f}%)<br>
❌ {neg} negative ({neg/total*100:.1f}%)<br>
😐 {neu} neutral ({neu/total*100:.1f}%)
</div>
""", unsafe_allow_html=True)

    with col_ins2:
        recs = []
        if neg/total > 0.3:
            recs.append("⚠️ High negative rate — investigate common complaints")
        if pos/total > 0.6:
            recs.append("🌟 Strong positive sentiment — use in marketing")
        if avg_pol < -0.1:
            recs.append("🔴 Overall tone is negative — prioritise improvements")
        elif avg_pol > 0.1:
            recs.append("🟢 Overall tone is positive — maintain quality")
        if not recs:
            recs.append("📊 Mixed sentiment — look for specific themes")
        recs.append("📬 Respond promptly to all negative reviews")

        for r in recs:
            st.markdown(f'<div class="insight-box">{r}</div>', unsafe_allow_html=True)

    st.divider()

    # Detailed table
    st.markdown("#### 📄 Detailed Results")
    display = res[[text_col, 'sentiment', 'confidence', 'polarity', 'subjectivity']].copy()
    display.columns = ['Text', 'Sentiment', 'Confidence', 'Polarity', 'Subjectivity']

    def colour_sentiment(val):
        if val == 'POSITIVE': return 'background-color: #d4edda; color: #155724'
        if val == 'NEGATIVE': return 'background-color: #f8d7da; color: #721c24'
        return 'background-color: #fff3cd; color: #856404'

    styled = display.style.map(colour_sentiment, subset=['Sentiment'])
    st.dataframe(styled, use_container_width=True)

    st.divider()

    # Downloads
    st.markdown("#### 💾 Export Your Work")
    dl1, dl2 = st.columns(2)

    with dl1:
        csv_data = res.to_csv(index=False)
        st.download_button(
            "📥 Download CSV Results",
            data=csv_data,
            file_name="sentiment_results.csv",
            mime="text/csv",
            use_container_width=True
        )

    with dl2:
        report = f"""# Sentiment Analysis — Insights Report
**My Individual Project**

---

## Executive Summary
| Metric | Value |
|--------|-------|
| Total texts analysed | {total} |
| Positive | {pos} ({pos/total*100:.1f}%) |
| Negative | {neg} ({neg/total*100:.1f}%) |
| Neutral | {neu} ({neu/total*100:.1f}%) |
| Average polarity | {avg_pol:+.3f} |
| Average confidence | {res['confidence'].mean():.1%} |
| Analysis method | TextBlob |

## Overall Sentiment
The overall tone is **{overall}** with an average polarity of **{avg_pol:+.3f}**.

## Recommendations
{''.join([f"- {r}\\n" for r in recs])}

## How Sentiment Analysis Works
- **Polarity** ranges from -1.0 (very negative) to +1.0 (very positive)
- **Subjectivity** ranges from 0 (factual) to 1 (opinionated)
- Texts scoring above +0.05 polarity are classified **POSITIVE**
- Texts scoring below -0.05 polarity are classified **NEGATIVE**
- Scores in between are classified **NEUTRAL**

---
*Generated by My Sentiment Analysis Tool*
"""
        st.download_button(
            "📄 Download Insights Report",
            data=report,
            file_name="sentiment_insights_report.md",
            mime="text/markdown",
            use_container_width=True
        )

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Sentiment Analysis Tool · Individual Project · 2026"
    "</div>",
    unsafe_allow_html=True
)
