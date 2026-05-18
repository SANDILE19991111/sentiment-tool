"""
Sentiment Analysis Module
Uses TextBlob for fast, reliable sentiment analysis
"""

import re
import pandas as pd
from textblob import TextBlob
from typing import Dict, List

class SentimentAnalyzer:
    """Sentiment analyzer using TextBlob"""

    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def analyze(self, text) -> Dict:
        # Ensure text is always a plain string (Python 3.14 compatible)
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = text.strip()

        cleaned = self.clean_text(text)
        blob = TextBlob(cleaned)
        polarity = float(blob.sentiment.polarity)
        subjectivity = float(blob.sentiment.subjectivity)

        if polarity > 0.05:
            sentiment = "POSITIVE"
            confidence = round(min(0.55 + abs(polarity) * 0.45, 0.99), 3)
        elif polarity < -0.05:
            sentiment = "NEGATIVE"
            confidence = round(min(0.55 + abs(polarity) * 0.45, 0.99), 3)
        else:
            sentiment = "NEUTRAL"
            confidence = round(0.50 + (0.05 - abs(polarity)), 3)

        # Safe text truncation
        text_str = str(text)
        display_text = (text_str[:120] + "...") if len(text_str) > 120 else text_str

        return {
            "text": display_text,
            "sentiment": sentiment,
            "confidence": confidence,
            "polarity": round(polarity, 3),
            "subjectivity": round(subjectivity, 3),
            "method": "TextBlob"
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        return [self.analyze(t) for t in texts]

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        # Use explicit loop for Python 3.14 compatibility
        records = []
        for val in df[text_column]:
            records.append(self.analyze(val))
        results = pd.DataFrame(records)
        return pd.concat([df.reset_index(drop=True), results], axis=1)


def get_sample_reviews():
    return pd.DataFrame({
        "review": [
            "This product is absolutely amazing! Best purchase I have ever made. Highly recommended!",
            "Terrible quality. Broke after 2 days. Complete waste of money. Very disappointed.",
            "It is okay for the price. Nothing special but gets the job done.",
            "Customer service was extremely helpful and resolved my issue quickly. Thank you!",
            "The shipping was late and the packaging was damaged. Not happy at all.",
            "Neutral about this product. It works as expected but there is no wow factor.",
            "Excellent value! Would buy again. My whole family absolutely loves it.",
            "Poor build quality. Does not match the description. Avoid this seller.",
            "Pretty good overall. Some minor issues but satisfied with the purchase.",
            "Worst experience ever. Will never shop here again. Terrible customer service.",
            "Fast delivery and great packaging. Product works perfectly as described.",
            "Disappointed with quality. Expected much better for the price paid.",
            "Solid product, does what it says. Good value for money.",
            "Outstanding! Exceeded all my expectations. Will definitely order again.",
            "Average product, nothing to write home about. Just okay."
        ],
        "product": [
            "Wireless Headphones", "Phone Case", "USB Cable", "Smart Watch",
            "Laptop Stand", "Desk Lamp", "Coffee Maker", "Backpack", "Mouse",
            "Charger", "Keyboard", "Monitor", "Webcam", "Speaker", "Router"
        ],
        "rating": [5, 1, 3, 5, 2, 3, 5, 1, 4, 1, 5, 2, 4, 5, 3]
    })


def get_sample_social():
    return pd.DataFrame({
        "comment": [
            "Love the new update! So many great features! Great job team!",
            "This is the worst app ever made. So buggy and slow. Uninstalling.",
            "Meh. It is fine I guess. Nothing to write home about.",
            "Absolutely incredible! Cannot believe how good this is!",
            "Why does this keep crashing? So frustrating!",
            "Just downloaded it. Looks promising so far.",
            "Been using this for years. Still my favorite. Never switching.",
            "Customer support ignored my emails. Terrible service.",
            "Great concept, poor execution. Needs more work.",
            "Finally! A product that actually works as advertised. Ten out of ten.",
            "Highly recommend this to everyone. Life changing!",
            "Do not waste your money on this. Total scam.",
            "Pretty decent for everyday use. Gets the job done.",
        ],
        "platform": ["Social Media"] * 13
    })
