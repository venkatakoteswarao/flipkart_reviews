import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

# Predefined list of User-Agent strings
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# Function to scrape reviews from a single page with retry and anti-bot headers
def scrape_reviews(url):
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.flipkart.com/",
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "no-cache",
    }

    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                break
            elif response.status_code in [429, 529]:
                st.warning("Too many requests. Retrying...")
                time.sleep(5 * (attempt + 1))
            else:
                st.error(f"Failed to retrieve the page: {response.status_code}")
                return None
        except requests.RequestException as e:
            st.error(f"Network error: {e}")
            return None
    else:
        st.error("Failed after multiple attempts. You may be blocked.")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    reviews = [
        review.text.strip().replace("READ MORE", "").strip() if review.text.strip() else "No review"
        for review in soup.find_all('div', class_='ZmyHeo')
    ]

    if not reviews:
        st.warning("No reviews found. Flipkart might have updated the page layout.")
        return None

    return pd.DataFrame({"Review": reviews})

# Scrape multiple pages
def scrape_multiple_pages(base_url, total_reviews=None):
    all_reviews = []
    page = 1
    scraped_reviews = 0

    while total_reviews is None or scraped_reviews < total_reviews:
        url = f"{base_url}&page={page}"
        st.info(f"Scraping page {page}...")

        df = scrape_reviews(url)
        if df is not None and not df.empty:
            all_reviews.append(df)
            scraped_reviews += len(df)
        else:
            st.info("No more data found. Stopping scrape.")
            break

        time.sleep(random.uniform(5, 10))  # Safer delay
        page += 1

    if all_reviews:
        return pd.concat(all_reviews, ignore_index=True).head(total_reviews if total_reviews else None)
    else:
        return None

# Sentiment analysis using VADER and BERT
def analyze_sentiment(df):
    st.info("Analyzing sentiments...")
    vader = SentimentIntensityAnalyzer()
    df['VADER Sentiment'] = df['Review'].apply(lambda x: 'Positive' if vader.polarity_scores(x)['compound'] >= 0 else 'Negative')

    try:
        bert = pipeline("sentiment-analysis")
        df['BERT Sentiment'] = df['Review'].apply(lambda x: bert(x)[0]['label'])
    except Exception as e:
        st.error(f"BERT analysis failed: {e}")
        df['BERT Sentiment'] = 'N/A'

    return df

# Display categorized reviews
def display_reviews(df):
    if df is not None:
        st.subheader("Categorized Reviews")

        positives = df[df['VADER Sentiment'] == 'Positive']
        negatives = df[df['VADER Sentiment'] == 'Negative']

        col1, col2 = st.columns(2)
        with col1:
            st.write("### Positive Reviews")
            st.dataframe(positives[['Review']])

        with col2:
            st.write("### Negative Reviews")
            st.dataframe(negatives[['Review']])

        # Pie chart
        st.subheader("Sentiment Chart")
        labels = ['Positive', 'Negative']
        sizes = [len(positives), len(negatives)]
        colors = ['#4caf50', '#f44336']
        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        total = len(df)
        pos_percent = len(positives) / total * 100
        neg_percent = len(negatives) / total * 100

        st.subheader("Product Recommendation")
        if pos_percent > 80:
            st.success(f"🌟 {pos_percent:.2f}% positive reviews. Excellent product!")
        elif pos_percent > 60:
            st.info(f"👍 {pos_percent:.2f}% positive reviews. Good product!")
        elif pos_percent > 40:
            st.warning(f"⚠ {pos_percent:.2f}% positive reviews. Average product.")
        else:
            st.error(f"❌ Only {pos_percent:.2f}% positive reviews. Not recommended.")
        st.write(f"Negative Reviews: {neg_percent:.2f}%")

# Streamlit UI
st.title("📦 Flipkart Product Review Scraper with Sentiment Analysis")
st.markdown("Scrapes reviews from Flipkart and analyzes them using VADER and BERT.")

url = st.text_input("🔗 Enter the Flipkart product review URL:")

if url:
    option = st.radio("Select scraping option:", ("Custom Reviews", "Full Scraping"))

    if option == "Custom Reviews":
        total_reviews = st.number_input("How many reviews do you want to scrape?", min_value=1, step=1)
    else:
        total_reviews = None

    if st.button("🔍 Scrape Reviews"):
        with st.spinner("Scraping reviews..."):
            df = scrape_multiple_pages(url, total_reviews)

        if df is not None:
            df = analyze_sentiment(df)
            display_reviews(df)

            csv = df.to_csv(index=False)
            st.download_button("⬇ Download CSV", data=csv, file_name="reviews.csv")
else:
    st.warning("⚠️ Please enter a Flipkart review URL.")
