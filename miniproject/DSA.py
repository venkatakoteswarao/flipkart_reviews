import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import time


# Function to scrape reviews from a single page
def scrape_reviews(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            st.error(f"Failed to retrieve the page: {response.status_code}")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')

        # Scrape reviews and remove 'READ MORE' from each review
        reviews = [
            review.text.strip().replace("READ MORE", "").strip() if review.text.strip() else "No review"
            for review in soup.find_all('div', class_='ZmyHeo')
        ]

        if not reviews:
            st.warning("No data found. The structure of the webpage might have changed.")
            return None

        return pd.DataFrame({"Review": reviews})

    except requests.exceptions.RequestException as e:
        st.error(f"A network error occurred: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# Function to scrape multiple pages
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

        time.sleep(2)
        page += 1

    if all_reviews:
        return pd.concat(all_reviews, ignore_index=True).head(total_reviews if total_reviews else None)
    else:
        return None


# Sentiment analysis using VADER and BERT
def analyze_sentiment(df):
    st.info("Analyzing sentiments...")

    # VADER for rule-based sentiment analysis
    vader = SentimentIntensityAnalyzer()
    df['VADER Sentiment'] = df['Review'].apply(lambda x: 'Positive' if vader.polarity_scores(x)['compound'] >= 0 else 'Negative')

    # BERT for more nuanced sentiment analysis
    bert = pipeline("sentiment-analysis")
    df['BERT Sentiment'] = df['Review'].apply(lambda x: bert(x)[0]['label'])

    return df


# Display the reviews with sentiment categorization
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

        # Sentiment analysis pie chart
        st.subheader("Sentiment Analysis")
        labels = ['Positive', 'Negative']
        sizes = [len(positives), len(negatives)]
        colors = ['#4caf50', '#f44336']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        # Calculate percentages
        total_reviews = len(df)
        positive_percentage = len(positives) / total_reviews * 100
        negative_percentage = len(negatives) / total_reviews * 100

        # Display message based on percentages
        st.subheader("Product Recommendation")
        if positive_percentage > 80:
            st.success(f"🌟 {positive_percentage:.2f}% of reviews are positive. The product performance is **excellent**! Highly recommended!")
        elif positive_percentage > 60:
            st.info(f"👍 {positive_percentage:.2f}% of reviews are positive. The product performance is **good**. Worth considering!")
        elif positive_percentage > 40:
            st.warning(f"⚠️ {positive_percentage:.2f}% of reviews are positive. The product performance is **average**. Consider other options.")
        else:
            st.error(f"❌ Only {positive_percentage:.2f}% of reviews are positive. The product performance is **poor**. Not recommended.")

        st.write(f"🔍 **Negative Reviews Percentage:** {negative_percentage:.2f}%.")


# Streamlit App
st.title("Product Review Scraper with Sentiment Analysis")
st.markdown("This tool scrapes product reviews from Flipkart, analyzes sentiments using VADER and BERT, and categorizes them into Positive and Negative.")

url = st.text_input("Enter the Flipkart product review URL:")

if url:
    scrape_option = st.radio("Select scraping option:", ("Custom Reviews", "Full Scraping"))

    if scrape_option == "Custom Reviews":
        total_reviews = st.number_input("Enter the total number of reviews to scrape:", min_value=1, step=1)
    else:
        total_reviews = None

    if st.button("Scrape Reviews"):
        with st.spinner("Scraping reviews..."):
            reviews_df = scrape_multiple_pages(url, total_reviews)

        if reviews_df is not None:
            reviews_df = analyze_sentiment(reviews_df)
            display_reviews(reviews_df)

            if st.button("Download Reviews as CSV"):
                csv_data = reviews_df.to_csv(index=False)
                st.download_button(label="Download CSV", data=csv_data, file_name="reviews.csv")

else:
    st.warning("Please enter a valid URL to begin scraping.")
