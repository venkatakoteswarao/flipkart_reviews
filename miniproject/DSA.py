import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import time

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

        ratings = [
            int(rating.text.strip()) if rating.text.strip().isdigit() else 0
            for rating in soup.find_all('div', class_='XQDdHH Ga3i8K')
        ]
        reviews = [
            review.text.strip() if review.text.strip() else "No review"
            for review in soup.find_all('div', class_='ZmyHeo')
        ]

        if not ratings and not reviews:
            st.warning("No data found. The structure of the webpage might have changed.")
            return None

        max_length = max(len(ratings), len(reviews))
        ratings.extend([0] * (max_length - len(ratings)))
        reviews.extend(["No review"] * (max_length - len(reviews)))

        data = {'Rating': ratings, 'Review': reviews}
        df = pd.DataFrame(data)

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"A network error occurred: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def scrape_all_pages(base_url):
    all_reviews = []
    page = 1

    while True:
        url = f"{base_url}&page={page}"
        st.info(f"Scraping page {page}...")

        df = scrape_reviews(url)

        if df is not None and not df.empty:
            all_reviews.append(df)
        else:
            st.info("No more data found. Stopping scrape.")
            break

        time.sleep(2)
        page += 1

    if all_reviews:
        return pd.concat(all_reviews, ignore_index=True)
    else:
        return None

def scrape_custom_pages(base_url, total_reviews):
    all_reviews = []
    page = 1
    scraped_reviews = 0

    while scraped_reviews < total_reviews:
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
        return pd.concat(all_reviews, ignore_index=True).head(total_reviews)
    else:
        return None

def display_reviews(df):
    if df is not None:
        st.subheader("All Reviews")

        def highlight_negative(val):
            return 'background-color: #f8d7da; color: #721c24;' if val < 3 else ''

        styled_df = df.style.applymap(lambda val: highlight_negative(val) if isinstance(val, (int, float)) else '')
        st.dataframe(styled_df, use_container_width=True)

        negatives = df[df['Rating'] < 3].shape[0]
        positives = df[df['Rating'] >= 3].shape[0]

        st.subheader("Sentiment Analysis")

        labels = ['Positive', 'Negative']
        sizes = [positives, negatives]
        colors = ['#4caf50', '#f44336']

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

        st.subheader("Recommendation")
        if positives / (positives + negatives) > 0.7:
            st.success("The product has a good performance. It's worth considering buying!")
        else:
            st.warning("The product has mixed reviews. Proceed with caution.")

# Streamlit App
st.title("Product Review Scraper")
st.markdown("This tool scrapes product reviews from Flipkart and analyzes them.")

url = st.text_input("Enter the Flipkart product review URL:")

scrape_option = st.radio("Select scraping option:", ("All Reviews", "Custom Reviews"))

total_reviews = None
if scrape_option == "Custom Reviews":
    total_reviews = st.number_input("Enter the total number of reviews to scrape:", min_value=1, step=1)

if st.button("Scrape Reviews"):
    if url:
        if scrape_option == "All Reviews":
            with st.spinner("Scraping all reviews..."):
                reviews_df = scrape_all_pages(url)
                if reviews_df is not None:
                    display_reviews(reviews_df)
                    if st.button("Download Reviews as CSV"):
                        reviews_df.to_csv('reviews.csv', index=False)
                        st.download_button(label="Download CSV", data=reviews_df.to_csv(index=False), file_name='reviews.csv')
        elif scrape_option == "Custom Reviews" and total_reviews:
            with st.spinner(f"Scraping {total_reviews} reviews..."):
                reviews_df = scrape_custom_pages(url, total_reviews)
                if reviews_df is not None:
                    display_reviews(reviews_df)
                    if st.button("Download Reviews as CSV"):
                        reviews_df.to_csv('reviews.csv', index=False)
                        st.download_button(label="Download CSV", data=reviews_df.to_csv(index=False), file_name='reviews.csv')
    else:
        st.error("Please provide a valid URL.")
