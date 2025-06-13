# 📦 Flipkart Product Review Scraper with Sentiment Analysis

This is a **Streamlit web application** that scrapes customer reviews from Flipkart product pages and performs sentiment analysis using **VADER** and **BERT** models. It helps users quickly understand whether a product is worth buying based on real customer feedback.

---

## 🚀 Features

- 🔗 Accepts any **Flipkart product review URL**
- 🔄 Supports scraping multiple pages of reviews
- 🧠 Performs **sentiment analysis** using:
  - 🔹 VADER (lexicon-based)
  - 🔹 BERT (transformer-based deep learning)
- 📊 Visualizes review distribution with pie charts
- ✅ Recommends whether the product is good based on % of positive reviews
- 📥 Option to **download the results as a CSV file**
- 🎛️ User-friendly interface via Streamlit

---

## 🧪 Technologies Used

- 🐍 Python
- 🧰 Streamlit
- 🌐 BeautifulSoup (for scraping)
- 🤗 HuggingFace Transformers (for BERT)
- 📈 Matplotlib
- 📊 Pandas
- 💬 VADER SentimentIntensityAnalyzer

---

## 💻 How to Run This Project Locally

Follow these steps to set up and run the project on your own system:

### ✅ 1. Clone the Repository

- git clone https://github.com/venkatakoteswarao/flipkart-review-analyzer.git
- cd flipkart-review-analyzer
- after it run the command streamlit run sv.py
