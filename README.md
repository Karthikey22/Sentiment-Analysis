# Sentiment Analysis and Text Processing API

This FastAPI application provides endpoints for processing text, including summarization, named entity recognition (NER) for keyword extraction, and sentiment analysis. It also maintains a history of processed texts and allows searching through the history.

## Features

*   **Text Processing:** Summarizes input text, extracts keywords (named entities), and performs sentiment analysis (positive, neutral, negative).
*   **History Tracking:** Stores processed text, keywords, summaries, and sentiments, allowing users to retrieve past results.
*   **Search Functionality:** Enables searching through the history based on keywords or input text.
*   **History Clearing:** Provides an endpoint to clear the processing history.

## Getting Started

### Prerequisites

*   Python 3.7+
*   Required Python packages: `fastapi`, `uvicorn`, `torch`, `transformers`, `pydantic`

### Installation

1.  Clone the repository (if applicable).
2.  Install the required packages:

    ```bash
    pip install fastapi uvicorn torch transformers pydantic
    ```

3.  Download the necessary pre-trained models. The code will attempt to download them automatically upon the first run, but you might need a stable internet connection.

## Running the Application

1.  python .\app.py

## API Endpoints

### 1. `/process` (POST)

Processes the input text.

**Request Body (JSON):**

```json
{
  "text": "The input text to be processed."
}

# SENTIMENT ANALYSIS

This is a RESTful API built with **FastAPI** that processes text using a **pre-trained Large Language Model (LLM)**. It provides:
- **Summarization**
- **Named Entity Recognition (NER)**
- **Sentiment Analysis**

## 🚀 Features
- Uses **Facebook BART** for summarization.
- Uses **BERT-CRF** for Named Entity Recognition.
- Uses **BERT Multilingual Sentiment** model for sentiment analysis.
- Implements **FastAPI** for fast and easy deployment.

## 📌 Endpoints

### **1️⃣ Process Text**
- **URL**: `POST /process`
- **Request:**
```json
{
    "text": "The food at the restaurant was amazing!"
}
