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

1.  Clone the repository.
    ```bash
    git clone https://github.com/Karthikey22/Sentiment-Analysis.git
    cd Sentiment-Analysis
    ```
2.  Install the required packages:

    ```bash
    pip install -r requirements.txt
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
    "text":"Alex told food was terrible"
}
```
**Response (JSON):**
```json
{
    "input_text": "Alex told food was terrible",
    "keywords": [
        "Alex"
    ],
    "summary": "Alex told food was terrible. He was told it was because the food was so bad.",
    "sentiment": "Negative"
}
```

