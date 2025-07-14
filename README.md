# Document Summarization with LangChain and OpenAI

This project demonstrates and compares two different document summarization techniques using the LangChain library and OpenAI's language models:

1.  **Stuffing:** A simple approach where the entire document is passed to the language model in a single call. This is suitable for smaller documents that fit within the model's context window.
2.  **Map-Reduce:** A more advanced technique for handling larger documents. It involves three steps:
    *   **Map:** The document is split into smaller chunks, and each chunk is summarized individually.
    *   **Combine:** The individual summaries are then combined.
    *   **Reduce:** The combined summaries are further summarized to create a final, coherent summary.

## Project Structure

```
.
├── .env
├── .gitignore
├── compare_summarizers.py
├── mapreduce_summarizer.py
├── README.md
├── requirements.txt
└── stuff_summarizer.py
```

*   `compare_summarizers.py`: This script runs both the "stuff" and "map-reduce" summarizers and prints the results for comparison.
*   `mapreduce_summarizer.py`: This script implements the map-reduce summarization technique.
*   `stuff_summarizer.py`: This script implements the "stuff" summarization technique.
*   `requirements.txt`: This file lists the Python dependencies required to run the project.
*   `.env`: This file stores the OpenAI API key.

## Getting Started

### Prerequisites

*   Python 3.8 or higher
*   An OpenAI API key

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/wassupjay/Summarization-chatbot.git
    cd Summarization-chatbot
    ```

2.  **Create a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**

    Create a `.env` file in the root of the project and add your OpenAI API key:

    ```
    OPENAI_API_KEY="your-openai-api-key"
    ```

## Usage

To run the summarization comparison, execute the `compare_summarizers.py` script:

```bash
python compare_summarizers.py
```

This will fetch the content of the specified web page, generate summaries using both the "stuff" and "map-reduce" methods, and print the results to the console.

## Example Output

```
============================================================
COMPARING SUMMARIZATION APPROACHES
============================================================

Document chunks: 1
Total documents: 1

============================================================
STUFF SUMMARIZER (Single Pass)
============================================================
Jayanth's blog discusses a project focused on enhancing signup conversion rates using machine learning and A/B testing. The project involves training a random forest model to predict user conversions and personalizing call-to-action messages based on the model's predictions. A simulated dataset of 10,000 user sessions is created to train the model, after which an A/B test is conducted to validate the effectiveness of personalization. Sattineni also emphasizes the importance of statistical validation for growth ideas, highlighting A/B testing as a key tool for assessing the impact of changes. Future steps involve applying real user data and deploying the model for real-time personalization.

============================================================
MAP-REDUCE SUMMARIZER (Multi-Pass)
============================================================
Jayanth's article highlights a project that leverages machine learning, specifically a Random Forest model, and A/B testing to improve signup conversion rates. The project involves predicting user conversions and personalizing call-to-action (CTA) messages based on user behavior. Key steps include data simulation, model training, CTA customization, and A/B testing to assess effectiveness, with results analyzed using statistical methods. The findings underscore the significance of A/B testing in confirming the success of personalization strategies. Recommendations for the future include utilizing real user data and establishing live dashboards for continuous performance monitoring.

============================================================
COMPARISON NOTES
============================================================
• Stuff Summarizer: Single pass, processes all content at once
• Map-Reduce Summarizer: Multi-pass, summarizes chunks then combines
• Map-Reduce is better for very long documents that exceed token limits
• Stuff Summarizer is simpler and faster for shorter documents
```

## How It Works

### Stuff Summarizer

The "stuff" summarizer is the most straightforward approach. It takes the entire document and "stuffs" it into the prompt that is sent to the language model. This method is fast and simple, but it has a major limitation: it can only handle documents that are smaller than the language model's context window.

### Map-Reduce Summarizer

The map-reduce summarizer is designed to handle large documents that exceed the context window of the language model. It works in two stages:

1.  **Map:** The large document is split into smaller chunks. Each chunk is then sent to the language model to be summarized individually.
2.  **Reduce:** The summaries of the individual chunks are then combined into a single document. This combined document is then sent to the language model to be summarized again, producing the final summary.

This approach allows you to summarize documents of any size, but it is more complex and can be slower than the "stuff" summarizer.

## Conclusion

This project provides a practical demonstration of two different document summarization techniques. The "stuff" summarizer is a good choice for small documents, while the map-reduce summarizer is a more robust solution for handling large documents. By understanding the trade-offs between these two approaches, you can choose the best summarization technique for your specific needs.
