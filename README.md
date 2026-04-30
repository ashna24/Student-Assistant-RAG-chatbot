## AI Student Assistant (RAG)

A Retrieval-Augmented Generation (RAG) system built from scratch. It allows students to chat with their syllabus using Semantic Search.

##  How it Works
- **Manual Vector Search**: Uses NumPy to calculate Cosine Similarity manually.
- **Top-K Retrieval**: Fetches multiple context chunks to ensure accuracy.
- **Persistence**: Uses Pickle to cache embeddings and save API costs.
- **Tech Stack**: Python, Google Gemini API, NumPy.

##  Setup
1. Clone this repo.
2. Export your `GOOGLE_API_KEY`.
3. Run `python main.py`.
