from embedder import EmbeddingGenerator
import os 
import numpy as np 
from google import genai

class Assistant:
    def __init__(self, results):
        self.results = results
        api_key = os.getenv("GOOGLE_API_KEY")
        self.google_client = genai.Client(api_key=api_key)

    def convert_query(self, user_query):
        query_response = self.google_client.models.embed_content(
            model="gemini-embedding-001",
            contents=user_query,
            config={'task_type': 'RETRIEVAL_QUERY'}
        )
        return query_response.embeddings[0].values

    def find_similarity(self, user_query):
        # Converts the user query into a vector
        vec_user_query = self.convert_query(user_query)
        vec1 = np.array(vec_user_query) 

        all_scores = []

        # Looping through all stored vectors and calculaing Cosine Similarity
        for item in self.results:
            vec2 = np.array(item["vector"])
            # Manual Cosine Similarity formula
            score = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            all_scores.append((score, item["text"]))
        
        # Sort by score highest first and take the top 2 chunks
        all_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Combine the top 2 matches with a separator
        best_context = "\n---\n".join([all_scores[0][1], all_scores[1][1]])
        top_score = all_scores[0][0]
        
        return best_context, top_score

    def generate_answer(self, user_query, retrieved_context):
        try:
            prompt = f"""
            You are a helpful student assistant for the CS301 Artificial Intelligence course.
            Based ONLY on the context provided below, answer the student's question clearly.
            If the information isn't there, tell them to contact Dr. Aris Totle.
            
            Context: {retrieved_context}
            Student Question: {user_query}

            Answer:
            """

            response = self.google_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            return response.text
        
        except Exception as e:
            # error handling for API Rate Limits
            return f"I found this in the syllabus: {retrieved_context[:300]}... [Note: AI Generator is busy, try again in 10s]"