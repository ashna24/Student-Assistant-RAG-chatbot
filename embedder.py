from loader import Document 
from google import genai
import os 

class EmbeddingGenerator:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")

        if not api_key:
            print("No api key found!")
        else:
            print("success!")

        self.google_client = genai.Client(api_key = api_key)

    def generate(self, chunk):
        vector_list= []
        for i, chunked in enumerate(chunk):
            response = self.google_client.models.embed_content(
                model= "gemini-embedding-001",
                contents = chunked,
                config = {'task_type' : 'RETRIEVAL_DOCUMENT'}
            )

            vector = response.embeddings[0].values

            vector_list.append({
                "id" :i,
                "text" : chunked,
                "vector": vector
            })

            print(f"Chunk {i} | Text Preview: {chunked[:50]}... | Vector Size: {len(vector)}")
        return vector_list


            
        
