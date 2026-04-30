import os
from loader import Document
from embedder import EmbeddingGenerator
from assistant import Assistant
import pickle 

def run_assistant():
    print("--- Student Assistant ---")
    pickle_file = "storage.pkl"

    if os.path.exists(pickle_file):
        print("loading from disk..") 
        with open(pickle_file,'rb') as f:
            results = pickle.load(f)
    
    else:
        print("First-time setup: Embedding syllabus..")
        doc = Document("docs/course_outline.pdf")
        doc.Reader()
        all_chunks = doc.chunking()
        
        embedder = EmbeddingGenerator()
        results = embedder.generate(all_chunks)

        # Saving for next time
        with open(pickle_file, 'wb') as f:
            pickle.dump(results, f)
        print("Saved to storage.pkl")
    
    #Initializing the Search Assistant
    bot = Assistant(results)
    
    print("\n System Ready! Ask me anything about your course!")
    
    #The Interaction Loop
    while True:
        user_query = input("\nStudent Question (or type 'exit'): ")
        
        if user_query.lower() == 'exit':
            print("Goodbye! Good luck with your studies.")
            break
            
        # Performing Semantic Search
        best_match, score = bot.find_similarity(user_query)
        
        if score > 0.65:
            clean_answer = bot.generate_answer(user_query, best_match)
            print(f"\nAssistant: {clean_answer}")
        else:
            print("\nAssistant: I'm not quite sure. Please check the course outline or contact Dr. Totle.")

if __name__ == "__main__":
    run_assistant()