import openai
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Set your OpenAI API key
openai.api_key = 'openaikey'

def generate_embedding(text, model="text-embedding-3-large"):
  response = openai.embeddings.create(
        input=text,
        model=model
    )
  embeddings = [item.embedding for item in response.data]
  return embeddings

  #return [embedding['embedding'] for embedding in openai.embeddings.create(input = text, model=model).data]
# Example usage:
def get_human_readable_text(prompt):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=prompt,
        max_tokens=100  
    )
    return response.choices[0].message.content.strip()

def main():
    # Define your textbook texts and query
    textbook_texts = [
        "The theory of relativity explains how space and time are linked for objects moving at a constant speed.",
        "Quantum mechanics provides a description of the physical properties of nature at the scale of atoms and subatomic particles.",
        "Newton's laws of motion describe the relationship between the motion of an object and the forces acting on it."
    ]

    query = "Explain the relationship between space and time."

    # Generate embeddings for textbook texts and query
    textbook_embeddings = generate_embedding(textbook_texts)
    query_embedding = generate_embedding([query])[0]

    # Compute similarity between the query and textbook texts
    similarities = cosine_similarity([query_embedding], textbook_embeddings)[0]
    most_similar_index = np.argmax(similarities)
    most_similar_text = textbook_texts[most_similar_index]

    # Convert the most similar text to human-readable format
    messages = [
        {"role": "system", "content": "You are an assistant that provides clear, simple explanations based on the provided text."},
        {"role": "user", "content": f"Please explain the following:\n{most_similar_text}"}
    ]
    
    human_readable_text = get_human_readable_text(messages)

    # Output results
    print("Most Similar Text:")
    print(most_similar_text)
    print("\nHuman Readable Explanation:")
    print(human_readable_text)

if __name__ == "__main__":
    main()