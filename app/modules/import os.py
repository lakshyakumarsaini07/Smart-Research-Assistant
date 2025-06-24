import os
from together import Together

# Initialize Together client
together = Together(api_key=os.getenv("TOGETHER_API_KEY"))
async def generate_response(prompt, context=""):
    try:
        response = together.complete(
            prompt=f"{context}\n{prompt}",
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Free high-quality model
            max_tokens=1000,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.1
        )
        return response['output']['text']
    except Exception as e:
        print(f"Error generating response: {e}")
        return None