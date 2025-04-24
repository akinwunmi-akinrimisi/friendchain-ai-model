import json
from transformers import pipeline

# Load mock profile
with open("mock_profiles.json", "r") as file:
    profiles = json.load(file)
profile = profiles[0]  # Use alex.base’s profile

# Combine bio and posts for input
profile_text = f"Basename: {profile['basename']}\nBio: {profile['bio']}\nPosts: {' '.join(profile['posts'])}"

# Load question generation pipeline
question_generator = pipeline("text2text-generation", model="t5-small")

# Generate a trivia question
prompt = f"Generate a trivia question with 4 answer options based on this profile: {profile_text}"
result = question_generator(prompt, max_length=100, num_return_sequences=1)

# Mock formatting for JSON output (we’ll refine this later)
question = {
    "stage": 1,
    "questionId": 1,
    "questionText": result[0]["generated_text"],
    "options": ["Tech", "Sports", "Music", "Food"],  # Static for now
    "correctAnswer": 0,  # Static for now
    "personalizedContext": f"Based on {profile['basename']}'s X posts"
}

# Print the question
print(json.dumps(question, indent=2))