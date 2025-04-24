import json
import torch
from transformers import DistilBertTokenizer, DistilBertModel, T5Tokenizer, T5ForConditionalGeneration

# Load mock profile
with open("mock_profiles.json", "r") as file:
    profiles = json.load(file)
profile = profiles[0]  # Use alex.base’s profile

# Truncate profile text
bio = profile['bio'][:100]
posts = ' '.join(profile['posts'])[:150]
profile_text = f"Bio: {bio} Posts: {posts}"

# Load DistilBERT for topic extraction
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Extract topics (mocked for now)
inputs = distilbert_tokenizer(profile_text, return_tensors="pt", truncation=True, max_length=512)
with torch.no_grad():
    outputs = distilbert_model(**inputs)
embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
detected_topic = "tech"  # Static, will refine in Step 7

# Load T5-base question generation model
t5_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
t5_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")

# Generate question
input_text = f"{profile_text} [HL] {detected_topic} [HL]"
prompt = f"question: {input_text}"
inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
outputs = t5_model.generate(**inputs, max_length=100, num_return_sequences=1)
generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Debug: Print raw T5 output
print("Raw T5 output:", generated_text)

# Parse output
try:
    # Expect: A valid question
    question_part = generated_text.strip()
    if not question_part.endswith("?"):
        question_part += "?"
    # Generate options based on question content
    if "hackathon" in question_part.lower() or "base" in question_part.lower():
        options = ["Base", "Ethereum", "Polygon", "Solana"]  # Blockchain-related
    else:
        options = [detected_topic.capitalize(), "Sports", "Music", "Food"]
    # Validate question length
    if len(question_part) < 10:
        raise ValueError("Invalid question")
except:
    # Fallback
    question_part = f"What topic does {profile['basename']} frequently post about?"
    options = ["Tech", "Sports", "Music", "Food"]

# Format JSON output
question = {
    "stage": 1,
    "questionId": 1,
    "questionText": question_part,
    "options": options,
    "correctAnswer": 0,  # First option is correct
    "personalizedContext": f"Based on {profile['basename']}'s X posts"
}

# Print the question
print(json.dumps(question, indent=2))