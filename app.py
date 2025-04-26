import json
import time
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertModel, T5Tokenizer, T5ForConditionalGeneration

# Initialize FastAPI app
app = FastAPI()

# Define input model
class ProfileInput(BaseModel):
    userId: str
    basename: str
    bio: str
    posts: list[str]

# Load models
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
t5_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
t5_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")

# Prompt templates for diversity
PROMPT_TEMPLATES = [
    ("interest", "question: What is a key interest of {basename} based on: {input_text} [HL] {topic} [HL]"),
    ("technology", "question: What technology is discussed in: {input_text} [HL] {topic} [HL]"),
    ("blockchain", "question: What blockchain platform is mentioned in the posts: {input_text} [HL] {topic} [HL]"),
    ("location", "question: What location is mentioned in: {input_text} [HL] {topic} [HL]")
]

# Fallback questions for variety
FALLBACK_QUESTIONS = [
    ("What topic does {basename} frequently post about?", ["Tech", "Sports", "Music", "Food"], 0),
    ("What city is {basename} based in?", ["San Francisco", "New York", "London", "Tokyo"], 0),
    ("What field is {basename} passionate about?", ["AI", "Blockchain", "Web3", "Gaming"], 0)
]

# Endpoint to generate 15 questions
@app.post("/generateQuestions")
async def generate_questions(profile: ProfileInput):
    start_time = time.time()

    # Truncate profile text
    bio = profile.bio[:100]
    posts = ' '.join(profile.posts)[:150]
    profile_text = f"Bio: {bio} Posts: {posts}"

    # Extract topics (mocked for now)
    inputs = distilbert_tokenizer(profile_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    detected_topic = "tech"  # Static, refine in Step 7

    # Track question types to limit repetition
    question_counts = {"interest": 0, "technology": 0, "blockchain": 0, "location": 0}
    max_per_type = 4  # Limit each type to 4 occurrences

    questions = []
    for stage in range(1, 4):
        for qid in range(1, 6):
            # Select prompt, prioritizing underused types
            available_types = [t for t, _ in PROMPT_TEMPLATES if question_counts[t] < max_per_type]
            if not available_types:
                available_types = [t for t, _ in PROMPT_TEMPLATES]  # Fallback if all maxed
            prompt_type, prompt_template = next((t, p) for t, p in PROMPT_TEMPLATES if t == available_types[(stage + qid) % len(available_types)])
            question_counts[prompt_type] += 1

            input_text = f"{profile_text}"
            prompt = prompt_template.format(input_text=input_text, topic=detected_topic, basename=profile.basename)
            inputs = t5_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = t5_model.generate(**inputs, max_length=100, num_return_sequences=1)
            generated_text = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse output
            try:
                question_part = generated_text.strip()
                if not question_part.endswith("?"):
                    question_part += "?"
                # Select options based on question or prompt
                if prompt_type == "blockchain" or "hackathon" in question_part.lower() or "base" in question_part.lower():
                    options = ["Base", "Ethereum", "Polygon", "Solana"]
                    correct_answer = 0  # Base is in posts
                elif prompt_type == "technology" or "ai" in question_part.lower() or "distilbert" in question_part.lower():
                    options = ["DistilBERT", "TensorFlow", "PyTorch", "Scikit-learn"]
                    correct_answer = 0  # DistilBERT is in posts
                elif prompt_type == "location" or "sf" in question_part.lower() or "city" in question_part.lower():
                    options = ["San Francisco", "New York", "London", "Tokyo"]
                    correct_answer = 0  # Bio mentions SF
                elif prompt_type == "interest":
                    options = ["AI", "Blockchain", "Web3", "Gaming"]
                    correct_answer = 0  # Bio mentions AI
                else:
                    options = ["Tech", "Sports", "Music", "Food"]
                    correct_answer = 0
                if len(question_part) < 10 or not question_part.startswith("What"):
                    raise ValueError("Invalid question")
            except:
                # Use fallback questions
                fallback_idx = (stage * 5 + qid) % len(FALLBACK_QUESTIONS)
                question_part, options, correct_answer = FALLBACK_QUESTIONS[fallback_idx]
                question_part = question_part.format(basename=profile.basename)

            question = {
                "stage": stage,
                "questionId": (stage - 1) * 5 + qid,
                "questionText": question_part,
                "options": options,
                "correctAnswer": correct_answer,
                "personalizedContext": f"Based on {profile.basename}'s X posts"
            }
            questions.append(question)

    # Measure generation time
    elapsed_time = time.time() - start_time
    print(f"Generated 15 questions in {elapsed_time:.2f} seconds")

    return {"questions": questions}

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)