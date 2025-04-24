from transformers import DistilBertTokenizer, DistilBertModel

# Load pre-trained DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Test with a sample input
text = "Hello, FriendChain AI!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print("DistilBERT loaded successfully!")
print("Output shape:", outputs.last_hidden_state.shape)
