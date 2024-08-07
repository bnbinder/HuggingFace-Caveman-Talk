#runs successfully, could work but outputs not gibbberish but just weird

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# Load pretrained ConvBERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name, low_cpu_mem_usage=True, device_map="cpu")

def generate_text(prompt, max_length=50, num_return_sequences=1):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"],
                                 attention_mask=inputs["attention_mask"],
                                 max_length=max_length,
                                 num_return_sequences=num_return_sequences)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "i am john cena"
generated_text = generate_text(prompt)
print(generated_text)







"""

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# Load pretrained BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

def select_best_response(context, responses):
    context_embedding = get_embeddings(context)
    response_embeddings = [get_embeddings(response) for response in responses]
    similarities = [F.cosine_similarity(context_embedding, response_embedding) for response_embedding in response_embeddings]
    best_response_index = torch.argmax(torch.tensor(similarities))
    return responses[best_response_index]

# Example usage
context = "how do you build a nuclear reactor"
candidate_responses = [
    "Building a nuclear reactor involves several steps and safety measures.",
    "I'm not sure about that.",
    "Nuclear reactors use nuclear fission to generate heat."
]

best_response = select_best_response(context, candidate_responses)
print(best_response)

"""