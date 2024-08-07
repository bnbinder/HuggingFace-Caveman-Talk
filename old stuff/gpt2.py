# weird responses, but does mulitple entences, usually repeating or very much like the first sentence




from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Set random seed for reproducibility (or vary it for different results)
seed = 40
torch.manual_seed(seed)


# Load pretrained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', low_cpu_mem_usage=True, device_map="cpu")

max_length=50 # Adjust the max_length to a smaller value
num_return_sequences=1
no_repeat_ngram_size=2 # Avoid repeating n-grams of 2 tokens
top_k=50  # Use top-k sampling
top_p=0.95 # Use top-p (nucleus) sampling
repetition_penalty=1.5  # Apply a repetition penalty
temperature=0.7  # Increase the temperature for more randomness
do_sample=True

# Example usage: Generate text
input_text = "User: Can you have a banana?\nAI:"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(
    input_ids,
    max_length=max_length,  # Adjust the max_length to a smaller value
    num_return_sequences=num_return_sequences,
    no_repeat_ngram_size=no_repeat_ngram_size,  # Avoid repeating n-grams of 2 tokens
    top_k=top_k,  # Use top-k sampling
    top_p=top_p,  # Use top-p (nucleus) sampling
    repetition_penalty=repetition_penalty,  # Apply a repetition penalty
    temperature=temperature,  # Increase the temperature for more randomness
    pad_token_id=tokenizer.eos_token_id,  # Use the EOS token for padding
    do_sample=do_sample
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)









"""
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pretrained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Example usage: Generate text
input_text = "can you have a banana"
input_ids = tokenizer.encode(input_text, return_tensors='pt')
output = model.generate(input_ids, max_length=200, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
"""
