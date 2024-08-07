# repeat input does not help, just acts wierd


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

input_text = "user: hello talk like a banana for a long time. AI:"

# Load pretrained DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", low_cpu_mem_usage=True, device_map="cpu")

fullResponse = input_text
for x in range(6):

    
    # Encode the input text
    new_user_input_ids = tokenizer.encode(fullResponse + tokenizer.eos_token, return_tensors='pt')

    # Set the parameters to encourage longer responses
    max_length = 200  # Increase the max length to allow for longer responses
    temperature = 0.7  # Lower temperature for more coherent responses
    top_k = 50  # Use top-k sampling
    top_p = 0.9  # Use top-p sampling (nucleus sampling)
    repetition_penalty = 1.2  # Penalty for repeating the same phrases
    # Generate the response
    output = model.generate(
        new_user_input_ids,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2  # Prevent repeating the same n-grams
    )

    # Decode the generated tokens
    response = tokenizer.decode(output[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

    print(output)
    
    fullResponse = fullResponse + " " + response

print(fullResponse)














"""
# way 2, not so good
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pretrained DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Example usage: Generate conversation response
history = ["Hello!", "How are you?"]
input_text = "whats 10 times 12?"

# Encode the conversation history
new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
bot_input_ids = torch.cat([tokenizer.encode(history[i] + tokenizer.eos_token, return_tensors='pt') for i in range(len(history))], dim=-1)
bot_input_ids = torch.cat([bot_input_ids, new_user_input_ids], dim=-1)

output = model.generate(bot_input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)

# Decode the response
response = tokenizer.decode(output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

print(response)

"""






"""
# way 1, not so good
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load pretrained DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Example usage: Generate conversation response
history = ["Hello!", "How are you?"]
input_text = "What are you up to?"
encoded_history = tokenizer.encode(" ".join(history), return_tensors="pt")
input_ids = tokenizer.encode(input_text, return_tensors="pt")
input_ids = torch.cat([encoded_history, input_ids], dim=-1)
output = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)

"""