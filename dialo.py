#sucks as well, does not do multiple sentences and gives 
#same responses for same questions. is coherent and has legible responses tho

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

input_text = "speak like a caveman"

# Load pretrained DialoGPT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
output = model.generate(new_user_input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(output[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

print(response)















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