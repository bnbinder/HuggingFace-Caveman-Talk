# weird responses, but does mulitple entences, usually repeating or very much like the first sentence

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