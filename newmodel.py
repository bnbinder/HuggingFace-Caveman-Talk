import torch
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from huggingface_hub import snapshot_download, login
from pathlib import Path

# Ensure you have the correct versions of these packages

# Path setup for the model
mistral_models_path = Path.home().joinpath('mistral_models', 'Nemo-Instruct')
mistral_models_path.mkdir(parents=True, exist_ok=True)

# Replace 'your_huggingface_token' with your actual token
login(token='hiiiiiiii :3')

# Load the tokenizer and model from Hugging Face
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    # Load the tokenizer and model from Hugging Face
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-Nemo-Instruct-2407")
    print("Tokenizer and model loaded successfully")
except Exception as e:
    print(f"Failed to load tokenizer and model: {e}")
    exit()
# Ensure the model is on the CPU
model.to('cpu')

prompt = "How expensive would it be to ask a window cleaner to clean all windows in Paris. Make a reasonable guess in US Dollar."

# Create a ChatCompletionRequest
completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])

# Encode the completion request
tokens = tokenizer.encode_chat_completion(completion_request).tokens

# Ensure tokens are on the CPU
tokens_tensor = torch.tensor(tokens).unsqueeze(0).to('cpu')

print("Model device:", next(model.parameters()).device)
print("Tokens device:", tokens_tensor.device)

# Generate output tokens
out_tokens, _ = generate([tokens_tensor], model, max_tokens=64, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)

# Ensure output tokens are on the CPU
out_tokens = [token.to('cpu') for token in out_tokens]

# Decode the output tokens
result = tokenizer.decode(out_tokens[0])

print(result)
