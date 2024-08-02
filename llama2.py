from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import torch

# Initialize the tokenizer
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B")
    print("Tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading tokenizer: {e}")

# Initialize the model with empty weights and then load the checkpoint with disk offloading
try:
    print("Initializing model with empty weights...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-70B")
    
    print("Loading model checkpoint with disk offloading...")
    model = load_checkpoint_and_dispatch(
        model,
        "meta-llama/Meta-Llama-3.1-70B",  # Specify the path to your checkpoint or model
        device_map="auto",
        offload_folder="./offload"  # Specify the directory for offloading
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define the prompt
prompt = "hello there, how are you?"

# Tokenize the input
try:
    print("Tokenizing input...")
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids
    print(f"Input IDs: {input_ids}")
except Exception as e:
    print(f"Error tokenizing input: {e}")

# Move input to the correct device
try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ids = input_ids.to(device)
    model = model.to(device)
    print(f"Input IDs and model moved to device: {device}")
except Exception as e:
    print(f"Error moving input to the correct device: {e}")

# Generate a response
try:
    print("Generating response...")
    output_ids = model.generate(input_ids, max_length=50, num_return_sequences=1)
    print(f"Output IDs: {output_ids}")
except Exception as e:
    print(f"Error generating response: {e}")

# Decode the generated response
try:
    print("Decoding response...")
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Response: {response}")
except Exception as e:
    print(f"Error decoding response: {e}")
