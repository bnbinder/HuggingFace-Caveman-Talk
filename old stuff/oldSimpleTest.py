# DOESNT WORK, DONT KNOW WHY, NOT USING SIMPLE TRANSFORMER PROBABLY
from simpletransformers.conv_ai import ConvAIModel
import torch

# Check if CUDA is available
use_cuda = torch.cuda.is_available()

# Create a ConvAIModel
model = ConvAIModel("gpt", "microsoft/DialoGPT-medium", use_cuda=use_cuda, args={"max_length": 50, "num_beams": 5, "temperature": 0.7})

input_text = "Hello, how are you?"

response = model.interact(input_text)

personality=[
    "My name is Geralt.",
    "I hunt monsters.",
    "I say hmm a lot.",
]

# Single interaction
history = [
    "Hello, what's your name?",
    "Geralt",
    "What do you do for a living?",
    "I hunt monsters",
]

response, history = model.interact_single(
    "Is it dangerous?",
    history,
    personality=personality
)

# Remove leading Ġ and split by Ġ
tokens = response.lstrip("Ġ").split("Ġ")

# Join tokens into a coherent string
clean_response = " ".join(tokens)

print(clean_response)
