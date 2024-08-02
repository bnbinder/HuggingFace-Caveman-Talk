import torch
import transformers
from huggingface_hub import login

class Llama3:
    def __init__(self, model_path, api_key):
        # Log in to Hugging Face using the API key
        login(api_key)
        
        self.model_id = model_path
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.float16},  # No quantization
)

        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids(""),
        ]

    def get_response(self, prompt, max_tokens=4096, temperature=0.6, top_p=0.9):
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        response = outputs[0]["generated_text"]
        return response

if __name__ == "__main__":
    api_key = "hihiihihihi"
    model_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    prompt = "Hello, how can I assist you today?"

    bot = Llama3(model_path, api_key)
    response = bot.get_response(prompt)
    print(f"Response: {response}")
