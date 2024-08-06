import torch
import transformers
from huggingface_hub import login

login("eeee")

class Llama3:
    def __init__(self, model_path):
        self.model_id = model_path
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
            },
        )
        eos_token_id = self.pipeline.tokenizer.eos_token_id
        empty_token_id = self.pipeline.tokenizer.convert_tokens_to_ids("")
        
        # Ensure terminators don't contain None values
        self.terminators = [eos_token_id] if eos_token_id is not None else []
        if empty_token_id is not None:
            self.terminators.append(empty_token_id)
  
    def get_response(
        self, query, message_history=[], max_tokens=4096, temperature=0.6, top_p=0.9
    ):
        print("before user_prompt")
        user_prompt = message_history + [{"role": "user", "content": query}]
        print("before prompt")
        prompt = self.pipeline.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        print("before outputs")
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        print("before response")
        response = outputs[0]["generated_text"][len(prompt):]
        print("before return")
        return response, user_prompt + [{"role": "assistant", "content": response}]
    
    def chatbot(self, system_instructions=""):
        conversation = [{"role": "system", "content": system_instructions}]
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chatbot. Goodbye!")
                break
            print("before running get response")
            response, conversation = self.get_response(user_input, conversation)
            print("after running get response")
            print(f"Assistant: {response}")
  
if __name__ == "__main__":
    bot = Llama3("meta-llama/Meta-Llama-3-8B-Instruct")
    bot.chatbot()
