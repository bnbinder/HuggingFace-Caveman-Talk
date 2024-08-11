import torch
import transformers
from huggingface_hub import login

login("this token is deleted dont even try")

class Llama3:
    def __init__(self, model_path):
        self.model_id = model_path
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids(""),
        ]
  
    def get_response(
        self, query, message_history=[], max_tokens=4096, temperature=0.6, top_p=0.9
    ):
        user_prompt = message_history + [{"role": "user", "content": query}]
        print("message_history: " + str(message_history))
        print("user_prompt: " + str(user_prompt))
        prompt = self.pipeline.tokenizer.apply_chat_template(
            user_prompt, tokenize=False, add_generation_prompt=True
        )
        print("prompt: " + str(prompt))
    
        # Ensure terminators are set correctly
        terminator = self.pipeline.tokenizer.eos_token_id
        
        # Generate the response
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            eos_token_id=terminator,  # Use single eos_token_id instead of a list
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )
        
        print("outputs: " + str(outputs))
        
        response = outputs[0]["generated_text"][len(prompt):]
        
        print("response: " + response)
        return response, user_prompt + [{"role": "assistant", "content": response}]

    def chatbot(self, system_instructions=""):
        conversation = [{"role": "system", "content": system_instructions}]
        print("conversation: "  + str(conversation))
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the chatbot. Goodbye!")
                break
            response, conversation = self.get_response(user_input, conversation)
            print("response: " + response)
            print("conversation: " + str(conversation))
            
            
            print(f"Assistant: {response}")
  
if __name__ == "__main__":
    bot = Llama3("meta-llama/Meta-Llama-3-8B-Instruct")
    bot.chatbot()
