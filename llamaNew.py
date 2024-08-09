import torch
import transformers
from huggingface_hub import login
import spacy

login("hhhhh")
nlp = spacy.load("en_core_web_sm")
class Llama3:
    def __init__ (self, modelPath):
        self.modelID = modelPath
        self.pipeline = transformers.pipeline(
            "text-generation",
            model = self.modelID,
            model_kwargs = {
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )
        
        self.terminator = self.pipeline.tokenizer.eos_token_id
        
    def getResponse (self, query, maxTokens = 4096, temp = 0.6, topP = 0.9):
        userPrompt = [{"role": "system", "content": ""}] + [{"role": "user", "content": query}]
        prompt = self.pipeline.tokenizer.apply_chat_template(
            userPrompt, tokenize=False, add_generation_prompt=True
        )
        outputs = self.pipeline(
            prompt,
            max_new_tokens = maxTokens,
            eos_token_id = self.terminator,
            do_sample = True,
            temperature = temp,
            top_p = topP
        )
        response = outputs[0]["generated_text"][len(prompt):]
        return response
    
    def generateSentences (self):
        while True:
            userInput = input("Ask the AI to write something or write \"theme\" and a theme after for it to generate some sentences based on the theme: ")
            response = ""
            if(userInput.lower().replace("\"", "")[:5] == "theme"):
                response = self.getResponse("Using this theme, generate at least five sentences in paragraph form: " + userInput)
            else:
                response = self.getResponse(userInput)
            print("What Llama cooked up: " + response)
            yorn = input("Is this good (Yes/No)").lower()[0]
            if(yorn == "y"):
                return response
    
def caveManModify ():
    while True:
        output = """i'm going to the supermarket. will it be a good trip? 
        i don't know for sure, but i'll try my best"""
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(output)
        for token in doc:
            if token.tag_ == "VBP" or token.tag_ == "MD":
                output = output.replace(token.text, "", 1)
                continue
            if token.dep_ == "det":
                output = output.replace(token.text, "", 1)
                #convdert these to token.methods so they dont screw up likke string methods do
                continue
            output = output.replace(token.text, token.lemma_, 1)
            #print(token.lemma_)
            print(f"Text: {token.text}\n"
            f"Lemma: {token.lemma_}\n"
            f"POS: {token.pos_}\n"
            f"Tag: {token.tag_}\n"
            f"Dep: {token.dep_}\n"
            f"Shape: {token.shape_}\n"
            f"Is Alpha: {token.is_alpha}\n"
            f"Is Stop: {token.is_stop}\n"
            f"Is Punct: {token.is_punct}\n"
            f"Like Num: {token.like_num}\n"
            f"Ent IOB: {token.ent_iob_}\n"
            f"Ent Type: {token.ent_type_}\n")
        print(output)
        break

if __name__ == "__main__":
    #bot = Llama3("meta-llama/Meta-Llama-3-8B-Instruct")
    caveManModify()
    
"""

Text: 'm, Lemma: be, POS: AUX, Tag: VBP, Dep: aux, Shape: 'x, Is Alpha: False, Is Stop: True, Is Punct: False, Like Num: False, Ent IOB: O, 
Ent Type:

Text: be, Lemma: be, POS: AUX, Tag: VB, Dep: ROOT, Shape: xx, Is Alpha: True, Is Stop: True, Is Punct: False, Like Num: False, Ent IOB: O, Ent Type:



Text: i, Lemma: I, POS: PRON, Tag: PRP, Dep: nsubj, Shape: x, Is Alpha: True, Is Stop: True, 
Is Punct: False, Like Num: False, Ent IOB: O, Ent Type:

Text: it, Lemma: it, POS: PRON, Tag: PRP, Dep: nsubj, Shape: xx, Is Alpha: True, Is Stop: True, 
Is Punct: False, Like Num: False, Ent IOB: O, Ent Type:
"""