import torch
import transformers
from huggingface_hub import login
import spacy
from spacy.tokens import Doc, Span
import re
from sentence_transformers import SentenceTransformer, util
import textstat
from gensim.models import KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import sys
import os
import warnings
from rich.progress import track
from time import sleep
from rich import print
from datetime import datetime
from rich.progress import Progress
from rich.console import Console
from rich.panel import Panel

def log_message(task):
    current_time = datetime.now().strftime('%H:%M:%S')
    log_text = f"[{current_time}] {task}"
    print(log_text)
    
def log_time():
    current_time = datetime.now().strftime('%H:%M:%S')
    log_text = f"[{current_time}]"
    print(log_text, end="")

warnings.filterwarnings('ignore', module='transformers')
warnings.filterwarnings('ignore', module='bitsandbytes')

log_time()
login("eee")
log_message("Login To HuggingFace Account Successful")

word2vec = Word2Vec(
    common_texts, 
    vector_size=100, 
    window=5, 
    min_count=1, 
    workers=4
)
wordVectors = word2vec.wv
model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")
resultT = ""
outputT = ""
console = Console()
list = [
    "Prompt Set", 
    "Chat Template Applied", 
    "Output Pipeline Done", 
    "Output Recorded"
]
console = Console()

class Llama3:
    def __init__ (self, modelPath):

        log_time()
        device = 0 if torch.cuda.is_available() else -1
        self.pipeline = transformers.pipeline(
            "text-generation",
            model = modelPath,
            device = device,
            model_kwargs = {
                "torch_dtype": torch.float16,
                "quantization_config": {"load_in_4bit": True},
                "low_cpu_mem_usage": True,
            },
        )
        log_message("Model Pipeline Loaded")        
    
        self.terminator = self.pipeline.tokenizer.eos_token_id   
        log_message("Terminator Loaded")
        
    def getResponse (self, query, maxTokens = 4096, temp = 0.6, topP = 0.9):
        userPrompt = [{"role": "system", "content": ""}] + [{"role": "user", "content": query}]
                    
        task = list[0]
        console.log(f"{task}")
        
        prompt = self.pipeline.tokenizer.apply_chat_template(
            userPrompt, 
            tokenize = False, 
            add_generation_prompt = True
        )
                    
        task = list[1]
        console.log(f"{task}")
        
        print("hi", end="")
        outputs = self.pipeline(
            prompt,
            max_new_tokens = maxTokens,
            eos_token_id = self.terminator,
            do_sample = True,
            temperature = temp,
            top_p = topP
        )
                    
        task = list[2]
        console.log(f"{task} complete")
        
        response = outputs[0]["generated_text"][len(prompt):]    
        
        task = list[3]
        console.log(f"{task} complete")

        return response
    
    def generateSentences (self):
        global resultT
        
        while True:
            userInput = input("Ask the AI to write something or write \"theme\" and a theme after for it to generate some sentences based on the theme: ")
            log_message("Input Recieved")
            response = ""
            
            if userInput.lower().replace("\"", "")[:5] == "theme":
                response = self.getResponse("Using this theme, generate at least five sentences in paragraph form. Do not include any introduction sentence responding to my prompt, only respond with the sentences generated. So something like \"Here are five sentences in paragraph form:\" do not include: " + userInput)
            else:
                response = self.getResponse("Do not include any introduction sentence responding to my prompt, only respond with the sentences generated. So something like \"Here are five sentences in paragraph form:\" do not include: " + userInput)

            llamaCookedUp = Panel(
                response, 
                title = "What Llama Cooked Up", 
                expand = False
            )
            
            print(llamaCookedUp)
            
            yorn = input("Is this good (Yes/No)").lower()[0]
            
            if yorn == "y":
                resultT = response
                return response
            elif yorn == "n":
                print("Printing new response")
            elif yorn == None:
                print("Try again")
            else:
                print("That wasnt a yes or no")
                
    def llamaCaveman (self, text):
        response = self.getResponse("Using this paragraph of text, make this text sound like a caveman wrote it. Do not include any introduction sentence responding to my prompt, only respond with the sentences generated. So something like \"Here are five sentences in paragraph form:\" do not include: " + text)
        llamaCookedUp = Panel(
            response, 
            title="What Caveman Llama Cooked Up", 
            expand=False
        )
            
        print(llamaCookedUp)

        return response
    
    def cleanup (self, words):
        t = ' '.join(words).capitalize()
        t = re.sub(r'\s+', ' ', t)
        t = re.sub(r' \.', '.', t)
        t = re.sub(r' \?', '?', t)
        t = re.sub(r' \'', '', t)
        t = re.sub(r' \,', ',', t)
        t = t.strip()
        log_message("Text Cleaned")
        return t
        
    def lemmasDoc (self, text, nlp):
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc]
        simpWord = self.cleanup(lemmas)
        doc = nlp(simpWord)
        log_message("Doc Cleaned")

        return doc
    
    def caveManModify (self):
        global outputT
        
        while True:
            words = []
            output = self.generateSentences()
            doc = self.lemmasDoc(output, nlp)
            total_tokens = len(doc) 
            
            with Progress() as progress:
                task = progress.add_task("Processing tokens...", total=total_tokens)
                
                for i, token in enumerate(doc, start=1):
                    progress.update(task, advance=1, description=f"Processing tokens... [{i}/{total_tokens}]")
                    
                    if token.tag_ in ["MD", "JJR", "JJS"]:
                        continue
                    if token.dep_ in ["det"]:
                        continue
                    if token.text in ["my", "to"]:
                        continue
                    if token.pos_ == "ADJ" and token.text != token.lemma_[:len(token.text) - 1]:
                        words.append(token.text)
                        continue
                    
                    words.append(token.lemma_)
            
            log_message("Tokens Cavemanized")
            simpWord = self.cleanup(words)
            iCookedUp = Panel(
                simpWord, 
                title="What I Cooked Up", 
                expand=False
            )
            
            print(iCookedUp)
            
            outputT = simpWord

            break

class Similarity:
    # encodes the texts to vectors and compares them using cosine dot product formula
    def computeSimilarity(self, text1, text2):
        embeddings1 = model.encode(text1, convert_to_tensor=True)
        embeddings2 = model.encode(text2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        log_message("Similarity Using Dot Product Method Done")

        return similarity.item()

    # uses flesch kincaid formula to compute what grade level the text is at
    def getReadability(self, text):
        log_message("Similarity Using Kincaid Method Done")

        return textstat.flesch_kincaid_grade(text)
        
    # uses simple method to see whether any tokens are in the complexWords token list
    def isCavemanLike(self, text):
        doc = nlp(text)
        complexWords = {"my", "to", "the", "a", "an", "was", "were", "had"}

        if any(token.text.lower() in complexWords for token in doc):
            log_message("Similarity Using Simple Method Done")
            return False

        log_message("Similarity Using Simple Method Done")

        return True

    # compares ngrams of candidate text to ngrams of reference texts and sees if they are similar or not (using 3 ngram and smoothing function)
    def calculateBleu(self, reference, candidate):
        reference = [reference.split()]
        candidate = candidate.split()
        blueScore = sentence_bleu (
            reference, 
            candidate, 
            weights=(1/3, 1/3, 1/3), 
            smoothing_function = SmoothingFunction().method1
        )
        log_message("Similarity Using Bleu Method Done")

        return blueScore

    # compares ngrams of candidate text to ngrams of reference text, but uses 1 ngrams, 2 ngrams, longest common subsequence between the two, 
    def calculateRouge(self, reference, candidate):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        log_message("Similarity Using Rouge Method Done")
        return scores

    def wordMoversDistance(self, text1, text2):
        wordScore = wordVectors.wmdistance(text1.split(), text2.split())
        log_message("Similarity Using WMD Done")
        return wordScore

if __name__ == "__main__":
    bot = Llama3("meta-llama/Meta-Llama-3-8B-Instruct")
    bot.caveManModify()
    
    sim = Similarity()
    
    referenceText = resultT
    generatedText = outputT
    betterGeneratedText = bot.llamaCaveman(referenceText)
    
    similarityScoreRG = sim.computeSimilarity(referenceText, generatedText)
    similarityScoreRB = sim.computeSimilarity(referenceText, betterGeneratedText)
    similarityScoreGB = sim.computeSimilarity(betterGeneratedText, generatedText)
    
    referenceScore = sim.getReadability(referenceText)
    generatedScore = sim.getReadability(generatedText)
    betterScore = sim.getReadability(betterGeneratedText)
    
    referenceCheck = sim.isCavemanLike(referenceText)
    generatedCheck = sim.isCavemanLike(generatedText)
    betterCheck = sim.isCavemanLike(betterGeneratedText)
    
    bleuScoreRG = sim.calculateBleu(referenceText, generatedText)
    bleuScoreRB = sim.calculateBleu(referenceText, betterGeneratedText)
    bleuScoreGB = sim.calculateBleu(betterGeneratedText, generatedText)
    
    rougeScoresRG = sim.calculateRouge(referenceText, generatedText)
    rougeScoresRB = sim.calculateRouge(referenceText, betterGeneratedText)
    rougeScoresGB = sim.calculateRouge(betterGeneratedText, generatedText)
    
    distanceRG = sim.wordMoversDistance(referenceText, generatedText)
    distanceRB = sim.wordMoversDistance(referenceText, betterGeneratedText)
    distanceGB = sim.wordMoversDistance(betterGeneratedText, generatedText)