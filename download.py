# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")

if __name__ == "__main__":
    download_model()