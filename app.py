from potassium import Potassium, Request, Response

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Potassium("TinyStories-33M")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M")
    model.to(device)

    context = {
        "model": model,
        "tokenizer": tokenizer
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    tokenizer = context.get("tokenizer")

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    outputs = model.generate(input_ids, max_length = 1000, num_beams=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return Response(
        json = {"outputs": text}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()
