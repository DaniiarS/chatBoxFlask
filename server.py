from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import json

device = torch.device('cpu')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = torch.load('model.pt', map_location=device)

def generate_response(model, query):
  model.eval()

  encoding_test = tokenizer('<|startoftext|>'+query+'<|question|>', truncation=True, return_tensors='pt')
  test_input = encoding_test['input_ids'].to(device)
  test_attn_mask = encoding_test['attention_mask'].to(device)

  output = model.generate(test_input, attention_mask=test_attn_mask, max_length=len(query)+50, pad_token_id=tokenizer.eos_token_id, top_k=50, temperature=0.8, do_sample=True)
  query_length = len(query) + len('<|startoftext|><|question|>')
  return tokenizer.decode(output[0], skip_special_token=True)[len('<|startoftext|>')+len('<|question|>')+len(query):]


app = Flask(__name__)

@app.route("/")
def hello():
	var = "python sends an argument"
	return render_template("main.html", arg=json.dumps(var) )


@app.route('/process_data', methods=['POST'])
def process_data():
    print("Received a request!")
    data_from_js = request.get_json()
    print("Data from JavaScript:", data_from_js['value'])

    result = data_from_js['value']
    
    answer = generate_response(model, result)
    

    return jsonify(result=answer)

if __name__ == "__main__":
	# this will help to run the file from termianl directly: python3 server.py
	app.run(debug=True)