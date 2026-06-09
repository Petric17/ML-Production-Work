import os 
from transformers import MarianMTModel, MarianTokenizer
import torch

def load_translator():
    model_name = "Helsinki-NLP/opus-mt-roa-en"
    print("Initializing Transformer Weights from Local Cache...")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer

def translate_line(line, model, tokenizer):
    if line.strip() == "":
        return ""
    # Tokenize input string
    input_tokens = tokenizer(line, return_tensors="pt", padding=True)
    # Generate translation vectors
    with torch.no_grad():
        output_tokens = model.generate(**input_tokens)
    # Decode back to raw string characters
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)

# Main Execution Loop
if __name__ == "__main__":
    print("=====================================================")
    print("      Live Transformer Lyric Translator Terminal     ")
    print("=====================================================")
    
    # Load model once into RAM memory
    model, tokenizer = load_translator()
    
    print("\nSystem Ready! The Transformer is listening.")
    print("Paste your lyrics line below (or type 'exit' to quit):\n")
    
    while True:
        user_input = input("Enter Lyric Line: ")
        
        if user_input.strip().lower() == 'exit':
            print("Closing translation terminal. Goodbye!")
            break
            
        if user_input.strip() == "":
            continue
            
        # Run translation instantly
        english_output = translate_line(user_input, model, tokenizer)
        
        print(f"AI Translation : {english_output}")
        print("-" * 50)