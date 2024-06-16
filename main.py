import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# Load Hugging Face API token from config.json
with open("config.json", "r") as f:
    config_data = json.load(f)

HF_TOKEN = config_data["HF_TOKEN"]

model_name = "meta-llama/Meta-Llama-3-8B"

# Configure model and tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device="cuda" if torch.cuda.is_available() else "cpu",
    quantization_config=bnb_config,
    token=HF_TOKEN
)

# Set up text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128
)

# Function to generate response
def get_response(prompt):
    sequences = text_generator(prompt)
    gen_text = sequences[0]["generated_text"]
    return gen_text

# Infinite chat loop
def chat_with_llama3():
    """
    Starts an infinite chat loop with Llama-3.
    The function welcomes the user and continuously prompts for input.
    If the user types 'exit', the chat ends and a goodbye message is printed.
    Otherwise, the user's input is passed to the `get_response` function,
    which generates a response from Llama-3 using the text generation pipeline.
    The generated response is then printed.

    Parameters:
    None

    Returns:
    None

    Raises:
    None
    """
    print("Welcome to the Llama-3 chat! Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Ending chat with Llama-3. Goodbye!")
            break

        llama3_response = get_response(user_input)
        print("Llama-3: " + llama3_response)

if __name__ == "__main__":
    chat_with_llama3()
