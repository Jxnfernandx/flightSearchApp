import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class GPT3Chatbot:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate_response(self, input_text, max_length=100):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=max_length, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

def main():
    chatbot = GPT3Chatbot()
    print("Welcome to the chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            print("Chatbot: Goodbye!")
            break
        response = chatbot.generate_response(user_input)
        print("Chatbot:", response)

if __name__ == "__main__":
    main()
