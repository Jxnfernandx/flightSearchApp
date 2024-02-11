import nltk
from nltk.chat.util import Chat, reflections

# Define some patterns and responses
patterns = [
    (r'hi|hello|hey|whats up| sup', ['Hello!', 'Hey there!', 'Hi!']),
    (r'how are you?| doing well?', ['I am good, thank you!', 'I am doing well, thank you for asking.']),
    (r'what is your name?', ['You can call me Chatbot.', 'My name is Chatbot.']),
    (r'(.*) your name?', ["I'm just a chatbot, I don't have a name."]),
    (r'quit', ['Bye, take care!', 'Goodbye!']),
    (r'ticket price|cheapest tickets| most affordable flights', ['Prices range from 6 cheapest categories. The price ranges are $25-$100, $101-$500, $501-$1000, $1001-$5000, $5001-$10000, $1001+'])
]

# Create a chatbot
chatbot = Chat(patterns, reflections)

def main():
    print("Welcome to the chatbot. Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        response = chatbot.respond(user_input)
        print("Chatbot:", response)
        if user_input.lower() == 'quit':
            break

if __name__ == "__main__":
    main()
