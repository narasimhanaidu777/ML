responses = {
    "hello": "Hi!",
    "how are you": "I am fine.",
    "what is ai": "AI means Artificial Intelligence.",
    "bye": "Goodbye!"
}

while True:
    user_input = input("You: ").lower()

    if user_input == "bye":
        print("Bot:", responses["bye"])
        break

    response = responses.get(user_input, "Sorry, I don't understand.")

    print("Bot:", response)