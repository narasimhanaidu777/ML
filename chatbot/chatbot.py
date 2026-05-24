import streamlit as st
import pandas as pd
import nltk
import re
import random

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    st.error("Please install scikit-learn using: pip install scikit-learn")

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Define intents with patterns and responses
INTENTS = {
    "greeting": {
        "patterns": [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "what's up", "howdy", "greetings"
        ],
        "responses": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! Nice to see you. How can I assist?",
            "Greetings! How may I help you?"
        ]
    },

    "farewell": {
        "patterns": [
            "bye", "goodbye", "see you", "take care",
            "farewell", "see you later"
        ],
        "responses": [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Bye! Happy to help anytime!"
        ]
    },

    "thanks": {
        "patterns": [
            "thank you", "thanks", "appreciate it"
        ],
        "responses": [
            "You're welcome!",
            "Happy to help!",
            "Anytime!"
        ]
    },

    "pricing": {
        "patterns": [
            "pricing", "cost", "price", "how much",
            "subscription", "plan"
        ],
        "responses": [
            "We offer:\n• Basic Plan: Free\n• Pro Plan: $29/month\n• Enterprise: Custom pricing",
            "Our pricing starts at $29/month."
        ]
    },

    "hours": {
        "patterns": [
            "hours", "support hours", "open",
            "business hours"
        ],
        "responses": [
            "Support is available Monday-Friday, 9 AM to 6 PM.",
            "We provide 24/7 chat support."
        ]
    },

    "contact": {
        "patterns": [
            "contact", "email", "phone", "support"
        ],
        "responses": [
            "Contact us at:\n📧 narasimhanaidumeda@gmail.com\n📞 +91 8317505944"
        ]
    },

    "features": {
        "patterns": [
            "features", "capabilities", "what can you do"
        ],
        "responses": [
            "I can help with:\n• Pricing info\n• Features\n• Contact support\n• Business hours"
        ]
    }
}


class RuleBasedChatbot:

    def __init__(self):
        self.intents = INTENTS
        self.corpus = []
        self.intent_labels = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.build_corpus()

    def preprocess_text(self, text):

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Lemmatize
        lemmatized_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
        ]

        # Stop words
        stop_words = {
            'i', 'me', 'my', 'you', 'your',
            'he', 'she', 'it', 'they',
            'a', 'an', 'and', 'are', 'is',
            'of', 'on', 'to', 'the', 'in'
        }

        filtered_tokens = [
            token for token in lemmatized_tokens
            if token not in stop_words
        ]

        return ' '.join(filtered_tokens)

    def build_corpus(self):

        for intent_name, intent_data in self.intents.items():

            for pattern in intent_data['patterns']:

                processed_pattern = self.preprocess_text(pattern)

                self.corpus.append(processed_pattern)

                self.intent_labels.append(intent_name)

        # Train TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

    def classify_intent(self, user_input):

        processed_input = self.preprocess_text(user_input)

        if not processed_input:
            return "fallback"

        # Convert user input to vector
        user_vector = self.vectorizer.transform([processed_input])

        # Calculate similarity
        similarities = cosine_similarity(
            user_vector,
            self.tfidf_matrix
        ).flatten()

        max_similarity = max(similarities)

        threshold = 0.2

        if max_similarity >= threshold:

            best_match_idx = similarities.argmax()

            return self.intent_labels[best_match_idx]

        return "fallback"

    def get_response(self, intent):

        if intent == "fallback":

            fallback_responses = [
                "Sorry, I didn't understand that.",
                "Can you please rephrase?",
                "I'm not trained for that question."
            ]

            return random.choice(fallback_responses)

        responses = self.intents[intent]["responses"]

        return random.choice(responses)


# Streamlit UI
def main():

    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="wide"
    )

    st.title("🤖 AI Rule-Based Chatbot")

    st.markdown("""
    Using NLTK preprocessing and TF-IDF with cosine similarity ; created by "Narasimha Naidu"
    """)

    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RuleBasedChatbot()

    # Chat history
    if "messages" not in st.session_state:

        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! Ask me something."
            }
        ]

    # Display messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_input = st.chat_input("Type your message...")

    if user_input:

        # Show user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        # Get response
        intent = st.session_state.chatbot.classify_intent(user_input)

        response = st.session_state.chatbot.get_response(intent)

        # Store response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )

        st.rerun()

    # Sidebar
    with st.sidebar:

        st.header("About")

        st.write("""
        This chatbot uses:
        - NLTK
        - TF-IDF Vectorization
        - Cosine Similarity
        - Rule-Based Intent Matching
        """)

        if st.button("Clear Chat"):

            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat cleared!"
                }
            ]

            st.rerun()


if __name__ == "__main__":
    main()