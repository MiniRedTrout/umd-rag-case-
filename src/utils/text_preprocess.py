import tiktoken
from nltk.corpus import stopwords

encoder = tiktoken.encoding_for_model("gpt-4")
russian_stopwords = stopwords.words("russian")


def preprocess_func(text: str) -> list[str]:
    """Tokenize the string for BM25

    Args:
        text: chunk text

    Returns:
        List of tokens
    """
    # Lowercase the input text
    lowered = text.lower()

    # Filter stop words
    words = lowered.split()
    filtered_words = [word for word in words if word.lower() not in russian_stopwords]
    text = " ".join(filtered_words)

    # Convert the lowered text into tokens
    tokens = encoder.encode(text)

    # Stringify the tokens
    return [str(token) for token in tokens]
