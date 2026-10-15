
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Multilingual model: understands English and French
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)


def compare_sentences(sentence1: str, sentence2: str) -> float:
    """Return semantic similarity between two sentences as a percentage."""
    if not sentence1.strip() or not sentence2.strip():
        raise ValueError("Both sentences must contain text.")

    embeddings = model.encode([sentence1, sentence2])

    score = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    # Convert similarity score to a percentage for easy display
    return round(float(score) * 100, 2)


if __name__ == "__main__":
    english = input("Enter the English sentence: ")
    french = input("Enter the French sentence: ")

    try:
        similarity = compare_sentences(english, french)
        print(f"\nCosine similarity: {similarity}%")
        print("This score measures semantic similarity, not translation accuracy.")
    except ValueError as error:
        print(error)
