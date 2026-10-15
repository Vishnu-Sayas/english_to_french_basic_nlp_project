
from functools import lru_cache

import torch
from transformers import MarianTokenizer, MarianMTModel


MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


@lru_cache(maxsize=1)
def load_translation_model():
    print(f"Loading {MODEL_NAME}...")
    print(f"Using device: {DEVICE}")

    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return tokenizer, model


def translate_sentence(sentence: str) -> str:
    if not isinstance(sentence, str):
        return "Please enter a valid English sentence."

    sentence = sentence.strip()

    if not sentence:
        return ""

    try:
        tokenizer, model = load_translation_model()

        inputs = tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=128
            )

        translation = tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        return translation.strip()

    except Exception as error:
        print("Translation error:", repr(error))
        return "Translation failed. Check the VS Code terminal."


if __name__ == "__main__":
    test_sentences = [
        "Hello",
        "How are you?",
        "I want an ice cream.",
        "I want a job.",
        "Thank you."
    ]

    for sentence in test_sentences:
        print(f"\nEnglish: {sentence}")
        print(f"French:  {translate_sentence(sentence)}")
