
## Inspect the saved model architecture and vectorizer configuration
import tensorflow as tf

from translator import (
    model,
    english_vectorizer,
    french_vectorizer,
)


## 1. Display the complete model architecture
print("\n===== MODEL SUMMARY =====")
model.summary()


## 2. Inspect the English vectorizer configuration
print("\n===== ENGLISH VECTORIZER =====")
print("Output sequence length:",
      english_vectorizer.get_config().get("output_sequence_length"))

print("Standardization:",
      english_vectorizer.get_config().get("standardize"))

print("Vocabulary size:",
      english_vectorizer.vocabulary_size())


## 3. Inspect the French vectorizer configuration
print("\n===== FRENCH VECTORIZER =====")
print("Output sequence length:",
      french_vectorizer.get_config().get("output_sequence_length"))

print("Standardization:",
      french_vectorizer.get_config().get("standardize"))

print("Vocabulary size:",
      french_vectorizer.vocabulary_size())


## 4. Show how a sample French sentence is tokenized
sample = "startseq je veux une glace endseq"

print("\n===== SAMPLE FRENCH TOKENIZATION =====")
print("Input:", sample)
print("Token IDs:",
      french_vectorizer(tf.constant([sample])).numpy()[0].tolist())

print("Tokens:")
print(french_vectorizer.get_vocabulary()[:60])
