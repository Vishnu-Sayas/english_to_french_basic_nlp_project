
## 1. Import required libraries
from pathlib import Path
import tensorflow as tf

## 2. Locate your existing model files
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

model_path = MODEL_DIR / "english_french_lstm.keras"
english_path = MODEL_DIR / "english_vectorizer.keras"
french_path = MODEL_DIR / "french_vectorizer.keras"

## 3. Check which files exist
print("\n========== FILE CHECK ==========")

for path in [model_path, english_path, french_path]:
    print(f"{path.name}: {'FOUND' if path.exists() else 'MISSING'}")

## 4. Load and inspect the trained model
print("\n========== MODEL INSPECTION ==========")

if model_path.exists():
    model = tf.keras.models.load_model(model_path, compile=False)

    model.summary()

    print("\nModel input shapes:")
    for tensor in model.inputs:
        print(f"Name: {tensor.name}, Shape: {tensor.shape}")

    print("\nModel output shape:")
    print(model.output_shape)
else:
    print("Model file is missing.")

## 5. Inspect each saved vectorizer
def inspect_vectorizer(path, label):
    print(f"\n========== {label} VECTORIZER ==========")

    if not path.exists():
        print(f"File missing: {path}")
        return

    try:
        obj = tf.keras.models.load_model(path, compile=False)
        print("Loaded as a Keras model.")
        obj.summary()

        for layer in obj.layers:
            if isinstance(layer, tf.keras.layers.TextVectorization):
                print("Vocabulary examples:")
                print(layer.get_vocabulary()[:40])

    except Exception as error:
        print("Could not load as a Keras model:")
        print(type(error).__name__, str(error))

        print(
            "This file may have been saved as a standalone "
            "TextVectorization layer rather than a model."
        )

inspect_vectorizer(english_path, "ENGLISH")
inspect_vectorizer(french_path, "FRENCH")

print("\n========== INSPECTION FINISHED ==========")
