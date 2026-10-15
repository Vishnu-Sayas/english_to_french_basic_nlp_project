
from pathlib import Path
import random
import shutil
import time
import re

import numpy as np
import tensorflow as tf


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "english_french_lstm.keras"
ENGLISH_VECTORIZER_PATH = MODEL_DIR / "english_vectorizer.keras"
FRENCH_VECTORIZER_PATH = MODEL_DIR / "french_vectorizer.keras"

# Training settings
MAX_SAMPLES = 50000
MAX_LEN = 20
MAX_TOKENS = 12000
EMBED_DIM = 128
LSTM_UNITS = 256
BATCH_SIZE = 128
EPOCHS = 20
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

print("=" * 60)
print("ENGLISH TO FRENCH LSTM TRAINING")
print("=" * 60)
print("Project folder:", BASE_DIR)
print("Models folder:", MODEL_DIR)


# ============================================================
# 2. FIND THE EXISTING DATASET
# ============================================================

possible_paths = [
    BASE_DIR / "fra.txt",
    BASE_DIR / "data" / "fra.txt",
    BASE_DIR / "dataset" / "fra.txt",
    BASE_DIR / "fra_eng_data" / "fra.txt",
    BASE_DIR / "fra_eng_data" / "fra.txt",
]

dataset_path = next(
    (path for path in possible_paths if path.is_file()),
    None
)

# If the file is in another subfolder, search the project.
if dataset_path is None:
    matches = list(BASE_DIR.glob("**/fra.txt"))

    # Ignore virtual environments and backup folders.
    matches = [
        path for path in matches
        if not any(
            part.lower() in {
                ".venv", "venv", "env",
                "node_modules", "__pycache__",
                "backup", "backups"
            }
            for part in path.parts
        )
        and MODEL_DIR not in path.parents
    ]

    if matches:
        dataset_path = matches[0]

if dataset_path is None:
    raise FileNotFoundError(
        "Could not find fra.txt.\n"
        "Place fra.txt in your project folder or inside a data folder."
    )

print("\nDataset found:", dataset_path)


# ============================================================
# 3. READ AND CLEAN THE SENTENCE PAIRS
# ============================================================

english_sentences = []
french_sentences = []

with open(dataset_path, "r", encoding="utf-8-sig") as file:
    for line in file:
        columns = line.rstrip("\r\n").split("\t")

        if len(columns) < 2:
            continue

        english = columns[0].strip().lower()
        french = columns[1].strip().lower()

        if not english or not french:
            continue

        # Skip sentences that are too long for our model.
        if len(english.split()) > MAX_LEN:
            continue

        if len(french.split()) > MAX_LEN - 2:
            continue

        # Avoid blank or malformed records.
        if "\ufffd" in english or "\ufffd" in french:
            continue

        english_sentences.append(english)
        french_sentences.append(
            "startseq " + french + " endseq"
        )

if not english_sentences:
    raise ValueError(
        "No valid sentence pairs were found in fra.txt."
    )

print("Valid sentence pairs found:", len(english_sentences))

# Check whether the text might have an encoding problem.
sample_french = " ".join(french_sentences[:100])

if re.search(r"Ã.|Â.|â€™|â€œ|â€", sample_french):
    print("\nWARNING: Possible French text encoding corruption detected.")
    print("Inspect fra.txt for text such as 'Ã§a' or 'Ã©tÃ©'.")
    print("Training will continue, but corrupted text can reduce quality.")

# IMPORTANT:
# Shuffle BEFORE taking a subset. Do not simply take the first
# 50,000 lines, because the dataset is ordered in groups.
pairs = list(zip(english_sentences, french_sentences))
random.shuffle(pairs)

if len(pairs) > MAX_SAMPLES:
    pairs = pairs[:MAX_SAMPLES]

english_sentences = [pair[0] for pair in pairs]
french_sentences = [pair[1] for pair in pairs]

print("Sentence pairs selected for training:", len(pairs))
print("\nExample training pairs:")

for i in range(min(5, len(pairs))):
    print("English:", english_sentences[i])
    print("French :", french_sentences[i])


# ============================================================
# 4. SPLIT INTO TRAINING, VALIDATION AND TEST DATA
# ============================================================

total = len(pairs)

train_end = int(total * 0.80)
val_end = int(total * 0.90)

english_train = english_sentences[:train_end]
french_train = french_sentences[:train_end]

english_val = english_sentences[train_end:val_end]
french_val = french_sentences[train_end:val_end]

english_test = english_sentences[val_end:]
french_test = french_sentences[val_end:]

print("\nTraining examples:", len(english_train))
print("Validation examples:", len(english_val))
print("Test examples:", len(english_test))


# ============================================================
# 5. CREATE AND ADAPT THE VECTORIZERS
# ============================================================

english_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LEN,
    standardize="lower_and_strip_punctuation",
    name="english_text_vectorizer"
)

french_vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_LEN,
    standardize="lower_and_strip_punctuation",
    name="french_text_vectorizer"
)

print("\nBuilding English vocabulary...")

english_vectorizer.adapt(
    tf.data.Dataset.from_tensor_slices(
        english_train
    ).batch(256)
)

print("Building French vocabulary...")

french_vectorizer.adapt(
    tf.data.Dataset.from_tensor_slices(
        french_train
    ).batch(256)
)

english_vocab = english_vectorizer.get_vocabulary()
french_vocab = french_vectorizer.get_vocabulary()

english_vocab_size = len(english_vocab)
french_vocab_size = len(french_vocab)

print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)

# Confirm that the special tokens survived vectorization.
if "startseq" not in french_vocab or "endseq" not in french_vocab:
    raise ValueError(
        "startseq or endseq is missing from the French vocabulary."
    )


# ============================================================
# 6. CONVERT SENTENCES INTO TOKEN IDS
# ============================================================

print("\nConverting sentences into token IDs...")

x_train = english_vectorizer(
    tf.constant(english_train)
).numpy()

x_val = english_vectorizer(
    tf.constant(english_val)
).numpy()

x_test = english_vectorizer(
    tf.constant(english_test)
).numpy()

target_train = french_vectorizer(
    tf.constant(french_train)
).numpy()

target_val = french_vectorizer(
    tf.constant(french_val)
).numpy()

target_test = french_vectorizer(
    tf.constant(french_test)
).numpy()

# Decoder input: startseq + French words
decoder_train = target_train[:, :-1]
decoder_val = target_val[:, :-1]

# Expected output: French words + endseq
y_train = target_train[:, 1:]
y_val = target_val[:, 1:]

# Ignore padding tokens in the loss.
sample_weights_train = (y_train != 0).astype("float32")
sample_weights_val = (y_val != 0).astype("float32")

print("Encoder input shape:", x_train.shape)
print("Decoder input shape:", decoder_train.shape)
print("Expected output shape:", y_train.shape)


# ============================================================
# 7. BUILD THE LSTM ENCODER-DECODER
# ============================================================

print("\nBuilding the LSTM model...")

# Encoder
encoder_inputs = tf.keras.Input(
    shape=(MAX_LEN,),
    dtype="int32",
    name="english_input"
)

encoder_embedding = tf.keras.layers.Embedding(
    input_dim=english_vocab_size,
    output_dim=EMBED_DIM,
    mask_zero=True,
    name="english_embedding"
)

encoder_lstm = tf.keras.layers.LSTM(
    LSTM_UNITS,
    return_state=True,
    name="encoder_lstm"
)

embedded_english = encoder_embedding(encoder_inputs)

_, encoder_h, encoder_c = encoder_lstm(embedded_english)

# Decoder
decoder_inputs = tf.keras.Input(
    shape=(MAX_LEN - 1,),
    dtype="int32",
    name="french_input"
)

decoder_embedding = tf.keras.layers.Embedding(
    input_dim=french_vocab_size,
    output_dim=EMBED_DIM,
    mask_zero=True,
    name="french_embedding"
)

decoder_lstm = tf.keras.layers.LSTM(
    LSTM_UNITS,
    return_sequences=True,
    return_state=True,
    name="decoder_lstm"
)

embedded_french = decoder_embedding(decoder_inputs)

decoder_outputs, _, _ = decoder_lstm(
    embedded_french,
    initial_state=[encoder_h, encoder_c]
)

output_layer = tf.keras.layers.Dense(
    french_vocab_size,
    activation="softmax",
    name="word_probabilities"
)

predictions = output_layer(decoder_outputs)

model = tf.keras.Model(
    inputs=[encoder_inputs, decoder_inputs],
    outputs=predictions,
    name="english_french_lstm"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    weighted_metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(
            name="token_accuracy"
        )
    ]
)

model.summary()


# ============================================================
# 8. TRAIN THE MODEL
# ============================================================

print("\nStarting local training...")
print("This may take some time, depending on your computer.")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
]

start_time = time.time()

history = model.fit(
    [x_train, decoder_train],
    y_train,
    sample_weight=sample_weights_train,
    validation_data=(
        [x_val, decoder_val],
        y_val,
        sample_weights_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    callbacks=callbacks,
    verbose=1
)

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time / 60:.2f} minutes.")


# ============================================================
# 9. EVALUATE THE MODEL
# ============================================================

print("\nEvaluating on held-out test data...")

decoder_test = target_test[:, :-1]
y_test = target_test[:, 1:]
sample_weights_test = (y_test != 0).astype("float32")

test_results = model.evaluate(
    [x_test, decoder_test],
    y_test,
    sample_weight=sample_weights_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

print("\nTest loss:", test_results[0])

if len(test_results) > 1:
    print("Test token accuracy:", test_results[1])


# ============================================================
# 10. PREPARE SAVED VECTORIZER MODELS
# ============================================================

english_vectorizer_model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(), dtype=tf.string),
        english_vectorizer
    ],
    name="english_vectorizer_model"
)

french_vectorizer_model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(), dtype=tf.string),
        french_vectorizer
    ],
    name="french_vectorizer_model"
)


# ============================================================
# 11. SAVE TEMPORARY ARTIFACTS FIRST
# ============================================================

# Do not replace your existing model until all new files save.
TEMP_MODEL_PATH = MODEL_DIR / "english_french_lstm.new.keras"
TEMP_ENGLISH_PATH = MODEL_DIR / "english_vectorizer.new.keras"
TEMP_FRENCH_PATH = MODEL_DIR / "french_vectorizer.new.keras"

print("\nSaving the new model and vectorizers...")

model.save(TEMP_MODEL_PATH)
english_vectorizer_model.save(TEMP_ENGLISH_PATH)
french_vectorizer_model.save(TEMP_FRENCH_PATH)

# Verify that all three new files exist and are non-empty.
for path in [
    TEMP_MODEL_PATH,
    TEMP_ENGLISH_PATH,
    TEMP_FRENCH_PATH
]:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Saving failed; missing or empty file: {path}"
        )


# ============================================================
# 12. BACK UP EXISTING FILES, THEN REPLACE THEM
# ============================================================

backup_dir = MODEL_DIR / "backup_before_retraining"
backup_dir.mkdir(exist_ok=True)

old_files = [
    MODEL_PATH,
    ENGLISH_VECTORIZER_PATH,
    FRENCH_VECTORIZER_PATH
]

for path in old_files:
    if path.exists():
        backup_path = backup_dir / path.name
        shutil.copy2(path, backup_path)
        print("Backed up:", backup_path)

# Replace the old artifacts only after all temporary saves succeeded.
TEMP_MODEL_PATH.replace(MODEL_PATH)
TEMP_ENGLISH_PATH.replace(ENGLISH_VECTORIZER_PATH)
TEMP_FRENCH_PATH.replace(FRENCH_VECTORIZER_PATH)

print("\n" + "=" * 60)
print("TRAINING AND SAVING COMPLETED")
print("=" * 60)

print("Model:", MODEL_PATH)
print("English vectorizer:", ENGLISH_VECTORIZER_PATH)
print("French vectorizer:", FRENCH_VECTORIZER_PATH)
print("Backups:", backup_dir)

print("\nThese three files now belong to the same trained model.")
print("You can now test your existing translator.py.")
