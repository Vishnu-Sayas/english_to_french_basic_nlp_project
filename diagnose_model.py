
## Import the trained model and translation components
import tensorflow as tf

from translator import (
    model,
    english_vectorizer,
    french_vectorizer,
    encoder_model,
    decoder_model,
    START_ID,
    END_ID,
    id_to_token,
)


## 1. Display vocabulary information
print("\n--- VOCABULARY CHECK ---")

for word in ["ice", "cream", "job", "want"]:
    word_id = english_vectorizer(
        tf.constant([word])
    ).numpy()[0]

    print(f"{word!r} token IDs: {word_id.tolist()}")


## 2. Inspect the start and end tokens
print("\n--- FRENCH TOKENS ---")
print("Start token ID:", START_ID)
print("Start token:", id_to_token.get(START_ID))
print("End token ID:", END_ID)
print("End token:", id_to_token.get(END_ID))


## 3. Show the model's greedy predictions
def inspect_prediction(sentence):
    print(f"\nEnglish input: {sentence}")

    ## Convert English text into token IDs
    english_ids = english_vectorizer(
        tf.constant([sentence])
    )

    ## Encode the sentence
    h, c = encoder_model(
        tf.cast(english_ids, tf.int32),
        training=False
    )

    ## Begin with the French start token
    current_token = tf.constant(
        [[START_ID]],
        dtype=tf.int32
    )

    words = []

    ## Generate up to 19 tokens
    for step in range(19):
        probabilities, h, c = decoder_model(
            [current_token, h, c],
            training=False
        )

        ## Get the top five predictions at this step
        top_values, top_ids = tf.math.top_k(
            probabilities[0, -1, :],
            k=5
        )

        if step == 0:
            print("First-step top predictions:")

            for token_id, score in zip(
                top_ids.numpy(),
                top_values.numpy()
            ):
                print(
                    f"  {id_to_token.get(int(token_id), '?')}: "
                    f"{float(score):.4f}"
                )

        ## Choose the most likely token
        next_id = int(
            tf.argmax(
                probabilities[0, -1, :]
            ).numpy()
        )

        if next_id == END_ID:
            break

        if next_id in (0, 1):
            print("Stopped at padding or unknown token.")
            break

        word = id_to_token.get(next_id, "")
        if not word:
            break

        words.append(word)

        current_token = tf.constant(
            [[next_id]],
            dtype=tf.int32
        )

    print("Generated French:", " ".join(words))


## 4. Compare different inputs
inspect_prediction("I want an ice cream")
inspect_prediction("I want a job")
inspect_prediction("How are you?")
