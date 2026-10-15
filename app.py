
## 1. Import the required libraries
import streamlit as st
import time

## Import the translation function from your own Python file
from translator import translate_sentence


## 2. Configure the browser page
st.set_page_config(
    page_title="English to French Translator",
    page_icon="🌍",
    layout="wide"
)


## 3. Display the application heading
st.title("🌍 AI Language Translator")
st.subheader("English 🇬🇧 → French 🇫🇷")

st.write(
    "Translate English sentences into French "
    "using a trained LSTM Encoder-Decoder model."
)

st.divider()


## 4. Display project information in the sidebar
with st.sidebar:
    st.title("About the Project")

    st.write("**Task:** Neural Machine Translation")
    st.write("**Model:** LSTM Encoder-Decoder")
    st.write("**Input Language:** English")
    st.write("**Output Language:** French")

    st.divider()
    st.info("Enter an English sentence to get its French translation.")


## 5. Create two columns for input and output
english_column, french_column = st.columns(2)

with english_column:
    st.subheader("English Input")

    ## User types their sentence in this text box
    english_text = st.text_area(
        "Enter an English sentence",
        placeholder="Type your sentence here...",
        height=180,
        max_chars=500
    )


## 6. Create the Translate button
st.divider()

if st.button("✨ Translate to French", type="primary"):

    ## Remove unnecessary spaces from the beginning and end
    cleaned_text = english_text.strip()

    ## Check whether the user entered any text
    if not cleaned_text:
        st.warning("Please enter an English sentence first.")

    else:
        try:
            ## Display a loading message while translating
            with st.spinner("Generating French translation..."):

                ## Record the starting time
                start_time = time.perf_counter()

                ## Call the trained model's translation function
                french_translation = translate_sentence(cleaned_text)

                ## Calculate the total translation time
                elapsed_time = time.perf_counter() - start_time

            ## Check whether the model returned a translation
            if french_translation and french_translation.strip():

                st.success("Translation completed successfully!")

                ## Display the generated French sentence
                st.subheader("🇫🇷 Generated French Translation")
                st.write(french_translation)

                ## Display translation performance information
                metric1, metric2 = st.columns(2)

                with metric1:
                    st.metric(
                        label="Translation Time",
                        value=f"{elapsed_time:.3f} seconds"
                    )

                with metric2:
                    st.metric(
                        label="Input Length",
                        value=f"{len(cleaned_text.split())} words"
                    )

            else:
                st.warning(
                    "No translation was generated. "
                    "Please try another sentence."
                )

        ## Handle errors without displaying a full traceback
        except Exception as error:
            st.error(
                "Translation failed. Check your model files "
                "and translator.py configuration."
            )

            ## Allow the user to expand and inspect the error
            with st.expander("View technical error"):
                st.code(str(error))


## 7. Display the footer
st.divider()

st.caption(
    "English-to-French Neural Machine Translation | "
    "Python • TensorFlow • Streamlit"
)
