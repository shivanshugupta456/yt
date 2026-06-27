import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(
    page_title="Paragraph Analyzer",
    page_icon="📄",
    layout="centered"
)

# Title
st.title("📄 Paragraph Information Extractor")

# Initialize Model
model = ChatMistralAI(model="mistral-small-2506")

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an advanced information extraction assistant.

Your job is to carefully read the provided paragraph and extract the most useful and meaningful information from it.

Extract details such as:
- Movie name
- Genre
- Director
- Main characters
- Cast (if mentioned)
- Themes
- Important concepts
- Setting
- Story overview
- Emotional tone
- Key topics
- Important highlights
- Overall message

Also generate:
1. A short plot summary
2. A quick summary (2-3 lines only)

Instructions:
- Keep the response clean and well-structured.
- Use headings and bullet points.
- Do not invent information that is not present.
- If some information is unavailable, simply mention "Not mentioned".
- Make the output readable and professional.
- Focus only on useful information.
"""
    ),
    (
        "human",
        """
Analyze the following paragraph and extract useful information.

Paragraph:
{paragraph}
"""
    )
])

# User Input
paragraph = st.text_area(
    "Enter the paragraph to analyze:",
    height=250
)

# Analyze Button
if st.button("Analyze Paragraph"):

    if paragraph.strip() == "":
        st.warning("Please enter a paragraph.")
    else:
        with st.spinner("Analyzing..."):

            final_prompt = prompt.invoke(
                {"paragraph": paragraph}
            )

            response = model.invoke(final_prompt)

            st.subheader("Analysis Result")
            st.write(response.content)