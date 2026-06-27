from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI
import streamlit as st
import json

# Load environment variables
load_dotenv()

# Initialize model
model = ChatMistralAI(model="mistral-small-2506")

# Define schema
class Movie(BaseModel):
    name: str
    genre: str
    director: str
    main_characters: List[str]
    cast: Optional[List[str]]
    themes: List[str]
    important_concepts: List[str]
    setting: str
    story_overview: str
    emotional_tone: str
    key_topics: List[str]
    important_highlights: List[str]

# Output parser
parser = PydanticOutputParser(pydantic_object=Movie)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Extract movie information from the paragraph.

        {format_instructions}
        """
    ),
    ("human", "{paragraph}")
])

# ---------------- STREAMLIT UI ---------------- #

st.set_page_config(page_title="Movie Information Extractor", page_icon="🎬")

st.title("🎬 Movie Information Extractor")
st.write("Enter a movie paragraph and extract structured information using AI.")

# Text input
para = st.text_area(
    "Enter movie paragraph:",
    height=250,
    placeholder="Paste movie description here..."
)

# Button
if st.button("Extract Information"):

    if para.strip() == "":
        st.warning("Please enter a paragraph.")
    else:
        with st.spinner("Analyzing movie information..."):

            # Create final prompt
            final_prompt = prompt.invoke({
                "paragraph": para,
                "format_instructions": parser.get_format_instructions()
            })

            # Get response
            response = model.invoke(final_prompt)

            try:
                # Parse structured output
                parsed_output = parser.parse(response.content)

                st.success("Extraction Completed ✅")

                # Show JSON output
                st.subheader("Structured Movie Information")
                st.json(parsed_output.dict())

                # Optional formatted display
                st.subheader("Formatted View")

                st.markdown(f"### 🎥 {parsed_output.name}")
                st.write(f"**Genre:** {parsed_output.genre}")
                st.write(f"**Director:** {parsed_output.director}")
                st.write(f"**Setting:** {parsed_output.setting}")
                st.write(f"**Emotional Tone:** {parsed_output.emotional_tone}")

                st.write("### Main Characters")
                st.write(", ".join(parsed_output.main_characters))

                if parsed_output.cast:
                    st.write("### Cast")
                    st.write(", ".join(parsed_output.cast))

                st.write("### Themes")
                st.write(", ".join(parsed_output.themes))

                st.write("### Important Concepts")
                st.write(", ".join(parsed_output.important_concepts))

                st.write("### Story Overview")
                st.write(parsed_output.story_overview)

                st.write("### Key Topics")
                st.write(", ".join(parsed_output.key_topics))

                st.write("### Important Highlights")
                for item in parsed_output.important_highlights:
                    st.write(f"• {item}")

            except Exception as e:
                st.error("Error parsing response")
                st.code(response.content)