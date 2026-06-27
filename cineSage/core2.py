from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
load_dotenv()


from langchain_mistralai import ChatMistralAI

model=ChatMistralAI(model="mistral-small-2506")

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
  
parser = PydanticOutputParser(pydantic_object=Movie)

prompt = ChatPromptTemplate.from_messages([
  ("system","""
   Extract movie information from the paragraph
   {format_instructions}
   """),
  ("human","{paragraph}")
]
)


para = input("Enter the paragraph to analyze: ")
final_prompt = prompt.invoke(
   {"paragraph": para, "format_instructions": parser.get_format_instructions()}
)


response= model.invoke(final_prompt)
print(response.content)