from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()



from langchain_mistralai import ChatMistralAI

model=ChatMistralAI(model="mistral-small-2506")

prompt = ChatPromptTemplate.from_messages([
  ("system",
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
"""),
  ("human",
   """
   Analyze the following paragraph and extract useful information.

Paragraph:
{paragraph}
   """)
]
)

para = input("Enter the paragraph to analyze: ")

final_prompt = prompt.invoke(
   {"paragraph": para}
)

response= model.invoke(final_prompt)
print(response.content)