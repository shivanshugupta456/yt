from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

model=ChatMistralAI(model="mistral-small-2506")

print("choose your AI mode")
print("press 1 for angry mode")
print("press 2 for funny mode")
print("press 3 for sad mode")

mode = input("Enter your choice: ")

if mode == "1":
    messages = [
        SystemMessage(content="You are an angry ai agent very angry")
    ]
elif mode == "2":
    messages = [
        SystemMessage(content="You are a funny ai agent very funny ")
    ]
elif mode == "3":
    messages = [
        SystemMessage(content="You are a sad ai agent very sad")
    ]
else:
    print("Invalid choice. Exiting.")
    exit()
messages = [
  SystemMessage(content=mode)
]

print("------------welcome type 0 to exit------------")
while True:
  
  prompt = input("Enter your prompt: ")
  messages.append(HumanMessage(content=prompt))
  if prompt == "0":
    break
  response= model.invoke(messages)
  messages.append(AIMessage(content=response.content))
  print("Bot :",response.content)
  
print(messages)