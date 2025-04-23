#create custom chatbot in phi data platform
import openai
from phi.agent import Agent
import phi.api
from phi.model.openai import OpenAIChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv 
from phi.model.groq import Groq

import os
import phi
from phi.playground import Playground, serve_playground_app
load_dotenv()

phi.api=os.getenv("PHI_API_KEY")

##Web search Agent
web_search_agent=Agent(
     name="Web Search Agent",
     role="Search the web for the information",
     model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
     tools=[DuckDuckGo()],
     instructions=["Always include sources"],
     show_tools_calls=True,
     markdown=True 
)

##Financial Agent
finance_agent=Agent(
     name="Finance AI Agent",
     model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
     tools=[
         YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                       company_news=True)
         ],
     instructions=["Use tables to display the data"],
     show_tools_calls=True,
     markdown=True 
)

app=Playground(agents=[finance_agent,web_search_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app",reload=True)


