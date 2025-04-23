from phi.agent import Agent
from phi.model.groq import Groq   #model
from phi.tools.yfinance import YFinanceTools # tool that enables Agent to access stock data, financial info
from phi.tools.duckduckgo import DuckDuckGo #tool that enables Agent to search the web
import openai
import os
from dotenv import load_dotenv 

load_dotenv()

openai.api_key=os.getenv("OPENAI_API_KEY")


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

multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    instructions=["Always include sources","Use tables to display the data"],
    show_tools_calls=True,
    markdown=True 
)

multi_ai_agent.print_response("Summarise analyst recommendation and share the latest news for NVDA",stream=True)
