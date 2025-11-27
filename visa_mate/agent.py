import asyncio
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search
from google.genai import types
from google.adk.runners import InMemoryRunner
print("✅ ADK components imported successfully.")

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)

root_agent = Agent(
    name="immigrant_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="An assistant for answering visa-related questions.",
    instruction="""
    You are a helpful assistant specialized in visa-related questions. 
    Provide accurate and concise answers to user inquiries.

    You should alwasy follow the steps below:
    1. Based on the user's input or memory, understand the destination,
    travel purpose, source conntry and citizenship. If you don't have enough information, ask
    the user for more details.
    2. Use the planning sub-agent to identify the suitable visa type and requirements.
    3. Tell the user the identified visa type and requirements in a concise manner.
    """,
    tools=[google_search]
)


if __name__ == "__main__":
    runner = InMemoryRunner(agent=root_agent)

    async def main(query: str):
        response = await runner.run_debug(query)

    query = "I will go to the USA"
    asyncio.run(main(query))