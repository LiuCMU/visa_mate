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
    description="An assistant for answering immigration-related questions.",
    instruction="""
    You are a helpful assistant specialized in immigration-related questions.
    Provide accurate and concise answers to user inquiries.
    """,
    tools=[google_search]
)

runner = InMemoryRunner(agent=root_agent)

async def main():
    response = await runner.run_debug("What are the options for immigrating to UK?")
    print("✅ Agent executed successfully.")
    print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())
