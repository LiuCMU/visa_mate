import asyncio
from google.adk.agents import Agent, LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import google_search, AgentTool, FunctionTool
from google.genai import types
from google.adk.runners import InMemoryRunner
print("✅ ADK components imported successfully.")

retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)


planning_agent = LlmAgent(
    name="planning_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="An agent for planning visa types and requirements based on user citizenship, source country, destination, and travel purpose.",
    # Is the sub-agent able to ask root_agent for more info if needed?
    instruction="""
    You are a legal expert specialized in visa regulations. 
    Given the user's citizenship, source country, destination, and travel purpose,
    identify the suitable visa type and list the requirements.
    If anything is unclear about the user, check the memory or ask the user for more details.
    If anything is unclear about visa regulations, use the google search tool to find more information.
    """,
    tools=[google_search],
    output_key="suitable_visa_info_and_requirements",
)

root_agent = LlmAgent(
    name="visa_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="An assistant for answering visa-related questions.",
    instruction="""
    You are a helpful assistant specialized in visa-related questions. 
    Provide accurate and concise answers to user inquiries.

    You should always follow the steps below:
    1. Based on the user's input or memory, understand the destination,
    travel purpose, source country and citizenship. If you don't have enough information, ask
    the user for more details.
    2. Use the planning sub-agent to identify the suitable visa type and requirements.
    3. Based on the results from the planning sub-agent, provide an overview of
    the pros and cons for each visa option, then provide a general choice
    guidance for the user.
    """,
    tools=[AgentTool(planning_agent)]
)


if __name__ == "__main__":
    runner = InMemoryRunner(agent=root_agent)

    async def main(query: str):
        response = await runner.run_debug(query)

    query = "I want to go to the USA"
    asyncio.run(main(query))