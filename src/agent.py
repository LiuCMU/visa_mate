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


news_agent = LlmAgent(
    name="news_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="An agent for providing the latest updates on visa policies and regulations.",
    instruction="""
    You are an expert in tracking visa policy changes worldwide. Use the google search tool to find the latest news and updates on visa policies and regulations for various countries based on user queries.
    """,
    # Does the news_agent respond to request from user or from the root_agent?
    tools=[google_search],
)

form_agent = LlmAgent(
    name="form_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="An agent for assisting with visa application forms.",
    instruction="""
    You are an expert in visa application processes. Given a specific visa type and its requirements, provide detailed guidance on filling out the necessary application forms. For each form, list the fields that need to be filled and the information required for each field. If you are unsure about any field, use the google search tool to find more information.
    """,
    output_key="visa_application_form_guidance",
    tools=[google_search],
)

prepare_agent = LlmAgent(
    name="prepare_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="An agent for preparing visa application documents based on visa type and requirements.",
    instruction="""
    You are an expert in visa application processes. Based on the identified visa type and its requirements, provide a checklist of necessary documents and steps the user needs to take to prepare for their visa application.
    For each required document, tell the user to bring or prepare it.
    For each form that needs to be filled out, print out what information is needed fo fill in each field. If any information is missing, ask the user to provide it.
    For any third-party documents that need to be obtained (e.g., bank statements, invitation letters), provide guidance on how to acquire them.
    For any fees that need to be paid, provide the payment methods and instructions.
    As the user completes each step, update the checklist to reflect their progress.
    If you are unsure about a fillable form, ask the form_agent to help you about what fields need to be filled and what information is needed.
    """,
    output_key="visa_application_preparation",
    tools=[AgentTool(form_agent)]
)

root_agent = LlmAgent(
    name="visa_assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="An assistant for visa-related things",
    instruction="""
    You are a helpful assistant specialized in visa-related questions and preparations. 
    Provide accurate and concise answers to user inquiries.

    You should always follow the steps below:
    1. Based on the user's input or memory, understand the destination,
    travel purpose, source country and citizenship. If you don't have enough information, ask
    the user for more details.
    2. Use the planning sub-agent to identify the suitable visa type and requirements.
    3. Based on the results from the planning sub-agent, provide an overview of visa options,
    then provide a general choice guidance for the user.

    If the user inquires anything you are unsure about regarding visa policies or regulations,
    use the sub-agent news_agent to get the latest updates before responding.

    Once the user confirms their visa choice, use the prepare_agent sub-agent
    to help the user prepare for their visa application process.
    """,
    
    tools=[AgentTool(planning_agent), AgentTool(news_agent), AgentTool(prepare_agent)]
)


if __name__ == "__main__":
    runner = InMemoryRunner(agent=root_agent)

    async def main(query: str):
        response = await runner.run_debug(query)

    query = "I want to go to the USA"
    asyncio.run(main(query))