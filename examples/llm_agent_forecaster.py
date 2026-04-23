import os
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from sktime.datasets import load_airline
from sktime.forecasting.naive import NaiveForecaster

# 1. Define the sktime Tool
@tool
def forecast_airline_passengers(months_ahead: int) -> str:
    """Forecasts the number of airline passengers for a given number of months into the future."""
    y = load_airline()
    
    forecaster = NaiveForecaster(strategy="drift")
    forecaster.fit(y)
    
    fh = list(range(1, months_ahead + 1))
    predictions = forecaster.predict(fh)
    
    return f"Forecasted passengers for next {months_ahead} months:\n{predictions.to_string()}"

if __name__ == "__main__":
    print("Initializing sktime agentic tool-caller...\n")
    
    # 2. Setup LLM and bind the sktime tool directly to the model's brain
    # Requires GROQ_API_KEY environment variable to be set
    llm = ChatGroq(temperature=0, model_name="llama-3.1-8b-instant")
    llm_with_tools = llm.bind_tools([forecast_airline_passengers])
    
    # 3. Pass the query to the LLM
    query = "Can you forecast the airline passengers for the next 4 months using sktime?"
    print(f"User Query: {query}\n")
    
    print("Agent is reasoning...")
    ai_msg = llm_with_tools.invoke(query)
    
    # 4. Extract the tool call and execute it (The Agentic Loop)
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            print(f"-> LLM decided to use tool: {tool_call['name']} with arguments: {tool_call['args']}")
            
            if tool_call["name"] == "forecast_airline_passengers":
                # Execute the actual sktime Python code based on the LLM's requested arguments
                result = forecast_airline_passengers.invoke(tool_call["args"])
                print("\n================ FINAL OUTPUT ================\n")
                print(result)
                print("\n==============================================")
    else:
        print("LLM did not call the tool. Response:", ai_msg.content)