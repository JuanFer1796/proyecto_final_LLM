from typing import Any, Dict
from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
import streamlit as st
import datetime
import os


def save_history(question, answer):
    with open("history.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}: {question}->{answer}\n")


def load_history():
    if os.path.exists("history.txt"):
        with open("history.txt", "r") as f:
            return f.readlines()
    return []


def main():
    st.set_page_config(page_title="Multi-Agent Financial & Sustainability Analyzer",
                       page_icon="👾",
                       layout="wide")

    st.title("👾 Multi-Agent Financial & Sustainability Analyzer")

    # Custom CSS
    st.markdown(
        """
        <style>
        .stApp {
            background-color: black;
        }
        .title {
            color: #ff4b4b;
        }
        .stButton>button {
            background-color: #ff4b4b;
            color: white;
            border-radius: 5px;
        }
        .stTextInput>div>div>input {
            border: 1px solid #ff4b4b;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Cargar variables de entorno
    load_dotenv()

    # Instrucciones iniciales para el agente

    instructions = """
    You are an agent designed to analyze data and write Python code. You have access to several tools:

    1. Python REPL for executing code
    2. Crypto Analysis for cryptocurrency data
    3. Stock Analysis for stock market data
    4. Sustainability Analysis for corporate sustainability metrics
      CRITICAL INFORMATION ABOUT CLIMATE GRADES it's important to take into account the following information for Sustainability Analysis:
            - Order from BEST to WORST is: 
              A+ (best) > A > B+ > B > C+ > C > C- > D+ > D > F (worst)
            - Any grade with '+' is better than the base grade
            - Any grade with '-' is worse than the base grade
    5. Combined DJIA Analysis for news and market data about DJIA or  Dow Jones Industrial Average
        In this case label functions as an indicator 1 indicates positive influence and 0 bad influence. Just show the top 1 when asked.

    For data analysis queries:
    - Use the appropriate tool based on the data being queried
    - Always provide a clear, structured response
    - If analyzing data, explain your findings
    - If you need to execute code, use the Python REPL

    For Python coding tasks:
    - Use the Python REPL tool
    - Write clear, executable code
    - Handle errors appropriately
    - Return the results of the execution

    If you encounter an error:
    1. Debug the code
    2. Try an alternative approach
    3. If still unsuccessful, explain what went wrong

    Remember to:
    - Always use a tool to answer questions
    - Provide clear explanations of your actions
    - Format outputs in a readable way
    """



    # Crear el prompt base
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)

    # Crear herramientas básicas (Python REPL Tool)
    tools = [PythonREPLTool()]

    # Crear agente Python básico
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        tools=tools,

    )
    python_agent_executor = AgentExecutor(agent=python_agent, tools=tools, verbose=True)

    # Crear agentes CSV
    crypto_agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4"),
        "1000_cryptos_top_100_last_week.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    stock_agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4"),
        "World_Stock_Prices_last_week.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    sustainability_agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4"),
        "most sustainable corporations.csv",
        verbose=True,
        allow_dangerous_code=True,
        pandas_kwargs={'encoding': 'latin-1'}
    )

    combined_agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4"),
        "Combined_News_DJIA_last_six_months_fixed.csv",
        verbose=True,
        allow_dangerous_code=True
    )

    # Wrapper para el Python Agent
    def python_agent_executor_wrapper(original_prompt: str) -> Dict[str, Any]:
        return python_agent_executor.invoke({"input": original_prompt})

    # Extender herramientas con el Python Agent y CSV Agents
    tools.extend([
        Tool(
            name="Python Agent",
            func=python_agent_executor_wrapper,
            description="""Useful when you need to transform natural language to Python 
            and execute the Python code, returning the results of the code execution. 
            DOES NOT ACCEPT CODE AS INPUT""",
        ),
        Tool(
            name="Crypto Analysis",
            func=crypto_agent.run,
            description="""Useful for analyzing cryptocurrency price data from 1000_cryptos_top_100_last_week.csv.
           Available columns include:
           - `dates`: Date and time of the record (9 unique timestamps, most frequent is "11/19/2024 0:00").
           - `symbol`: Cryptocurrency code (e.g., MANA-USD), 100 unique values.
           - `open`, `close`, `high`, `low`: Prices in USD with a wide range (0.0 to 467.33 USD).
           - `volume`: Trading volume (0 to ~20 billion, median ~85,000).
           - `adj_close`: Adjusted closing price (same scale as `close`).

           Key insights:
           - Significant variability in prices and trading volumes between cryptocurrencies.
           - Data contains 7 missing values in price-related columns.

           Use this tool for questions about cryptocurrency prices, trends, and historical data."""
        ),
        Tool(
            name="Stock Analysis",
            func=stock_agent.run,
            description="""Useful for analyzing stock prices from World_Stock_Prices_last_week.csv.
                Available columns include:
                - `Date`: Date and time of the record (5 unique dates, most frequent is "11/22/2024 5:00").
                - `Open`, `High`, `Low`, `Close`: Prices in USD (2.55 to 976.30 USD).
                - `Volume`: Number of shares traded (0 to ~400 million, median ~4.5 million).
                - `Dividends`: Monetary values (mostly 0.0).
                - `Stock Splits`: All values are 0 (no splits last week).
                - `Brand_Name` and `Ticker`: Company details (61 unique tickers, e.g., "PTON", "CROX").
                - `Industry_Tag`: Industry classification (23 categories, "technology" is most common).
                - `Country`: Country of origin (7 unique, most frequent is "usa").
            
                Key insights:
                - Contains detailed stock trading data segmented by company, industry, and country.
                - `Capital Gains` column has no valid data.
            
                Use this tool for questions about stock prices, company performance, and market trends."""
        ),
        Tool(
            name="Sustainability Analysis",
            func=sustainability_agent.run,
            description="""Useful for analyzing sustainability data from most sustainable corporations.csv.
            Available columns include:
            - `Rank`: Sustainability ranking (1 to 100).
            - `Previous Rank`: Last year’s rank (68 valid values, some missing).
            - `Company`: Company name (100 unique).
            - `Location`: Headquarters location (76 unique, e.g., "Paris, France").
            - `Industry`: Sector (38 unique, e.g., "Banks" is the most common with 10 companies).
            - `Revenue` and `Profit %`: Monetary and percentage data (requires cleaning for analysis).
            - `CEO Pay Ratio`: Ratio of CEO pay to average employee income (e.g., "70:1").
            - `Women on Board %`, `Women in Leadership %`, `Women in Workforce %`: Gender representation metrics.
            - `Climate Grade`: Sustainability grades (A+, A, etc., most frequent is C+).
            - `Sustainability Initiatives`: Key initiatives (e.g., "1.5°C, SBTi", most common with 48 occurrences).
            ****CRITICAL INFORMATION ABOUT CLIMATE GRADES******:
            - Order from BEST to WORST is: 
              A+ (best) > A > B+ > B > C+ > C > C- > D+ > D > F (worst)
            - Any grade with '+' is better than the base grade
            - Any grade with '-' is worse than the base grade
        
            Key insights:
            - Highlights gender representation, climate grades, and sustainability initiatives.
            - Contains missing values in `Previous Rank`, `Revenue`, and `Sustainability Initiatives`.
        
            Use this tool for questions about sustainable corporations and their key metrics."""
        ),
        Tool(
            name="Combined DJIA Analysis",
            func=combined_agent.run,
            description="""Useful for analyzing combined news and DJIA data from Combined_News_DJIA_last_six_months_fixed.csv.
        Available columns include:
        - `Date`: Record date (126 unique).
        - `Label`: Market movement indicator (0 = DJIA down, 1 = DJIA up, balanced distribution ~56% up).
        - `Top1` to `Top25`: News headlines for each day (text data).
    
        Key insights:
        - Provides a mix of numerical and text data linking market movements to daily news.
        - Ideal for NLP applications or correlating news sentiment with financial trends.
    
        Use this tool for questions relating news headlines to market performance.
        Just show the top 1 when asked"""
        )
    ])

    # Modificar la creación del grand agent
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        tools=tools,
    )

    agent_executor = AgentExecutor(
        agent=grand_agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,  # Añadir un límite de iteraciones
        return_intermediate_steps=True  # Para mejor debugging
    )

    # Python Agent predefined tasks
    python_tasks = [
        "Calcula la suma de 2 y 3",
        "Genera una lista del 1 al 10",
        "Crea una función que calcule el factorial de un número",
        "Crea un juego básico de snake con la librería pygame"
    ]

    # UI Components
    st.markdown("### Python Agent Tasks")
    selected_task = st.selectbox(
        "Select a predefined Python task:",
        python_tasks
    )

    if st.button("Execute Python Task"):
        try:
            with st.spinner("Executing Python task..."):
                response = agent_executor.invoke({"input": selected_task})
                st.code(response["output"], language="python")
                save_history(selected_task, response["output"])
        except Exception as e:
            st.error(f"Error executing task: {str(e)}")

    st.markdown("### Data Analysis Queries")
    example_queries = [
        "¿Cuáles son las 5 criptomonedas con mayor volumen de trading esta semana?",
        "Muestra las empresas con mejor Climate Grade (A+) y mayor porcentaje de mujeres en liderazgo (100%)",
        "¿Muestra 2 noticias con influencia positiva en DJIA?",
        "Compara el rendimiento de las top 3 acciones por volumen de la última semana"
    ]

    selected_query = st.selectbox(
        "Example queries (select or write your own below):",
        example_queries
    )

    user_query = st.text_area(
        "Enter your query about the datasets or request a Python program:",
        value=selected_query,
        height=100
    )

    if st.button("Execute Query"):
        try:
            with st.spinner("Processing query..."):
                response = agent_executor.invoke({"input": user_query})
                st.write("### Response:")
                st.write(response["output"])
                save_history(user_query, response["output"])
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")

    # Show history
    if st.checkbox("Show History"):
        history = load_history()
        for item in history:
            st.text(item)


if __name__ == "__main__":
    main()