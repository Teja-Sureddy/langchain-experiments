"""
Query Postgres → Write to Excel → Upload to S3 → Send Email
"""
import os
import textwrap
import warnings
import smtplib
from email.message import EmailMessage

import boto3
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from langchain.agents import initialize_agent, Tool
from langchain_community.llms import Ollama
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core._api.deprecation import LangChainDeprecationWarning


warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
load_dotenv()
user, password = os.environ.get('POSTGRES_USERNAME'), os.environ.get('POSTGRES_PASSWORD')
engine = create_engine(f"postgresql+psycopg2://{user}:{password}@localhost:5432/postgres")
TMP_DIR = "../tmp/"


def parse_tool_input(input_str: str) -> dict:
    input_str = input_str.strip()
    params = {}
    for each_input in input_str.split(','):
        key, value = each_input.split("=")
        params[key.strip()] = value.strip().replace('"', '').replace("'", '')
    return params


# -------------------- TOOLS --------------------
def write_excel_tool(input_str: str) -> str:
    print(f"\nwrite_excel_tool: {input_str}")
    try:
        params = parse_tool_input(input_str)
        query = params.get('query')
        filename = params.get('filename', 'output.xlsx')

        if not query:
            return "Error: 'query' parameter is required"

        df = pd.read_sql_query(query, con=engine)
        filepath = os.path.join(TMP_DIR, filename)
        df.to_excel(filepath, index=False)
        return f"{filename} written with {len(df)} rows"
    except Exception as e:
        return f"Error in write_excel_tool: {e}"


def upload_s3_tool(input_str: str) -> str:
    print(f"\nupload_s3_tool: {input_str}")
    try:
        params = parse_tool_input(input_str)

        filename = params.get('filename', 'output.xlsx')
        s3_path = params.get('s3_path', 'uploads/')

        filepath = os.path.join(TMP_DIR, filename)
        key = f"test/{s3_path.strip('/')}/{filename}"
        bucket = "procredio-data-local"

        s3 = boto3.client('s3')
        s3.upload_file(filepath, bucket, key)
        url = f"https://{bucket}.s3.amazonaws.com/{key}"
        return url
    except Exception as e:
        return f"Error in upload_s3_tool: {e}"


def send_email_tool(input_str: str) -> str:
    print(f"\nsend_email_tool: {input_str}")
    try:
        params = parse_tool_input(input_str)
        filename = params.get('filename', 'output.xlsx')
        email = params.get('email')
        s3_url = params.get('s3_url')

        if not email or not s3_url:
            return "Error: 'email' and 's3_url' parameters are required"

        filepath = os.path.join(TMP_DIR, filename)
        df = pd.read_excel(filepath)
        row_count = len(df.index)

        msg = EmailMessage()
        msg['Subject'] = 'Your requested data'
        msg['From'] = 'test@langchain.com'
        msg['To'] = email
        msg.set_content(f"S3 URL: {s3_url}\nRow count: {row_count}")

        with smtplib.SMTP("sandbox.smtp.mailtrap.io", 2525) as server:
            server.starttls()
            server.login(os.environ.get('MAILTRAP_USERNAME'), os.environ.get('MAILTRAP_PASSWORD'))
            server.send_message(msg)

        return f"Email sent to {email} with {row_count} rows and link {s3_url}"
    except Exception as e:
        return f"Error in send_email_tool: {e}"


# -------------------- MAIN --------------------
def main(model: str = 'gemma3'):
    db = SQLDatabase(engine)
    llm = Ollama(model=model)

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = toolkit.get_tools()

    tools = sql_tools + [
        Tool(
            name="write_to_excel",
            func=write_excel_tool,
            description="""
                Write SQL query results to Excel file. 
                Input format:
                query=SELECT * FROM table
                filename=output.xlsx
            """
        ),
        Tool(
            name="upload_to_s3",
            func=upload_s3_tool,
            description="""
                Upload file to S3. 
                Input format: 
                filename=output.xlsx
                s3_path=uploads/
            """
        ),
        Tool(
            name="send_email",
            func=send_email_tool,
            description="""
                Send email with file info. 
                Input format: 
                filename=output.xlsx
                email=user@example.com
                s3_url=https://...
            """
        ),
    ]
    agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    while True:
        prompt = input("\nEnter Prompt: ") or (
            "First, generate the sql query to get emails from customers table where id is less than 200011 "
            "write the file to excel with file name 'test_users.xlsx' "
            "Then upload the file to S3 with s3 path uploads/ "
            "Finally, send an email to 'test@gmail.com' with the S3 link and the count."
        )
        response = agent.run(prompt)
        print(textwrap.fill(response, width=100))


if __name__ == "__main__":
    main()
