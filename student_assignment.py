import json
import traceback
import requests
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from model_configurations import get_model_configuration
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)
llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )


def generateAnswer(response):
    response = response[8:-4]
    return json.dumps(json.loads(response), ensure_ascii=False, indent=4, sort_keys=True)

def generate_hw01(question):

    response_schema = [
        ResponseSchema(name = 'Result', description= '請將節慶放到此處', type= 'list'),
        ResponseSchema(name = 'date', description= '請將節慶日期放到此處', type= 'YYYY-MM-DD'),
        ResponseSchema(name = 'name', description= '請將節慶名稱放到此處', type= 'String'),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schema)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","answer the users question as best as possible.\n{format_instructions}"),
        ("human","{question}")
    ])
    prompt = prompt.partial(format_instructions = format_instructions)
    response = llm.invoke(prompt.format_messages(question = question)).content
    return generateAnswer(response)

def get_hw2_response(year: int, month: int):
    url = f"https://calendarific.com/api/v2/holidays?&api_key=JlDYea1pV7aqbgqlxlb6P6rrDurH7Uvs&country=tw&year={year}&month={month}"
    response = requests.get(url)
    data = response.json()
    holidays = data['response']['holidays']
    return holidays
class GetValue(BaseModel):
    year: int = Field(description= "年份")
    month: int = Field(description= "月份")

def generate_hw02(question):

    agent_prompt = hub.pull("hwchase17/openai-functions-agent")

    
    tool = StructuredTool.from_function(
        name= "get_response",
        description="查詢台灣紀念日",
        func= get_hw2_response,
        args_schema=GetValue,
    )    
    tools = [tool]
    agent = create_openai_functions_agent(llm,tools,agent_prompt)
    agent_exe = AgentExecutor(agent=agent,tools=tools)
    response = agent_exe.invoke({"input":question}).get('output')
    

    response_schema = [
        ResponseSchema(name = 'Result', description= '請將節慶放到此處', type= 'list'),
        ResponseSchema(name = 'date', description= '請將節慶日期放到此處', type= 'YYYY-MM-DD'),
        ResponseSchema(name = 'name', description= '請將節慶名稱放到此處', type= 'String'),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas=response_schema)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","將所有紀念日收集後，整理成Json格式.\n{format_instructions}"),
        ("human","{question}")
    ])
    prompt = prompt.partial(format_instructions = format_instructions)

    response = llm.invoke(prompt.format_messages(question = response)).content

    return generateAnswer(response)
    

class InnerResult(BaseModel):
    add: bool
    reason: str

class Result(BaseModel):
    Result: InnerResult

def generate_hw03(question2, question3):
    agent_prompt = hub.pull("hwchase17/openai-functions-agent")

    history = ChatMessageHistory()
    def get_history() -> ChatMessageHistory:
        return history
    
    

    tool = StructuredTool.from_function(
        name= "get_response",
        description="查詢台灣紀念日",
        func= generate_hw02,
    )    
    tools = [tool]
    agent = create_openai_functions_agent(llm,tools,agent_prompt)
    agent_exe = AgentExecutor(agent=agent,tools=tools)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_exe,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    response = agent_with_chat_history.invoke({"input":question2}).get('output')

    response = agent_with_chat_history.invoke({"input":question3}).get('output')

    output_parser = PydanticOutputParser(pydantic_object=Result)
    format_instructions = output_parser.get_format_instructions()
    prompt = ChatPromptTemplate.from_messages([
        ("system","回答使用者問題後，整理成Json格式.\n{format_instructions}"),
        ("human","{question}，請將回答放入Result中，有兩個欄位，第一add欄位，請填是否該添加此紀念日，第二，reason請填入原因")
    ])
    prompt = prompt.partial(format_instructions = format_instructions)
    response = llm.invoke(prompt.format_messages(question = response)).content
    return generateAnswer(response)
    
def generate_hw04(question):
    pass
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response





question = "2024年台灣10月紀念日有哪些?"

question2 = "2024年台灣10月紀念日有哪些?"
question3 = "蔣公誕辰紀念日是否有在該月份清單？"
print(generate_hw03(question2,question3))
