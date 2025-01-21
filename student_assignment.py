import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
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



def generate_hw01(question):

    response_schema = [
        ResponseSchema(name = 'result', description= '請將節慶放到此處', type= 'list'),
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
    return response


def generate_hw02(question):
    pass
    
def generate_hw03(question2, question3):
    pass
    
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
