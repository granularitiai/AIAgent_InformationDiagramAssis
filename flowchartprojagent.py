import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
from autogen.agentchat.contrib.capabilities.vision_capability import VisionCapability
from autogen.agentchat.contrib.img_utils import get_pil_image, pil_to_data_uri
from autogen.agentchat.contrib.multimodal_conversable_agent import MultimodalConversableAgent
from autogen.code_utils import content_str
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from pyflowchart import *
import matplotlib.pyplot as plt
import numpy
from PIL import Image
from termcolor import colored
import requests
from graphviz import Digraph

config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-3.5-turbo-0125"],
    },
)

llm_config = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_list_gpt4,
    "temperature": 0,
}

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message="A human admin. Interact with the diagram_creator and the architectural_expert to discuss the information flowchart diagram. Diagram execution needs to be approved by the admin.",
    code_execution_config=False,
)

diagram_creator = autogen.AssistantAgent(
    name="Diagram_Creator",
    system_message="As an expert in generating creative information flowcharts and software architectural diagrams, you will recommend python packages and methods that the engineer should use. The Diagram should be insightful, clear, organized and proportional.",
    llm_config=llm_config,
)

architectural_expert = autogen.AssistantAgent(
    name="Architectural_Expert",
    system_message= "As a technical expert in software architecture diagrams you will recommend methodologies and ideas to the diagram_creator to follow. do not code.",
    llm_config=llm_config,
)

engineer = autogen.AssistantAgent(
    name = "Engineer",
    system_message= """"Engineer. You follow an approved plan by the diagram_creator and architectural_expert. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. The user can't modify your code. So do not suggest incomplete code which requires others to modify. Don't use a code block if it's not intended to be executed by the executor.
Don't include multiple code blocks in one response. Do not ask others to copy and paste the result. Check the execution result returned by the executor.
If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
""",
llm_config=llm_config,
)

executor = autogen.UserProxyAgent(
    name = "Executor",
    system_message= "Executor. Execute the code written by the engineer and report the result.",
    human_input_mode="NEVER",
    code_execution_config={
        "last_n_messages": 3,
        "work_dir": "paper",
        "use_docker": False,
    },
)

critic = autogen.AssistantAgent(
    name="Critic",
    system_message= " Critic. Double check code and diagrams from the engineer and diagram_creator and provide feedback. Please make sure all graphics and images are formatted correctly.",
    llm_config=llm_config,
)

groupchat = autogen.GroupChat(
    agents = [user_proxy, diagram_creator, architectural_expert, engineer, executor, critic], messages=[], max_round= 50
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

user_proxy.initiate_chat(
    manager,
    message= """ please create a information flow diagram that is clear, insightful, organized and proportional based off these current steps:
    1. User expresses command to app.
    2. Application produces an ouput.
    3. Output is sent back to user. """
)


