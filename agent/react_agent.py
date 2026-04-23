import sys
import os
# 获取当前文件（vector_store.py）的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件所在目录（rag目录）
rag_dir = os.path.dirname(current_file_path)
# 获取根目录（heima_agent目录，即rag的父目录）
root_dir = os.path.dirname(rag_dir)
# 把根目录加入sys.path
sys.path.append(root_dir)

from langchain.agents import create_agent
from utils.prompt_loader import load_system_prompts
from model.factory import chat_model
from agent.tools.agent_tools import rag_summarize, get_weather, get_user_location,get_current_month,get_user_id,fetch_external_data,fill_context_for_report
from agent.tools.middleware import monitor_tool,log_before_model,report_prompt_switch

class ReactAgent():
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location,get_current_month,get_user_id,fetch_external_data,fill_context_for_report],
            middleware=[monitor_tool,log_before_model,report_prompt_switch],
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content":query}
            ]
        }
        # 第三个参数context就是上下文runtime中的信息
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip()+'\n'

if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("扫地机器人在我所在地区的气温下如何保养"):
        print(chunk, end="",flush=True)