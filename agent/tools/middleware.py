# callable就是函数
from typing import Callable
from langchain.agents.middleware import ModelRequest
from langchain.agents.middleware import dynamic_prompt,wrap_model_call,wrap_tool_call,before_model
from langchain.agents import AgentState
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langgraph.runtime import Runtime
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompts, load_report_prompts

@wrap_tool_call
def monitor_tool(
    # 请求的数据封装
    request: ToolCallRequest,
    # 执行的函数本身
    handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:       # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")
    try:
        result = handler(request)
        logger.info(f"")
        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.info(f"{str(e)}")
        raise e

@before_model
def log_before_model(                  # 在模型执行前输出日志
    state: AgentState,                 # 整个Agent智能体中的状态记录
    runtime: Runtime                   # 整个执行过程中的状态信息
):   
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息")
    
    logger.debug(f"[log_before_model]{state['messages'][-1]}{state['messages'][-1].content.strip()}条消息")
    return None

@dynamic_prompt      # 每次生成提示词前调用此函数
def report_prompt_switch(request: ModelRequest): # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompts()
    return load_system_prompts()