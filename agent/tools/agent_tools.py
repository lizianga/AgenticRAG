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
from langchain_core.tools import tool
from rag.rag_service import RagSummariceService
import random
from utils.config_handler import agent_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger
rag = RagSummariceService()
user_ids = ["0001","0002","0003","0004","0005","0006"]
month_arr = ["2025-1", "2025-2", "2025-3", "2025-4", "2025-5", "2025-6", "2025-7", "2025-8"
             , "2025-9", "2025-10", "2025-11", "2025-12"]

@tool(description="从向量存储中检索参考资料")
def rag_summarize(query: str) ->str:
    return rag.rag_summarize(query)

@tool(description="获取指定城市的天气，以消息字符串的形式返回")
def get_weather(city):
    return f"城市{city}天气为晴天，天气26摄氏度，空气湿度50%，南风1级"

@tool(description="获取用户所在城市名称，以纯字符串形式返回")
def get_user_location()->str:
    return random.choice(['深圳','合肥', '杭州'])

@tool(description="获取用户ID，以纯字符串形式返回")
def get_user_id() -> str:
    return random.choice(user_ids)

@tool(description="获取当前月份，以纯字符串形式返回")
def get_current_month():
    return random.choice(month_arr)

external_data = {}
def genarater_external_data():
    '''
    {
        user_id: {
            month:{"feature":***},
            month:{"feature":***},
        },
        user_id: {
            month:{"feature":***},
            month:{"feature":***},
        }
    }
    '''
    if not external_data:
        external_data_path = get_abs_path(agent_conf["external_data_path"])
        if not os.path.exists(external_data_path):
            raise FileNotFoundError(f"外部数据文件{external_data_path}不存在")
        with open(external_data_path, "r", encoding='utf-8') as f:
            for line in f.readlines()[1:]:
                arr:list[str] = line.strip().split(",")    # 去前后空格
                user_id = arr[0].replace('"','')
                feature = arr[1].replace('"','')
                efficiency = arr[2].replace('"','')
                consumables = arr[3].replace('"','')
                comparision = arr[4].replace('"','')
                time = arr[5].replace('"','')
                if user_id not in external_data:
                    external_data[user_id] = {}
                external_data[user_id][time] = {
                    "特征": feature,
                    "效率": efficiency,
                    "耗材": consumables,
                    "对比": comparision,
                }


@tool(description="从外部系统中获取指定用户在指定月份的使用记录，以纯字符串形式返回")
def fetch_external_data(user_id, month):
    genarater_external_data()
    try:
        return external_data[user_id][month]
    
    except KeyError as e:
        logger.warning(f"【fetch_external_data】未能检索到用户：{user_id}在{month}的使用记录")
        return ""
    
@tool(description="无入参，无返回值，调用后触发中间件为报告生成的场景动态注入上下文信息，为后续提示词切换提供上下文信息")
def fill_context_for_report():
    return "1023"

