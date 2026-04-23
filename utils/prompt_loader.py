from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger

def load_system_prompts():
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置中没有主提示词路径")

    try:
        return open(system_prompt_path, "r", encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词出错，{str(e)}")
        raise e


def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_prompt_path"])
        # print(rag_prompt_path)
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置中没有rag提示词路径")

    try:
        return open(rag_prompt_path, "r", encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析rag提示词出错，{str(e)}")
        raise e
    
def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置中没有report提示词路径")

    try:
        return open(report_prompt_path, "r", encoding='utf-8').read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析report提示词出错，{str(e)}")
        raise e
    

if __name__ == '__main__':
    print(load_system_prompts())