import os
import re
import sys
import logging

# 计算并添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(project_root)
sys.path.append(project_root)

from typing import Dict, Any
from agent.langgraph.state import RagState
from model.factory import chat_model
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)

# 无需检索的关键词模式：问候、闲聊、简单数学、元对话
_SKIP_PATTERNS = re.compile(
    r"^(你好|hi|hello|hey|嗨|早上好|下午好|晚上好|在吗)[\s!！.。?？]*$"
    r"|^\d+\s*[\+\-\*/×÷]\s*\d+\s*[=＝]?\s*\d*$"
    r"|^(谢谢|感谢|thanks|ok|好的|嗯|哦|再见|拜拜)[\s!！.。?？]*$",
    re.IGNORECASE,
)

# 明确需要检索的关键词
_FORCE_RETRIEVE_PATTERNS = re.compile(
    r"扫地机器人|扫地机|扫拖一体|扫地机器|故障|无法|不能|连接.{0,4}wifi|"
    r"充电|清扫|打扫|滤网|边刷|滚刷|传感器|尘盒|水箱|拖布|基座|"
    r"app.*/|维护|保养|清洁|拆|安装|更换|报错|报警|红灯|离线|脱困|"
    r"建图|划区|禁区|定时|预约|模式|吸力|噪音|续航|充电桩",
    re.IGNORECASE,
)


class AgentNode:
    """Agent节点，用于判断是否需要检索文档"""

    def __init__(self):
        self.prompt_template = PromptTemplate(
            template="""你是一个决策助手，根据用户查询和对话历史判断是否需要从知识库检索信息。

知识库范围：扫地机器人/扫拖一体机器人的产品使用、故障排查、维护保养、选购指南等。

对话历史:
{chat_history}

当前用户查询: {query}

判断规则：
1. 需要检索：查询涉及扫地机器人的具体问题（故障、操作、参数、维护、选购对比等）
2. 需要检索：对话历史中正在讨论扫地机器人相关话题，当前查询是追问或延续
3. 不需要检索：纯闲聊、问候、数学计算、与扫地机器人完全无关的问题
4. 不需要检索：通用常识问题（如"什么是WiFi"）

请严格按以下JSON格式返回，不要有任何其他内容：
{{"decision": "retrieve"}} 或 {{"decision": "skip"}}""",
            input_variables=["query", "chat_history"],
        )
        self.model = chat_model

    def _quick_classify(self, query: str) -> bool | None:
        """基于规则的快速分类，返回 True(需要检索)/False(不需要)/None(无法确定)"""
        stripped = query.strip()
        if not stripped:
            return False
        # 问候/闲聊/数学 → 直接跳过
        if _SKIP_PATTERNS.match(stripped):
            return False
        # 明确包含产品关键词 → 直接检索
        if _FORCE_RETRIEVE_PATTERNS.search(stripped):
            return True
        return None

    def _format_chat_history(self, chat_history: list) -> str:
        """将对话历史格式化为文本"""
        if not chat_history:
            return "（无对话历史）"
        lines = []
        for msg in chat_history[-6:]:  # 最近3轮（每轮含 user+assistant）
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _parse_llm_decision(self, content: str) -> bool:
        """解析LLM返回的决策结果，支持多种格式"""
        text = content.strip()
        # 优先匹配 JSON 格式
        json_match = re.search(r'"decision"\s*:\s*"(retrieve|skip)"', text)
        if json_match:
            return json_match.group(1) == "retrieve"
        # 兜底：关键词模糊匹配
        if re.search(r"需要检索|retrieve", text, re.IGNORECASE):
            return True
        return False

    def should_retrieve(self, state: RagState) -> RagState:
        """判断是否需要检索

        优先使用规则快速判断，无法确定时再调用LLM。
        """
        query = state["query"]
        logger.info("AgentNode query: %s", query)

        # 第一层：规则快速分类
        quick = self._quick_classify(query)
        if quick is not None:
            logger.info("AgentNode 快速分类结果: %s", quick)
            return {"needs_retrieval": quick}

        # 第二层：LLM 分类（结合对话历史）
        chat_history_text = self._format_chat_history(state.get("chat_history", []))
        prompt = self.prompt_template.format(query=query, chat_history=chat_history_text)
        response = self.model.invoke(prompt)
        logger.info("AgentNode LLM 原始返回: %s", response.content)

        needs_retrieval = self._parse_llm_decision(response.content)
        logger.info("AgentNode 最终决策 needs_retrieval=%s", needs_retrieval)

        return {"needs_retrieval": needs_retrieval}


def _new_state(query: str, chat_history: list | None = None) -> RagState:
    """构建测试用 state"""
    return {
        "query": query,
        "session_id": "",
        "chat_history": chat_history or [],
        "needs_retrieval": False,
        "is_relevant": False,
        "rewrite_count": 0,
        "rewritten_query": "",
        "retrieved_docs": [],
        "relevance_score": 0.0,
        "response": "",
        "error": "",
    }


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    agent_node = AgentNode()
    passed = 0
    failed = 0

    def run_test(name: str, query: str, expected: bool, chat_history: list | None = None):
        global passed, failed
        print(f"\n{'='*50}")
        print(f"测试: {name}")
        print(f"查询: {query}")
        if chat_history:
            print(f"对话历史: {chat_history[-1]}")  # 只显示最近一条
        state = _new_state(query, chat_history)
        t0 = time.time()
        result = agent_node.should_retrieve(state)
        elapsed = time.time() - t0
        actual = result["needs_retrieval"]
        ok = actual == expected
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"期望: {expected}  实际: {actual}  耗时: {elapsed:.3f}s  [{status}]")

    # ---- 第一组：规则快速分类（不调用 LLM） ----
    print("\n" + "=" * 50)
    print("第一组：规则快速分类")

    run_test("产品关键词-故障", "扫地机器人无法连接wifi了，怎么办？", True)
    run_test("产品关键词-维护", "如何清洁扫地机器人的滤网？", True)
    run_test("问候语", "你好", False)
    run_test("数学计算", "1+1等于多少？", False)
    run_test("闲聊结束语", "谢谢！", False)
    run_test("产品关键词-充电", "充电总是中断", True)

    # ---- 第二组：LLM 分类（规则无法确定） ----
    print("\n" + "=" * 50)
    print("第二组：LLM 分类")

    run_test("无关问题", "今天天气怎么样？", False)
    run_test("通用常识", "什么是WiFi？", False)

    # ---- 第三组：多轮对话上下文 ----
    print("\n" + "=" * 50)
    print("第三组：多轮对话上下文")

    run_test(
        "追问-充电",
        "那充电呢？",
        True,
        chat_history=[
            {"role": "user", "content": "扫地机器人无法开机怎么办？"},
            {"role": "assistant", "content": "请检查电池是否有电，尝试充满电后再开机。"},
        ],
    )
    run_test(
        "无关追问",
        "你觉得呢？",
        False,
        chat_history=[
            {"role": "user", "content": "你喜欢猫还是狗？"},
            {"role": "assistant", "content": "我更喜欢猫。"},
        ],
    )

    # ---- 第四组：边界情况 ----
    print("\n" + "=" * 50)
    print("第四组：边界情况")

    run_test("空查询", "", False)
    run_test("产品关键词-报错", "红灯闪烁报错E001", True)

    # ---- 汇总 ----
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"测试完成: {passed}/{total} 通过, {failed} 失败")

