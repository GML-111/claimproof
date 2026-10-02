import os
import json
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="ClaimProof", layout="wide")

# ============ 密钥读取：本地读 .env，线上读 Streamlit secrets ============
def _get_secret(key):
    v = os.getenv(key)
    if v:
        return v
    try:
        return st.secrets[key]
    except Exception:
        return None

# ============ 1. 工具注册表（Agent Harness 机制展示） ============
FUYAO_KEY = _get_secret("FUYAO_API_KEY")
BASE_URL = "https://fuyao.aicubes.cn"

def tool_get_snapshot(thscode: str):
    """工具1：获取A股行情快照"""
    url = f"{BASE_URL}/api/a-share/prices/snapshot"
    r = requests.get(url, headers={"X-api-key": FUYAO_KEY},
                     params={"thscodes": thscode}, timeout=30)
    d = r.json()
    if d.get("code") != 0:
        raise RuntimeError(f"扶摇接口错误 code={d.get('code')}: {d.get('message')}")
    item = d["data"]["item"][0]
    return {
        "source": "扶摇-行情快照",
        "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(d["data"]["timestamp"]/1000)),
        "unit": "元",
        "caliber": "实时快照",
        "raw_field": "last_price, price_change_ratio_pct",
        "value": {"最新价": item["last_price"], "涨跌幅%": item["price_change_ratio_pct"]},
    }

def tool_get_income(thscode: str):
    """工具2：获取利润表多期序列"""
    url = f"{BASE_URL}/api/a-share/financials/income-statements"
    r = requests.get(url, headers={"X-api-key": FUYAO_KEY},
                     params={"thscode": thscode, "period": "annual", "limit": 4}, timeout=30)
    d = r.json()
    if d.get("code") != 0:
        raise RuntimeError(f"扶摇接口错误 code={d.get('code')}: {d.get('message')}")
    items = d["data"].get("item", [])
    if not items:
        raise RuntimeError("利润表无数据")
    return {
        "source": "扶摇-利润表",
        "time": "最近4期年报",
        "unit": "元",
        "caliber": "合并报表",
        "raw_field": "revenue, net_profit_attr_p",
        "value": items[:4],
    }

TOOLS = {
    "get_snapshot": {"fn": tool_get_snapshot, "desc": "行情快照"},
    "get_income": {"fn": tool_get_income, "desc": "利润表多期"},
}

# ============ 2. LLM 客户端 ============
client = OpenAI(
    api_key=_get_secret("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

def llm_decompose(claim: str):
    """第一步：AI 拆解命题，返回 thscode 和子问题"""
    prompt = f"""你是投资研究助手。用户提出一个投资命题，请拆解。

命题：{claim}

请严格输出 JSON（不要任何其他文字）：
{{
  "target_name": "涉及的上市公司名称",
  "thscode": "6位代码.交易所后缀，例如 600519.SH 或 000001.SZ",
  "sub_questions": ["子问题1", "子问题2", "子问题3"]
}}

如果命题没有明确公司，请选一个最相关或最典型的A股公司。"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def llm_evidence(claim: str, sub_questions, data_bundle):
    """第二步：AI 基于真实数据生成证据卡"""
    prompt = f"""你是投资研究助手。请基于真实数据，对以下命题形成证据链。

命题：{claim}
子问题：{json.dumps(sub_questions, ensure_ascii=False)}
真实数据：{json.dumps(data_bundle, ensure_ascii=False, default=str)}

严格输出 JSON：
{{
  "evidence": [
    {{
      "sub_question": "对应子问题",
      "stance": "支持 / 反对 / 无法验证",
      "finding": "基于数据的事实描述",
      "source": "数据来源",
      "time": "时点",
      "unit": "单位",
      "caliber": "统计口径",
      "confidence": "高 / 中 / 低"
    }}
  ],
  "conflicts": ["证据冲突或数据缺失说明，没有就空数组"],
  "conclusion": {{
    "judgement": "支持 / 部分支持 / 不支持 / 无法验证",
    "reason": "结论理由",
    "uncertainty": "不确定性说明",
    "what_changes": "哪些信息变化会改变结论"
  }}
}}

规则：
- 事实和推断必须区分
- 数据缺失或冲突不得静默生成正常结论
- 不得输出买卖建议、收益承诺、涨跌预测"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

# ============ 3. 页面 ============
st.title("ClaimProof｜投资命题多证据验证器")
st.caption("输入投资命题 → AI 拆解 → 调用真实数据 → 支持/反对/无法验证证据链 → 条件化结论")

# 侧边栏：工具注册表 + 运行日志
with st.sidebar:
    st.header("🛠 工具注册表")
    for name, meta in TOOLS.items():
        st.write(f"• **{name}** — {meta['desc']}")
    st.divider()
    st.header("📋 执行日志")
    log_box = st.empty()

claim = st.text_area("输入投资命题", placeholder="例如：贵州茅台盈利改善来自主营业务", height=80)

if st.button("开始验证", type="primary"):
    if not claim.strip():
        st.warning("请先输入命题")
        st.stop()

    logs = []
    def log(msg):
        logs.append(f"{time.strftime('%H:%M:%S')} {msg}")
        log_box.code("\n".join(logs))

    try:
        # 步骤1：AI 拆解
        log("① AI 拆解命题...")
        plan = llm_decompose(claim)
        thscode = plan.get("thscode", "")
        log(f"   标的: {plan.get('target_name')} ({thscode})")

        st.subheader("① 命题拆解")
        st.write(f"**标的**：{plan.get('target_name')} `{thscode}`")
        for i, q in enumerate(plan.get("sub_questions", []), 1):
            st.write(f"{i}. {q}")

        # 步骤2：调用工具
        log("② 调用扶摇接口...")
        data_bundle = {}
        errors = []
        try:
            data_bundle["snapshot"] = TOOLS["get_snapshot"]["fn"](thscode)
            log("   ✓ 行情快照")
        except Exception as e:
            errors.append(f"行情快照失败: {e}")
            log(f"   ✗ 行情快照失败: {e}")
        try:
            data_bundle["income"] = TOOLS["get_income"]["fn"](thscode)
            log("   ✓ 利润表")
        except Exception as e:
            errors.append(f"利润表失败: {e}")
            log(f"   ✗ 利润表失败: {e}")

        # 步骤3：AI 生成证据链
        log("③ AI 组织证据链...")
        result = llm_evidence(claim, plan.get("sub_questions", []), data_bundle)

        # 步骤4：展示证据卡
        st.subheader("② 证据链")
        for ev in result.get("evidence", []):
            stance = ev.get("stance", "无法验证")
            color = {"支持": "🟢", "反对": "🔴", "无法验证": "⚪"}.get(stance, "⚪")
            with st.expander(f"{color} {stance}｜{ev.get('sub_question', '')}"):
                st.write(f"**发现**：{ev.get('finding', '')}")
                st.caption(f"来源：{ev.get('source', '')} ｜ 时点：{ev.get('time', '')} ｜ 单位：{ev.get('unit', '')} ｜ 口径：{ev.get('caliber', '')} ｜ 置信度：{ev.get('confidence', '')}")

        # 步骤5：冲突
        conflicts = result.get("conflicts", [])
        if conflicts:
            st.subheader("③ 证据冲突 / 数据缺失")
            for c in conflicts:
                st.warning(c)

        if errors:
            st.subheader("⚠️ 工具调用异常")
            for e in errors:
                st.error(e)

        # 步骤6：结论
        st.subheader("④ 条件化结论")
        concl = result.get("conclusion", {})
        st.info(f"**当前判断**：{concl.get('judgement', '无法验证')}")
        st.write(f"**理由**：{concl.get('reason', '')}")
        st.write(f"**不确定性**：{concl.get('uncertainty', '')}")
        st.write(f"**什么变化会改变结论**：{concl.get('what_changes', '')}")

        st.caption("⚠️ 本产品仅供研究参考，不构成任何投资建议。事实、推断与不确定信息已区分标注。")

        # 保存到会话（记忆）
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.append({"claim": claim, "result": result})
        log("④ 完成，已保存到研究任务")

    except Exception as e:
        st.error(f"执行失败：{e}")
        log(f"✗ 失败：{e}")

# 历史任务（记忆 + 检查点）
if "history" in st.session_state and st.session_state.history:
    st.divider()
    st.subheader("📚 已保存的研究任务")
    for i, h in enumerate(reversed(st.session_state.history), 1):
        st.write(f"{i}. {h['claim']}")