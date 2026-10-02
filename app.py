import os
import json
import time
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
st.set_page_config(page_title="ClaimProof", layout="wide", page_icon="🧭")

# ============ 密钥 ============
def _get_secret(key):
    v = os.getenv(key)
    if v:
        return v
    try:
        return st.secrets[key]
    except Exception:
        return None

FUYAO_KEY = _get_secret("FUYAO_API_KEY")
BASE_URL = "https://fuyao.aicubes.cn"

# ============ 工具 ============
def tool_get_snapshot(thscode: str):
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
        "unit": "元", "caliber": "实时快照",
        "raw_field": "last_price, price_change_ratio_pct",
        "value": {"最新价": item["last_price"], "涨跌幅%": item["price_change_ratio_pct"]},
    }

def tool_get_income(thscode: str):
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
        "source": "扶摇-利润表", "time": "最近4期年报",
        "unit": "元", "caliber": "合并报表",
        "raw_field": "revenue, net_profit_attr_p",
        "value": items[:4],
    }

TOOLS = {
    "get_snapshot": {"fn": tool_get_snapshot, "desc": "行情快照"},
    "get_income": {"fn": tool_get_income, "desc": "利润表多期"},
}

# ============ LLM ============
client = OpenAI(api_key=_get_secret("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

def llm_clarify(claim: str):
    prompt = f"""你是投资研究助手。用户提出一个投资命题，命题可能模糊、口径不清，需要澄清和修订。

原始命题：{claim}

严格输出 JSON：
{{
  "revised_claim": "修订后口径清晰的命题，包含标的、时间范围、指标、对比基准",
  "target_name": "涉及的上市公司名称",
  "thscode": "6位代码.交易所后缀",
  "assumptions": ["口径假设1", "口径假设2", "口径假设3"],
  "clarify_questions": [
    {{
      "dimension": "维度名称（如：时间维度、主营业务口径、对比基准）",
      "question": "具体澄清问题",
      "options": ["选项1", "选项2", "选项3"]
    }}
  ]
}}

规则：
- 修订后命题必须可验证、边界明确
- clarify_questions 提出 2-4 个关键澄清点，每个点给出 3-4 个候选选项
- 选项要具体、有边界"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def llm_decompose(revised_claim, thscode, target_name, user_clarifications=""):
    prompt = f"""你是投资研究助手。用户已确认以下修订后的投资命题，请拆解为可验证子问题。

修订后命题：{revised_claim}
标的：{target_name}（{thscode}）
用户补充澄清：{user_clarifications if user_clarifications else "无"}

严格输出 JSON：
{{"sub_questions": ["子问题1", "子问题2", "子问题3"]}}

子问题必须：可被数据验证、从不同角度验证命题、3-4 个即可。"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def llm_evidence(revised_claim, sub_questions, data_bundle, user_clarifications=""):
    prompt = f"""你是投资研究助手。请基于真实数据，对以下命题形成证据链。

修订后命题：{revised_claim}
用户补充澄清：{user_clarifications if user_clarifications else "无"}
子问题：{json.dumps(sub_questions, ensure_ascii=False)}
真实数据：{json.dumps(data_bundle, ensure_ascii=False, default=str)}

严格输出 JSON：
{{
  "evidence": [
    {{
      "id": 1,
      "sub_question": "对应子问题",
      "stance": "支持 / 反对 / 无法验证",
      "finding": "基于数据的事实描述，至少 80 字，必须包含具体数字，并引用原始字段名",
      "source": "数据来源",
      "time": "时点",
      "unit": "单位",
      "caliber": "统计口径",
      "confidence": "高 / 中 / 低"
    }}
  ],
  "conflicts": ["证据冲突或数据缺失说明，每条不超过 40 字"],
  "conclusion": {{
    "judgement": "支持 / 部分支持 / 不支持 / 无法验证",
    "reason": "结论理由，必须引用证据编号",
    "evidence_ids": [1, 3],
    "uncertainty": "不确定性说明",
    "what_changes": "哪些信息变化会改变结论"
  }}
}}

规则：
- finding 必须详细，带具体数字，不允许空泛
- 事实和推断必须区分
- 数据缺失或冲突不得静默生成正常结论
- 不得输出买卖建议、收益承诺、涨跌预测"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def llm_followup(question, context):
    prompt = f"""你是投资研究助手。用户对之前的研究结论有追问，请基于已有数据回答。

研究上下文：
{json.dumps(context, ensure_ascii=False, default=str)[:6000]}

用户追问：{question}

用简洁的中文回答，100-300 字。要求：只基于已有数据、区分事实与推断、不输出买卖建议、数据不足时明确说明。"""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content

# ============ 图表 ============
def extract_chart_data(income_items):
    rows = []
    for it in income_items:
        if not isinstance(it, dict):
            continue
        period = None
        for k in ["period_end_ms", "period_end", "end_date", "report_date_ms", "report_date", "period"]:
            if k in it and it[k]:
                period = it[k]
                break
        if isinstance(period, (int, float)) and period > 1e10:
            period = time.strftime("%Y", time.localtime(period/1000))
        elif isinstance(period, (int, float)) and period > 1e9:
            period = time.strftime("%Y", time.localtime(period))
        rev = next((it[k] for k in ["revenue", "total_revenue", "operating_revenue", "total_operating_revenue"] if it.get(k) is not None), None)
        np_ = next((it[k] for k in ["net_profit_attr_p", "net_profit", "n_income_attr_p", "net_profit_parent"] if it.get(k) is not None), None)
        if period and (rev is not None or np_ is not None):
            rows.append({"期间": str(period)[:10], "营业收入": rev, "归母净利润": np_})
    rows.sort(key=lambda x: x["期间"])
    return rows

def render_chart(income_items, chart_key):
    rows = extract_chart_data(income_items)
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["营业收入"] = pd.to_numeric(df["营业收入"], errors="coerce")
    df["归母净利润"] = pd.to_numeric(df["归母净利润"], errors="coerce")

    def _fmt(v):
        if pd.isna(v):
            return ""
        return f"{v/1e8:.0f}亿"

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["期间"], y=df["营业收入"], name="营业收入",
        marker_color="#4C78A8",
        text=[_fmt(v) for v in df["营业收入"]],
        textposition="inside", insidetextanchor="middle",
    ))
    fig.add_trace(go.Bar(
        x=df["期间"], y=df["归母净利润"], name="归母净利润",
        marker_color="#F58518",
        text=[_fmt(v) for v in df["归母净利润"]],
        textposition="inside", insidetextanchor="middle",
    ))
    fig.update_layout(
        barmode="group", height=380,
        xaxis=dict(tickangle=0, title="报告期"),
        yaxis=dict(title="金额（元）"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
        plot_bgcolor="white",
    )
    st.plotly_chart(
        fig, use_container_width=True, key=chart_key,
        config={"staticPlot": True, "displayModeBar": False},
    )
    st.caption("数据来源：扶摇-利润表 ｜ 单位：元 ｜ 口径：合并报表")

# ============ 状态 ============
defaults = {
    "stage": "input", "original_claim": "", "revised_claim": "",
    "clarify_result": None, "user_clarifications": {},
    "plan": None, "data_bundle": None, "result": None,
    "errors": [], "elapsed": 0, "history": [],
    "followup_history": [], "review_index": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============ 侧边栏 ============
with st.sidebar:
    st.header("🛠 工具注册表")
    for name, meta in TOOLS.items():
        st.write(f"• **{name}** — {meta['desc']}")
    st.divider()
    st.header("📋 执行日志")
    log_box = st.empty()
    st.divider()
    st.header("📚 历史任务")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history)):
            idx = len(st.session_state.history) - 1 - i
            label = h["claim"][:18] + ("…" if len(h["claim"]) > 18 else "")
            if st.button(f"• {label}", key=f"hist_{idx}"):
                st.session_state.review_index = idx
                st.session_state.stage = "review"
                st.rerun()
    else:
        st.caption("暂无历史任务")

# ============ 顶部 ============
st.title("🧭 ClaimProof")
st.markdown("##### 投资命题智能验证工作台")
st.caption("把投资命题拆成可验证子问题，用真实数据形成支持 / 反对 / 无法验证的证据链，并给出条件化结论。")

html = """
<div style="display:flex; gap:10px; margin:16px 0 24px 0;">
    <div style="flex:1; border:1px solid #e0e0e0; border-radius:10px; padding:12px; background:#fafafa; text-align:center;">
        <div style="font-size:18px; font-weight:600; color:#4C78A8;">① 澄清</div>
        <div style="font-size:13px; color:#666; margin-top:4px;">修订命题 · 明确口径</div>
    </div>
    <div style="flex:1; border:1px solid #e0e0e0; border-radius:10px; padding:12px; background:#fafafa; text-align:center;">
        <div style="font-size:18px; font-weight:600; color:#4C78A8;">② 拆解</div>
        <div style="font-size:13px; color:#666; margin-top:4px;">可验证子问题</div>
    </div>
    <div style="flex:1; border:1px solid #e0e0e0; border-radius:10px; padding:12px; background:#fafafa; text-align:center;">
        <div style="font-size:18px; font-weight:600; color:#4C78A8;">③ 分析</div>
        <div style="font-size:13px; color:#666; margin-top:4px;">支持/反对/无法验证</div>
    </div>
    <div style="flex:1; border:1px solid #e0e0e0; border-radius:10px; padding:12px; background:#fafafa; text-align:center;">
        <div style="font-size:18px; font-weight:600; color:#4C78A8;">④ 结论</div>
        <div style="font-size:13px; color:#666; margin-top:4px;">条件化判断</div>
    </div>
</div>
"""
st.markdown(html, unsafe_allow_html=True)

# ============ 原始命题橙框 ============
def render_original_claim(claim, label="📝 你的原始命题"):
    st.markdown(f"""
        <div style="background:#FFF7E6; border-left:4px solid #F58518;
                    padding:12px 16px; border-radius:8px; margin-bottom:16px;">
            <div style="font-size:12px; color:#8a6d3b; font-weight:600; margin-bottom:4px;">
                {label}
            </div>
            <div style="font-size:16px; color:#2c3e50; font-weight:500;">
                {claim}
            </div>
        </div>
    """, unsafe_allow_html=True)

# ============ 结果渲染 ============
def render_result(cr, revised, plan, data_bundle, result, errors, elapsed, chart_key, readonly=False):
    thscode = cr.get("thscode", "")
    target_name = cr.get("target_name", "")

    st.subheader("① 命题澄清与修订")
    st.markdown("**🔎 识别标的**")
    st.write(f"{target_name}")
    st.caption(f"`{thscode}`")
    st.markdown("**✏️ 修订后命题**")
    st.write(revised)

    st.subheader("② 子问题拆解")
    for i, q in enumerate(plan.get("sub_questions", []), 1):
        st.write(f"{i}. {q}")

    st.subheader("③ 证据链")
    for ev in result.get("evidence", []):
        stance = ev.get("stance", "无法验证")
        color = {"支持": "🟢", "反对": "🔴", "无法验证": "⚪"}.get(stance, "⚪")
        eid = ev.get("id", "?")
        with st.expander(f"{color} 证据 #{eid}｜{stance}｜{ev.get('sub_question', '')}", expanded=True):
            st.write(f"**发现**：{ev.get('finding', '')}")
            st.caption(f"来源：{ev.get('source', '')} ｜ 时点：{ev.get('time', '')} ｜ 单位：{ev.get('unit', '')} ｜ 口径：{ev.get('caliber', '')} ｜ 置信度：{ev.get('confidence', '')}")

    if data_bundle and "income" in data_bundle:
        st.subheader("④ 财务趋势")
        render_chart(data_bundle["income"]["value"], chart_key)

    conflicts = result.get("conflicts", [])
    if conflicts:
        with st.expander(f"⑤ 证据冲突 / 数据缺失（{len(conflicts)} 条）", expanded=False):
            for c in conflicts:
                st.caption(f"⚠️ {c}")

    if errors:
        st.subheader("⚠️ 工具调用异常")
        for e in errors:
            st.error(e)

    st.subheader("⑥ 条件化结论")
    concl = result.get("conclusion", {})
    st.info(f"**当前判断**：{concl.get('judgement', '无法验证')}")
    st.write(f"**理由**：{concl.get('reason', '')}")
    eids = concl.get("evidence_ids", [])
    if eids:
        st.caption(f"📎 本结论基于证据：{', '.join(['#' + str(i) for i in eids])}")
    st.write(f"**不确定性**：{concl.get('uncertainty', '')}")
    st.write(f"**什么变化会改变结论**：{concl.get('what_changes', '')}")
    st.caption("⚠️ 本产品仅供研究参考，不构成任何投资建议。事实、推断与不确定信息已区分标注。")
    if not readonly:
        st.caption(f"⏱ 本次执行耗时 {elapsed:.1f} 秒 ｜ 工具调用 {len(data_bundle)} 次 ｜ 模型调用 3 次")

# ============ 示例命题 ============
EXAMPLE_CLAIMS = [
    ("🍶 白酒·消费", "贵州茅台盈利改善来自主营业务"),
    ("🔋 新能源·电池", "宁德时代估值回落但基本面未恶化"),
    ("🏦 银行·金融", "招商银行盈利改善来自主营业务"),
    ("💊 医药·创新药", "恒瑞医药盈利增长是否来自主营业务"),
    ("💻 半导体·芯片", "中芯国际估值回落但基本面未恶化"),
    ("🏠 家电·消费", "美的集团盈利改善来自主营业务"),
    ("☂️ 保险·金融", "中国平安估值回落但基本面未恶化"),
    ("📈 券商·金融科技", "东方财富盈利增长是否来自主营业务"),
]

# ============ 阶段 1：输入 ============
if st.session_state.stage == "input":
    claim_input = st.selectbox(
        "💡 输入投资命题",
        options=[text for tag, text in EXAMPLE_CLAIMS],
        index=None,
        placeholder="点击选择示例，或直接输入自定义命题…",
        accept_new_options=True,
        key="claim_selectbox",
    )

    st.markdown("")
    if st.button("开始分析", type="primary"):
        if not claim_input or not claim_input.strip():
            st.warning("请先输入或选择命题")
        else:
            st.session_state.original_claim = claim_input.strip()
            with st.spinner("AI 正在澄清并修订命题…"):
                try:
                    cr = llm_clarify(st.session_state.original_claim)
                    st.session_state.clarify_result = cr
                    st.session_state.revised_claim = cr.get("revised_claim", "")
                    st.session_state.user_clarifications = {}
                    st.session_state.stage = "clarify"
                    st.rerun()
                except Exception as e:
                    st.error(f"澄清失败：{e}")

# ============ 阶段 2：澄清 ============
elif st.session_state.stage == "clarify":
    cr = st.session_state.clarify_result

    st.markdown("""
        <style>
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            padding-left: 28px;
        }
        div[data-testid="stRadio"] label p {
            font-size: 13px !important;
            color: #555 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    render_original_claim(st.session_state.original_claim)

    st.subheader("① 命题澄清与修订")

    st.markdown("**🔎 识别标的**")
    st.write(f"{cr.get('target_name', '')}")
    st.caption(f"`{cr.get('thscode', '')}`")

    st.markdown("**📐 口径假设**")
    for a in cr.get("assumptions", []):
        st.caption(f"• {a}")

    st.markdown("**✏️ 修订后命题**（可编辑）")
    revised = st.text_area("修订后命题", value=st.session_state.revised_claim,
                           height=140, label_visibility="collapsed")
    st.session_state.revised_claim = revised

    questions = cr.get("clarify_questions", [])
    if questions:
        st.markdown("**🔧 关键澄清点**")
        st.caption("点击选项后会自动记录，作为后续分析的边界条件。")
        for i, q in enumerate(questions):
            dim = q.get("dimension", f"澄清{i+1}")
            qtext = q.get("question", "")
            opts = list(q.get("options", [])) + ["其他（请补充）"]
            st.markdown(f"""
                <div style="background:#f5f7fa; border-left:3px solid #4C78A8;
                            padding:8px 14px; border-radius:6px; margin:14px 0 6px 0;">
                    <div style="font-size:14px; font-weight:600; color:#2c3e50;">{dim}</div>
                    <div style="font-size:12px; color:#7a8ba0; margin-top:2px;">{qtext}</div>
                </div>
            """, unsafe_allow_html=True)
            choice = st.radio("", opts, key=f"clarify_{i}",
                              label_visibility="collapsed", horizontal=True)
            if choice == "其他（请补充）":
                choice = st.text_input("请补充说明", key=f"clarify_other_{i}")
            st.session_state.user_clarifications[dim] = choice

    st.markdown("")
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("✓ 确认并分析", type="primary"):
            st.session_state.stage = "run"
            st.rerun()
    with col2:
        if st.button("↻ 重新修订"):
            with st.spinner("AI 重新修订中..."):
                try:
                    cr = llm_clarify(st.session_state.original_claim)
                    st.session_state.clarify_result = cr
                    st.session_state.revised_claim = cr.get("revised_claim", "")
                    st.session_state.user_clarifications = {}
                    st.rerun()
                except Exception as e:
                    st.error(f"修订失败：{e}")
    with col3:
        if st.button("← 返回修改原命题"):
            st.session_state.stage = "input"
            st.rerun()

# ============ 阶段 3：执行 ============
elif st.session_state.stage == "run":
    cr = st.session_state.clarify_result
    revised = st.session_state.revised_claim
    thscode = cr.get("thscode", "")
    target_name = cr.get("target_name", "")
    uc = st.session_state.user_clarifications
    uc_text = "；".join([f"{k}：{v}" for k, v in uc.items() if v])

    logs = []
    def log(msg):
        logs.append(f"{time.strftime('%H:%M:%S')} {msg}")
        log_box.code("\n".join(logs))

    with st.spinner("Agent 正在执行..."):
        try:
            t0 = time.time()
            log("① 拆解子问题...")
            plan = llm_decompose(revised, thscode, target_name, uc_text)
            st.session_state.plan = plan

            log("② 调用扶摇接口...")
            data_bundle = {}
            errors = []
            try:
                data_bundle["snapshot"] = TOOLS["get_snapshot"]["fn"](thscode)
                log("   ✓ 行情快照")
            except Exception as e:
                errors.append(f"行情快照失败：{e}")
                log(f"   ✗ {e}")
            try:
                data_bundle["income"] = TOOLS["get_income"]["fn"](thscode)
                log("   ✓ 利润表")
            except Exception as e:
                errors.append(f"利润表失败：{e}")
                log(f"   ✗ {e}")
            st.session_state.data_bundle = data_bundle
            st.session_state.errors = errors

            log("③ AI 组织证据链...")
            result = llm_evidence(revised, plan.get("sub_questions", []), data_bundle, uc_text)
            st.session_state.result = result
            st.session_state.elapsed = time.time() - t0

            st.session_state.history.append({
                "claim": st.session_state.original_claim,
                "revised": revised,
                "clarify_result": cr,
                "plan": plan,
                "data_bundle": data_bundle,
                "result": result,
                "errors": errors,
                "elapsed": st.session_state.elapsed,
            })

            log(f"④ 完成，耗时 {st.session_state.elapsed:.1f} 秒")
            st.session_state.stage = "done"
            st.rerun()
        except Exception as e:
            st.error(f"执行失败：{e}")
            log(f"✗ 失败：{e}")
            if st.button("← 返回修改"):
                st.session_state.stage = "clarify"
                st.rerun()

# ============ 阶段 4：结果 ============
elif st.session_state.stage == "done":
    render_original_claim(st.session_state.original_claim)
    render_result(
        st.session_state.clarify_result,
        st.session_state.revised_claim,
        st.session_state.plan,
        st.session_state.data_bundle,
        st.session_state.result,
        st.session_state.errors,
        st.session_state.elapsed,
        chart_key="chart_done",
        readonly=False,
    )

    st.divider()
    st.subheader("💬 追问")
    q = st.text_input("针对以上结论继续追问", placeholder="例如：毛利率变化主要来自哪里？")
    if st.button("提交追问"):
        if not q.strip():
            st.warning("请输入追问内容")
        else:
            with st.spinner("AI 正在回答..."):
                try:
                    context = {
                        "revised_claim": st.session_state.revised_claim,
                        "sub_questions": st.session_state.plan.get("sub_questions", []),
                        "evidence": st.session_state.result.get("evidence", []),
                        "conclusion": st.session_state.result.get("conclusion", {}),
                        "data": st.session_state.data_bundle,
                    }
                    answer = llm_followup(q, context)
                    st.session_state.followup_history.append({"q": q, "a": answer})
                except Exception as e:
                    st.error(f"回答失败：{e}")

    for item in st.session_state.followup_history:
        st.markdown(f"**Q：** {item['q']}")
        st.markdown(f"**A：** {item['a']}")

    st.divider()
    if st.button("🔄 开始新的研究"):
        for k in ["original_claim", "revised_claim"]:
            st.session_state[k] = ""
        for k in ["clarify_result", "plan", "data_bundle", "result"]:
            st.session_state[k] = None
        st.session_state.errors = []
        st.session_state.elapsed = 0
        st.session_state.user_clarifications = {}
        st.session_state.followup_history = []
        st.session_state.stage = "input"
        st.rerun()

# ============ 阶段 5：回看 ============
elif st.session_state.stage == "review":
    idx = st.session_state.review_index
    if idx is None or idx >= len(st.session_state.history):
        st.warning("找不到该历史任务")
        if st.button("← 返回"):
            st.session_state.stage = "input"
            st.rerun()
    else:
        h = st.session_state.history[idx]
        render_original_claim(h["claim"], label="📝 原始命题（历史回看）")
        render_result(
            h["clarify_result"], h["revised"], h["plan"],
            h["data_bundle"], h["result"], h.get("errors", []),
            h.get("elapsed", 0),
            chart_key=f"chart_review_{idx}",
            readonly=True,
        )
        if st.button("← 返回"):
            st.session_state.review_index = None
            st.session_state.stage = "done" if st.session_state.result else "input"
            st.rerun()