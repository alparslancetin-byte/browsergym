# -*- coding: utf-8 -*-
"""
WebLINX Green Agent Toolset
Fixed Logic & Debugging
"""

import agentbeats as ab
import json
import re
import ast
import os

# Global variables
weblinx_data = None
current_task = None
task_history = []

DATASET_DIR = os.getenv(
    "WEBLINX_DATA_PATH",
    r"D:\Agentbeats\weblinx_scenario\green_agent\weblinx_data"
)


def _ast_node_to_value(node):
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.Str): return node.s
    if isinstance(node, ast.Num): return node.n
    if isinstance(node, ast.NameConstant): return node.value
    if isinstance(node, ast.Name):
        if node.id in ("True", "False", "None"): return eval(node.id)
        return node.id
    if isinstance(node, ast.List): return [_ast_node_to_value(e) for e in node.elts]
    if isinstance(node, ast.Tuple): return tuple(_ast_node_to_value(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return {_ast_node_to_value(k): _ast_node_to_value(v) for k, v in zip(node.keys, node.values)}
    try: return ast.unparse(node)
    except: return repr(node)

def parse_weblinx_action(action_str: str):
    """Robust parsing of WebLINX actions."""
    if not action_str: return None, {}
    
    # 1. 基础清洗
    action_str = str(action_str).strip()
    # 2. 关键：移除转义符，防止 \" 导致解析失败
    action_str = action_str.replace('\\"', '"').replace("\\'", "'")

    # 3. 正则提取函数名和参数部分
    match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\((.*)\)$', action_str, re.DOTALL)
    if not match: return None, {}

    func_name = match.group(1).strip().lower()
    args_str = match.group(2).strip()

    if not args_str: return func_name, {}

    try:
        # 4. 利用 AST 安全解析参数
        tree = ast.parse(f"dummy({args_str})", mode="eval")
        call_node = tree.body
        if isinstance(call_node, ast.Expression): call_node = call_node.body
        
        kwargs = {}
        for kw in call_node.keywords:
            kwargs[kw.arg] = _ast_node_to_value(kw.value)
        return func_name, kwargs
    except:
        # 解析失败降级处理
        print(f"⚠️ AST Parse failed for: {args_str}")
        return func_name, {"raw": args_str}

@ab.tool
def reset_weblinx_env(split: str = "validation") -> str:
    global weblinx_data, task_history
    path = f"{DATASET_DIR}/valid.json.gz" if split in ["validation", "valid"] else f"{DATASET_DIR}/train.json.gz"
    
    try:
        import gzip, json
        print(f"📂 Loading WebLINX from {path}")
        tasks = []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip(): tasks.append(json.loads(line))
        
        weblinx_data = tasks
        task_history = []
        print(f"✅ Loaded {len(tasks)} tasks.")
        return json.dumps({"success": True, "total_tasks": len(tasks)})
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

@ab.tool
def get_weblinx_task(task_id: int = 0) -> str:
    global weblinx_data, current_task
    if not weblinx_data or task_id >= len(weblinx_data):
        return json.dumps({"error": "Invalid task_id or dataset not loaded"})
    
    current_task = weblinx_data[task_id]
    current_task["task_id"] = task_id
    
    return json.dumps({
        "task_id": task_id,
        "utterances": current_task.get("utterances"),
        "viewport": current_task.get("viewport"),
        "candidates": current_task.get("candidates"),
        "action_history": current_task.get("action_history"),
        "expected_action": current_task.get("action")
    }, indent=2, ensure_ascii=False)

def clean_val(v):
    """Normalize values for comparison."""
    return str(v).strip().replace('\\"', '"')

@ab.tool
async def evaluate_white_agent_action(agent_action: str) -> str:
    """Evaluate White Agent's action with DEBUG logging."""
    global current_task, task_history
    if not current_task: return json.dumps({"error": "No active task"})

    expected_action_str = current_task.get("expected_action", "")
    
    # 解析
    agent_func, agent_args = parse_weblinx_action(agent_action)
    exp_func, exp_args = parse_weblinx_action(expected_action_str)

    # --- 🔍 强力调试日志 (会在终端显示) ---
    print(f"\n--- EVALUATION DEBUG Task {current_task['task_id']} ---")
    print(f"Agent Raw: {agent_action}")
    print(f"Agent Parsed: Func={agent_func}, Args={agent_args}")
    print(f"Exp   Parsed: Func={exp_func}, Args={exp_args}")
    # ----------------------------------------

    success = False
    score = 0.0
    match_type = "mismatch"

    if not agent_func:
        match_type = "parse_error"
    elif agent_func != exp_func:
        match_type = f"wrong_action_type ({agent_func} vs {exp_func})"
    else:
        # 1. CLICK / HOVER / SUBMIT
        if agent_func in ["click", "hover", "submit"]:
            a_uid = clean_val(agent_args.get("uid"))
            e_uid = clean_val(exp_args.get("uid"))
            if a_uid == e_uid:
                success = True
                score = 1.0
                match_type = "exact_match"
            else:
                match_type = f"wrong_element (Got {a_uid}, Exp {e_uid})"
        
        # 2. TEXTINPUT
        elif agent_func == "textinput":
            if clean_val(agent_args.get("uid")) == clean_val(exp_args.get("uid")):
                if clean_val(agent_args.get("text")) == clean_val(exp_args.get("text")):
                    success = True
                    score = 1.0
                    match_type = "exact_match"
                else:
                    score = 0.5
                    match_type = "wrong_text_content"
            else:
                match_type = "wrong_element"
        
        # 3. SAY
        elif agent_func == "say":
            # 兼容 utterance 和 text 字段
            msg1 = clean_val(agent_args.get("utterance") or agent_args.get("text"))
            msg2 = clean_val(exp_args.get("utterance") or exp_args.get("text"))
            # 忽略空格对比
            if "".join(msg1.split()) == "".join(msg2.split()):
                success = True
                score = 1.0
                match_type = "exact_match"
            else:
                score = 0.5
                match_type = "message_mismatch"
        
        # 4. 其他情况 (Fallback)
        else:
            if agent_args == exp_args:
                success = True
                score = 1.0
                match_type = "exact_match"

    result = {
        "task_id": current_task["task_id"],
        "expected": expected_action_str,
        "actual": agent_action,
        "success": success,
        "score": score,
        "match_type": match_type
    }
    task_history.append(result)
    
    print(f"Result: {match_type}, Score: {score}") # 终端确认
    return json.dumps({"success": True, "evaluation": result}, ensure_ascii=False)

@ab.tool
async def get_weblinx_statistics() -> str:
    global task_history
    if not task_history: return json.dumps({"message": "No data"})
    
    success_count = sum(1 for t in task_history if t["success"])
    return json.dumps({
        "total": len(task_history),
        "success_rate": round(success_count / len(task_history), 2),
        "summary": [f"T{t['task_id']}: {t['match_type']}" for t in task_history]
    }, ensure_ascii=False)