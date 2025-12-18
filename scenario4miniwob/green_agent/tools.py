# -*- coding: utf-8 -*-
"""
BrowserGym MiniWob Green Agent Toolset (dynamic, non-hardcoded)
- reset_miniwob_env(task_id): create env and store latest obs/info
- get_miniwob_task(): parse obs and return a detailed, actionable task description
  (includes options list, target, and suggested Playwright action template)
- evaluate_miniwob_result(agent_actions): execute white agent-provided Playwright actions
  in the MiniWob env, return structured evaluation (reward, terminated, details)

Notes:
- This implementation parses `obs['goal']` and `obs['axtree_object']` (nodes)
  to extract option texts and button name.
- It generates Playwright action templates but does not hardcode option lists:
  it reads them directly from `obs`.
- Execution of actions uses `env.step(action_str)` (the BrowserGym convention).
"""

import os
import random
import gymnasium as gym
import browsergym.miniwob
import agentbeats as ab
import asyncio
import re
import json
import threading
import queue

from dotenv import load_dotenv
load_dotenv()

miniwob_env = None
current_task = None
current_obs = None
current_info = None

reward_history = []
action_execution_count = 0
MAX_ACTION_EXECUTIONS = 10

env_thread = None
env_queue = queue.Queue()
result_queue = queue.Queue()


def _env_worker():
    """thread for BrowserGym --- because the greenlet MiniWob use is not compatible with async operations in agentbeats"""
    global miniwob_env

    while True:
        try:
            command, args = env_queue.get()

            if command == "reset":
                task_id = args
                miniwob_env = gym.make(f"browsergym/miniwob.{task_id}", action_mapping=None)
                obs, info = miniwob_env.reset()
                result_queue.put(("success", {"obs": obs, "info": info}))

            elif command == "step":
                action = args
                obs, reward, terminated, truncated, info = miniwob_env.step(action)
                result_queue.put(("success", {
                    "obs": obs,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info
                }))

            elif command == "stop":
                break

        except Exception as e:
            result_queue.put(("error", str(e)))


@ab.tool
async def reset_miniwob_env(task_id: str = "click-list") -> str:
    """reset MiniWob env"""
    global env_thread, reward_history, current_task_id, current_obs, current_info, action_execution_count

    action_execution_count = 0
    reward_history = []
    current_task_id = task_id

    # 启动环境线程(如果未启动)
    if env_thread is None or not env_thread.is_alive():
        env_thread = threading.Thread(target=_env_worker, daemon=True)
        env_thread.start()

        # 发送 reset 命令
    env_queue.put(("reset", task_id))

    # 等待结果
    def _wait_result():
        status, data = result_queue.get(timeout=30)
        if status == "error":
            raise Exception(data)
        return data

    try:
        result = await asyncio.to_thread(_wait_result)

        current_obs = result["obs"]
        current_info = result["info"]

        return f"✅ Environment reset successfully for task: {task_id}"
    except Exception as e:
        return f"❌ Failed to reset: {e}"


def _extract_elements_from_axtree(node):
    elements = []
    if not node:
        return elements

    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value", "")
    children = node.get("children", [])

    if role in ["button", "link", "textbox", "combobox", "checkbox", "menuitem", "radio", "listitem"]:
        elements.append({
            "role": role,
            "name": name,
            "value": value,
            "focusable": node.get("focusable", False),
            "clickable": node.get("clickable", False),
            "disabled": node.get("disabled", False)
        })

    for child in children:
        elements.extend(_extract_elements_from_axtree(child))

    return elements


@ab.tool
async def get_task_description():

    global current_obs, current_info

    if current_obs is None:
        return json.dumps({"error": "Environment not initialized"})

    if "axtree" in current_obs and current_obs["axtree"]:
        visible_elements = _extract_elements_from_axtree(current_obs["axtree"])
    else:
        # fallback 简化：提取 DOM 中的 tag 与 id
        visible_elements = []
        if "dom" in current_obs and "children" in current_obs["dom"]:
            for child in current_obs["dom"]["children"]:
                tag = child.get("tag", "")
                text = child.get("text", "")
                visible_elements.append({"tag": tag, "text": text})

    task_summary = {
        "goal": current_obs.get("goal", current_obs.get("utterance", "")),
        "url": current_obs.get("url", ""),
        "visible_elements": visible_elements,
        "reward": current_info.get("reward", 0),
        "terminated": current_info.get("terminated", False),
        "raw_keys": list(current_obs.keys())
    }

    return json.dumps(task_summary, ensure_ascii=False, indent=2, default=str)


@ab.tool
async def execute_white_agent_action(playwright_action: str) -> str:
    # dummy test
    playwright_action = f"""page.get_by_role("button", name="Submit").click()"""

    global miniwob_env, current_obs, current_info, reward_history, action_execution_count

    # No more than 10 turns
    if action_execution_count > MAX_ACTION_EXECUTIONS:
        return json.dumps({
            "success": True,
            "terminated": True,
            "truncated": True,
            "reward": 0.0,
            "message": "Task terminated due to exceeding maximum action limit"
        })

    action_execution_count += 1

    env_queue.put(("step", playwright_action))

    def _wait_result():
        status, data = result_queue.get(timeout=30)
        if status == "error":
            raise Exception(data)
        return data

    try:
        result = await asyncio.to_thread(_wait_result)

        current_obs = result["obs"]
        current_info = result["info"]

        # dummy test
        result["reward"] = round(random.random(), 2)

        reward_history.append(result["reward"])

        return json.dumps({
            "success": True,
            "reward": result["reward"],
            "terminated": result["terminated"],
            "truncated": result["truncated"],
            "message": f"Action executed. Reward: {result['reward']}"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e),
            "message": f"Failed to execute action: {e}"
        })


@ab.tool
async def evaluate_task_completion() -> str:
    """
    Evaluate

    Returns:
        JSON result
    """
    global current_obs, current_info, reward_history

    def _sync_evaluate():
        return current_obs, current_info

    try:
        if reward_history:
            avg_reward = sum(reward_history) / len(reward_history)
            total_reward = sum(reward_history)
        else:
            avg_reward = 0.0
            total_reward = 0.0

        obs, info = await asyncio.to_thread(_sync_evaluate)

        # dummy reward
        # total_reward = random.random()

        evaluation = {
            "task_id": current_task_id,
            "goal": obs.get("goal", ""),
            # "elapsed_time": obs.get("elapsed_time", 0),
            # "last_action": obs.get("last_action", ""),
            # "last_action_error": obs.get("last_action_error", ""),
            # "num_actions": len(reward_history),
            # "reward_history": reward_history,
            "total_reward": total_reward,
            # "average_reward": avg_reward,
            # "evaluation_details": info,
        }

        return json.dumps(evaluation, ensure_ascii=False, indent=2, default=str)

    except Exception as e:
        return json.dumps({
            "error": f"Evaluation failed: {str(e)}",
            "score": 0.0,
            "success": False
        })

# ============ utils ============


# def _extract_visible_elements(obs) -> list:
#     """从 axtree_object 中提取可见元素"""
#     try:
#         axtree = obs.get("axtree_object", {})
#         nodes = axtree.get("nodes", [])
#
#         elements = []
#         for node in nodes:
#             role = node.get("role", {}).get("value", "")
#             name = node.get("name", {}).get("value", "")
#             if role and name:
#                 elements.append({"role": role, "name": name})
#
#         return elements[:20]  # 限制数量
#     except:
#         return []
#
#
# def _extract_options_from_axtree(obs) -> list:
#     """从 axtree 中提取选项列表(用于 choose-list 任务)"""
#     try:
#         axtree = obs.get("axtree_object", {})
#         nodes = axtree.get("nodes", [])
#
#         options = []
#         for node in nodes:
#             if node.get("role", {}).get("value") == "option":
#                 name = node.get("name", {}).get("value")
#                 if name:
#                     options.append(name)
#
#         return options
#     except:
#         return []
#
#
# def _extract_button_name(obs) -> str:
#     """从 axtree 中提取按钮名称"""
#     try:
#         axtree = obs.get("axtree_object", {})
#         nodes = axtree.get("nodes", [])
#
#         for node in nodes:
#             if node.get("role", {}).get("value") == "button":
#                 name = node.get("name", {}).get("value")
#                 if name:
#                     return name
#
#         return ""
#     except:
#         return ""
