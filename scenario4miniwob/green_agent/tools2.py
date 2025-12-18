# -*- coding: utf-8 -*-
"""
BrowserGym MiniWob Green Agent Toolset (Real Version)
-----------------------------------------------------
Interacts with actual MiniWob environment for task evaluation.
"""

import os
import json
import gymnasium as gym
import browsergym.miniwob
import agentbeats as ab
from dotenv import load_dotenv

load_dotenv()

# Global environment
miniwob_env = None


@ab.tool
def reset_miniwob_env() -> str:
    """Reset the MiniWob environment."""
    global miniwob_env

    try:
        miniwob_env = gym.make("browsergym/miniwob.choose-list", render_mode="human")
        obs, info = miniwob_env.reset()
        task_description = info.get("instruction", "No instruction provided.")
        return f"Environment reset. Task: {task_description}"
    except Exception as e:
        return f"FAILED to reset MiniWob environment: {e}"


@ab.tool
def get_miniwob_task() -> str:
    """Return the task description from the current MiniWob environment."""
    global miniwob_env
    if miniwob_env is None:
        return "Environment not initialized."

    try:
        _, info = miniwob_env.reset()
        return info.get("instruction", "No instruction provided.")
    except Exception as e:
        return f"FAILED to get task: {e}"


@ab.tool
def evaluate_miniwob_result(agent_actions: str) -> str:
    """Replay agent actions and compute performance metrics."""
    global miniwob_env

    if miniwob_env is None:
        return "Environment not initialized. Please call reset_miniwob_env() first."

    try:
        actions = json.loads(agent_actions)
        total_reward = 0.0
        for action in actions:
            obs, reward, terminated, truncated, info = miniwob_env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        success = total_reward > 0
        return f"Evaluation complete. Success: {success}, Total Reward: {total_reward:.2f}"
    except Exception as e:
        return f"FAILED to evaluate result: {e}"
