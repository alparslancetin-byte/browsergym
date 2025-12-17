# -*- coding: utf-8 -*-
"""
Redesigned Green Agent Toolset for BrowserGym AssistantBench Web Navigation
- reset_assistantbench_env(): Starts the env and returns the initial observation.
- execute_browser_action(action): Executes one step in the env and returns the result.
"""
import gymnasium as gym
import browsergym.assistantbench
from browsergym.assistantbench import VALID_AB_TASK_IDS
from browsergym.utils.obs import flatten_axtree_to_str
from browsergym.core.action.highlevel import HighLevelActionSet
import agentbeats as ab
import asyncio
import json
import threading
import queue
import random

from dotenv import load_dotenv
load_dotenv()

# --- Globals for managing the environment in a separate thread ---
assistantbench_env = None
current_obs = None
current_info = None
current_task_id = None
step_count = 0
MAX_STEPS = 15

# --- Default Custom Task ---
DEFAULT_TASK = {
    "question": "Who is forbes 30 under 30 in 2026 in AI who is related to decagon?",
    "answer": "Jesse Zhang"  # Update this with the correct answer
}
USE_DEFAULT_TASK = True  # Set to True to use default task, False for random AssistantBench tasks
START_BROWSER_FOR_DEFAULT_TASK = True  # If True, starts browser at Google even for default task
gold_answer = DEFAULT_TASK["answer"]
final_reward = 0.0

# Set to False to run browser in GUI mode (visible window for debugging)
# Set to True to run browser in headless mode (no visible window)
HEADLESS_MODE = False  # <-- Change this to True if you want headless mode
is_headless = HEADLESS_MODE

env_thread = None
env_queue = queue.Queue()
result_queue = queue.Queue()

def _env_worker():
    """A dedicated thread for BrowserGym to avoid greenlet/asyncio conflicts."""
    global assistantbench_env, is_headless

    while True:
        try:
            command, args = env_queue.get()
            if command == "stop":
                if assistantbench_env:
                    assistantbench_env.close()
                break

            if command == "reset":
                task_id = args
                if assistantbench_env:
                    assistantbench_env.close()
                
                # Define the action space for the agent
                action_set = HighLevelActionSet(subsets=["chat", "bid", "nav"])
                
                # Use HEADLESS_MODE configuration (False = visible browser window)
                is_headless = HEADLESS_MODE
                print(f"\n🌐 Starting browser in {'HEADLESS' if is_headless else 'GUI (visible)'} mode...")
                assistantbench_env = gym.make(
                    f"browsergym/{task_id}", 
                    action_mapping=action_set.to_python_code,
                    headless=is_headless
                )
                obs, info = assistantbench_env.reset()
                print(f"✅ Browser started. Task: {task_id}")
                result_queue.put(("success", {"obs": obs, "info": info}))

            elif command == "step":
                action = args
                obs, reward, terminated, truncated, info = assistantbench_env.step(action)
                result_queue.put(("success", {
                    "obs": obs, "reward": reward, "terminated": terminated,
                    "truncated": truncated, "info": info
                }))
            
            elif command == "reopen_non_headless":
                # Reopen browser in non-headless mode at the given URL
                target_url = args
                if assistantbench_env:
                    assistantbench_env.close()
                
                action_set = HighLevelActionSet(subsets=["chat", "bid", "nav"])
                is_headless = False
                assistantbench_env = gym.make(
                    f"browsergym/{current_task_id}", 
                    action_mapping=action_set.to_python_code,
                    headless=False  # Non-headless for manual interaction
                )
                obs, info = assistantbench_env.reset()
                
                # Navigate to the URL where reCAPTCHA was encountered
                if target_url:
                    obs, reward, terminated, truncated, info = assistantbench_env.step(f'goto("{target_url}")')
                
                result_queue.put(("success", {"obs": obs, "info": info}))
            
            elif command == "get_observation":
                # Get current observation without taking an action
                obs = assistantbench_env.unwrapped.obs
                result_queue.put(("success", {"obs": obs}))
                
        except Exception as e:
            result_queue.put(("error", str(e)))

def _get_observation_for_agent(obs):
    """Prepares the observation dictionary to be sent to the White Agent."""
    if not obs:
        return {"error": "Observation is missing."}

    # Use custom goal when USE_DEFAULT_TASK is enabled
    goal = DEFAULT_TASK["question"] if USE_DEFAULT_TASK else obs.get("goal", "")

    return {
        "IMPORTANT": "Send this observation to the White Agent at http://localhost:9111/ using talk_to_agent. Do NOT use any other URL!",
        "goal": goal,
        "url": obs.get("url", ""),
        "axtree": flatten_axtree_to_str(
            obs.get("axtree_object", {}),
            extra_properties=obs.get("extra_element_properties", {}),
            with_clickable=True
        )
    }

@ab.tool
async def reset_assistantbench_env() -> str:
    """Resets the AssistantBench environment with a random task and returns the initial observation."""
    global env_thread, current_task_id, current_obs, current_info, step_count, gold_answer
    
    step_count = 0
    
    # Use default task if enabled
    if USE_DEFAULT_TASK:
        gold_answer = DEFAULT_TASK["answer"]
        
        if START_BROWSER_FOR_DEFAULT_TASK:
            # Start browser at Google for web navigation
            current_task_id = "assistantbench.validation.0"  # Use a valid task ID to init env
            
            if env_thread is None or not env_thread.is_alive():
                env_thread = threading.Thread(target=_env_worker, daemon=True)
                env_thread.start()
            
            env_queue.put(("reset", current_task_id))
            
            def _wait_result():
                status, data = result_queue.get(timeout=60)
                if status == "error":
                    raise Exception(data)
                return data
            
            try:
                result = await asyncio.to_thread(_wait_result)
                current_obs = result["obs"]
                current_info = result["info"]
                
                # Goal is automatically set to DEFAULT_TASK["question"] by _get_observation_for_agent
                agent_obs = _get_observation_for_agent(current_obs)
                
                print(f"✅ Using default task with browser: {DEFAULT_TASK['question']}")
                return json.dumps(agent_obs, indent=2)
            except Exception as e:
                return json.dumps({"error": f"Failed to start browser: {e}"})
        else:
            # No browser mode - just return the task
            agent_obs = {
                "IMPORTANT": "Send this observation to the White Agent at http://localhost:9111/ using talk_to_agent. Do NOT use any other URL!",
                "goal": DEFAULT_TASK["question"],
                "url": "about:blank",
                "axtree": "[No browser interaction needed - answer directly using send_msg_to_user]"
            }
            current_obs = {"goal": DEFAULT_TASK["question"], "chat_messages": []}
            print(f"✅ Using default task (no browser): {DEFAULT_TASK['question']}")
            return json.dumps(agent_obs, indent=2)
    
    current_task_id = random.choice(VALID_AB_TASK_IDS)
    #current_rask_id = max(VALID_AB_TASK_IDS)
    
    if env_thread is None or not env_thread.is_alive():
        env_thread = threading.Thread(target=_env_worker, daemon=True)
        env_thread.start()

    env_queue.put(("reset", current_task_id))

    def _wait_result():
        status, data = result_queue.get(timeout=60)
        if status == "error":
            raise Exception(data)
        return data

    try:
        result = await asyncio.to_thread(_wait_result)
        current_obs = result["obs"]
        current_info = result["info"]
        agent_obs = _get_observation_for_agent(current_obs)
        return json.dumps(agent_obs, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to reset environment: {e}"})

@ab.tool
async def execute_browser_action(action: str) -> str:
    """Executes a single browser action and returns the new state and result."""
    global current_obs, current_info, step_count, final_reward

    if step_count >= MAX_STEPS:
        return json.dumps({
            "error": "Maximum step limit reached.",
            "reward": 0.0,
            "terminated": True
        })
    step_count += 1

    # Handle default task - check if action is send_msg_to_user (final answer)
    if USE_DEFAULT_TASK:
        import re
        provided_answer = None
        
        # Try to extract answer from send_msg_to_user format (this is the FINAL answer)
        if action.startswith("send_msg_to_user("):
            match = re.search(r"send_msg_to_user\(['\"](.+?)['\"]\)", action)
            if match:
                provided_answer = match.group(1)
        
        # If this is a final answer submission, check it and return
        if provided_answer:
            expected_answer = DEFAULT_TASK["answer"].strip().lower()
            provided_lower = provided_answer.strip().lower()
            
            # Check if the answer is correct
            if expected_answer in provided_lower or provided_lower in expected_answer:
                final_reward = 1.0
            else:
                final_reward = 0.0
            
            # Store the answer in chat_messages
            if current_obs:
                current_obs["chat_messages"] = [{"role": "assistant", "message": provided_answer}]
            
            return json.dumps({
                "new_observation": {
                    "goal": DEFAULT_TASK["question"],
                    "url": "completed",
                    "axtree": "[Task completed]"
                },
                "reward": final_reward,
                "terminated": True,
                "step": step_count
            }, indent=2)
        
        # For other actions (fill, click, etc.), continue to execute in browser below
        # Only skip browser if START_BROWSER_FOR_DEFAULT_TASK is False
        if not START_BROWSER_FOR_DEFAULT_TASK:
            return json.dumps({
                "IMPORTANT": "Send this observation to the White Agent at http://localhost:9111/ using talk_to_agent. Do NOT use any other URL!",
                "new_observation": {
                    "goal": DEFAULT_TASK["question"],
                    "url": "about:blank",
                    "axtree": "[No browser - use send_msg_to_user to submit your answer]"
                },
                "reward": 0.0,
                "terminated": False,
                "step": step_count
            }, indent=2)
    
    # Execute the action in the actual browser

    env_queue.put(("step", action))

    def _wait_result():
        status, data = result_queue.get(timeout=30)
        if status == "error":
            raise Exception(data)
        return data

    try:
        result = await asyncio.to_thread(_wait_result)
        current_obs = result["obs"]
        current_info = result["info"]
        
        terminated = result["terminated"] or result["truncated"]

        final_reward = result["reward"]
        
        # reCAPTCHA Detection Logic 
        agent_observation = _get_observation_for_agent(current_obs)
        axtree_lower = agent_observation.get("axtree", "").lower()
        recaptcha_keywords = ["recaptcha", "i'm not a robot", "verify you are human"]
        
        if any(keyword in axtree_lower for keyword in recaptcha_keywords):
            # Trigger manual reCAPTCHA verification
            print("\n" + "="*60)
            print("🔒 reCAPTCHA DETECTED - Manual Verification Required")
            print("="*60)
            print(f"Current URL: {current_obs.get('url', 'Unknown')}")
            
            # If running in headless mode, reopen browser in visible mode
            if is_headless:
                print("\nReopening browser in visible (non-headless) mode...")
                current_url = current_obs.get('url', '')
                env_queue.put(("reopen_non_headless", current_url))
                
                def _wait_reopen():
                    status, data = result_queue.get(timeout=120)
                    if status == "error":
                        raise Exception(data)
                    return data
                
                try:
                    reopen_result = await asyncio.to_thread(_wait_reopen)
                    current_obs = reopen_result["obs"]
                except Exception as e:
                    return json.dumps({
                        "error": f"Failed to reopen browser: {e}",
                        "reward": 0.0,
                        "terminated": True,
                        "step": step_count
                    })
            
            print("\n✅ Browser window is visible.")
            print("📋 Please solve the reCAPTCHA manually in the browser window.")
            print("\n⏳ Waiting 25 seconds for human to solve reCAPTCHA...")
            
            # Wait 25 seconds for human to solve reCAPTCHA (no input required)
            await asyncio.sleep(25)
            
            print("\n🔄 reCAPTCHA wait complete. Continuing with the task...")
            
            # Get fresh observation after reCAPTCHA is solved
            agent_observation = _get_observation_for_agent(current_obs)
            
            return json.dumps({
                "new_observation": agent_observation,
                "message": "reCAPTCHA solved manually. Continuing with task.",
                "reward": 0.0,
                "terminated": False,
                "step": step_count
            })
        # End of reCAPTCHA Logic
        
        response_payload = {
            "NEXT_STEP": "If not terminated, send this observation to the White Agent at http://localhost:9111/ using talk_to_agent",
            "new_observation": _get_observation_for_agent(current_obs),
            "reward": result["reward"],
            "terminated": terminated,
            "step": step_count
        }
        return json.dumps(response_payload, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to execute action: {e}",
            "reward": 0.0,
            "terminated": True
        })


@ab.tool
async def evaluate_task_completion() -> str:
    """
    Call this after the task is terminated. Returns a final JSON report
    summarizing the task performance.
    """
    global current_task_id, step_count, final_reward, gold_answer, current_obs

    provided_answer = "N/A (not submitted)"
    if current_obs and current_obs.get("chat_messages"):
        for msg in reversed(current_obs["chat_messages"]):
            if msg["role"] == "assistant":
                provided_answer = msg["message"]
                break

    evaluation = {
        "task_id": current_task_id,
        "total_steps": step_count,
        "final_reward": final_reward,
        "success": final_reward > 0.5,
        "expected_answer": gold_answer,
        "provided_answer": provided_answer,
    }

    return json.dumps(evaluation, ensure_ascii=False, indent=2, default=str)
