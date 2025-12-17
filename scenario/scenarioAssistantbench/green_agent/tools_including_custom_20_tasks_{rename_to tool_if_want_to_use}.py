# -*- coding: utf-8 -*-
"""
Self-contained custom benchmark implementation for AgentBeats.
This file defines, registers, and provides tools for a custom set of BrowserGym tasks.
"""
from functools import partial
import gymnasium as gym
from browsergym.core.task import AbstractBrowserTask
from browsergym.core.registration import register_task
from browsergym.utils.obs import flatten_axtree_to_str
from browsergym.core.action.highlevel import HighLevelActionSet
import playwright.sync_api
import agentbeats as ab
import asyncio
import json
import threading
import queue
import random
from typing import Tuple
import traceback

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# 1. CUSTOM TASK DEFINITION
# ============================================================================

# --- You can reuse the AssistantBench scorer for intelligent evaluation ---
try:
    from browsergym.assistantbench.evaluation.evaluator import question_scorer
    USE_ASSISTANTBENCH_SCORER = True
except ImportError:
    USE_ASSISTANTBENCH_SCORER = False
    print("Warning: browsergym[assistantbench] not installed. Using a simple string match for evaluation.")

class MyCustomTask(AbstractBrowserTask):
    """A custom task that can be configured with different goals and answers."""
      
    def __init__(self, seed: int, start_url: str, goal_text: str, gold_answer: str, **kwargs):
        super().__init__(seed)  
        self.start_url = start_url  
        self.goal = goal_text
        self.gold_answer = gold_answer
      
    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:  
        page.goto(self.start_url, timeout=10000)  
        return self.goal, {}  
      
    def validate(self, page: playwright.sync_api.Page, chat_messages: list[dict]) -> Tuple[float, bool, str, dict]:  
        reward, done, msg, info = 0.0, False, "", {}  
          
        if chat_messages and chat_messages[-1]["role"] == "assistant":  
            done = True  
            prediction = chat_messages[-1]["message"]  
              
            # Use the powerful AssistantBench scorer if available, otherwise fall back to a simple check.
            if USE_ASSISTANTBENCH_SCORER:
                accuracy, _ = question_scorer(prediction, self.gold_answer)
                reward = accuracy
            else:
                # Fallback simple evaluation
                if self.gold_answer.lower() in prediction.lower():
                    reward = 1.0
            
            info["success"] = reward > 0.5
            info["final_answer"] = prediction
              
        return reward, done, msg, info  
      
    def teardown(self) -> None:  
        pass

# ============================================================================
# 2. TASK REGISTRATION
# ============================================================================

# --- Define your custom tasks here ---
# You can add as many as you want.
# ============================================================================
# 2. TASK REGISTRATION
# ============================================================================

# --- Define your custom tasks here ---
# You can add as many as you want.
MY_TASKS_DATA = [
    {
        "id": "custom_task_bremen_tram",
        "goal": "What is the smallest number of the tram line (excluding zero) that does not exist in the german city, which is known for the tale of its town musician?",
        "answer": "7"
    },
    {
        "id": "custom_task_bremen_elevation",
        "goal": "How high is the natural highest elavation in the smallest state of Germany?",
        "answer": "32.5 meters"
    },
    {
        "id": "custom_task_aachen_university",
        "goal": "Which famous technical university in germany is located very close to belgium and netherlands?",
        "answer": "RWTH Aachen"
    },
    {
        "id": "custom_task_physics_fellow",
        "goal": "Who is the fellow of the foundation in theoretical physics in 2022 who was born in hungary, from the foundation of the founder of the medaillon found?",
        "answer": "Csaba Csáki"
    },
    {
        "id": "custom_task_berkeley_pizzeria",
        "goal": "In which city is the pizzeria who is named after the edible thistle with a tender heart and which has over 1000 reviews",
        "answer": "Berkeley"
    },
    {
        "id": "custom_task_ai_figure",
        "goal": "Who is the person who was named top 100 most influential figures in AI in 2023 and received a ERP scholarship",
        "answer": "Richard Socher"
    },
    {
        "id": "custom_task_climbing_home",
        "goal": "Which rock climbing side in the landlocked country in south-east-asia burned down at least two times?",
        "answer": "Green Climbers Home"
    },
    {
        "id": "custom_task_universum_cost",
        "goal": "How much do I pay, when I go to Universum Bremen every Friday on the start of overy month and twice on Tuesdays in the year 2025, converted to exchange rate in USD on the 11/03/2025",
        "answer": "27.66$"
    },
    {
        "id": "custom_task_movie_remake",
        "goal": "Which american remake of a famous south korean movie got way worse rating on IMDB than the original movie",
        "answer": "Oldboy"
    },
    {
        "id": "custom_task_geico_ceo",
        "goal": "Which insurance is advertised on the website of the Company which Greg Abel is going to be CEO from 2026 onwards?",
        "answer": "Geico"
    },
    {
        "id": "custom_task_free_solo_climber",
        "goal": "Find the third person who free soloed the route x. The first person who soloed route x died in an car accident on A9",
        "answer": "Dean Potter"
    },
    {
        "id": "custom_task_berkeley_student_housing",
        "goal": "What is a cheap possibilitiy for students to live in Berkeley and the name of the house is called after an animal?",
        "answer": "Wolf House"
    },
    {
        "id": "custom_task_climbing_route_rainshadow",
        "goal": "Which is the hardest climbing route, which the 2024 climbing olympics (male) winner redpoint in 2020",
        "answer": "Rainshadow"
    },
    {
        "id": "custom_task_berkeley_bouldering",
        "goal": "What is the cheapest option for Berkeley students with intermediate climbing experience to go bouldering within 1.5 mile radius of the campus?",
        "answer": "Mosaic Boulders"
    },
    {
        "id": "custom_task_turkish_billionaire_stock",
        "goal": "How much was the stock price of the company of the first turkish immigrant billionaire valued in January 2025, rounded down to the next 10$ dollar steps.",
        "answer": "110$"
    },
    {
        "id": "custom_task_bremen_glasses",
        "goal": "Where can I get new glasses after I have watched a movie in Schauburg Bremen?",
        "answer": "Frenz (fürs Auge Brillen und Kontaktlinsen)"
    }
]

CUSTOM_TASK_IDS = []
for task_data in MY_TASKS_DATA:
    task_id = f"my_benchmark/{task_data['id']}"
    task_id = f"my_benchmark-{task_data['id']}-v0"
    register_task(
        id=task_id,
        task_class=partial(
            MyCustomTask,
            start_url="https://www.google.com",
            goal_text=task_data['goal'],
            gold_answer=task_data['answer']
        ),
    )
    CUSTOM_TASK_IDS.append(task_id)

print(f"✅ Registered {len(CUSTOM_TASK_IDS)} custom tasks.")
# ============================================================================
# 3. ENVIRONMENT MANAGEMENT & AGENT TOOLS
# ============================================================================

# Globals
custom_env = None  
current_obs = None  
current_info = None  
current_task_id = None  
step_count = 0  
final_reward = 0.0 # Correctly store the final reward
MAX_STEPS = 15  
env_thread = None  
env_queue = queue.Queue()  
result_queue = queue.Queue()  

def _env_worker():  
    global custom_env
    # ... (This function remains the same as your proposal)
    while True:
        try:
            command, args = env_queue.get()
            if command == "stop":
                if custom_env: custom_env.close()
                break
            if command == "reset":
                task_id = args
                if custom_env: custom_env.close()
                action_set = HighLevelActionSet(subsets=["chat", "bid", "nav"])
                # IMPORTANT: gym.make needs the "browsergym/" prefix
                custom_env = gym.make(f"browsergym/{task_id}", action_mapping=action_set.to_python_code)
                obs, info = custom_env.reset()
                result_queue.put(("success", {"obs": obs, "info": info}))
            elif command == "step":
                action = args
                obs, reward, terminated, truncated, info = custom_env.step(action)
                result_queue.put(("success", {"obs": obs, "reward": reward, "terminated": terminated, "truncated": truncated, "info": info}))
        except Exception as e:
            result_queue.put(("error", f"{e}\n{traceback.format_exc()}"))

def _get_observation_for_agent(obs):  
    if not obs: return {"error": "Observation is missing."}  
    return { "goal": obs.get("goal", ""), "url": obs.get("url", ""), "axtree": flatten_axtree_to_str(obs.get("axtree_object", {}), extra_properties=obs.get("extra_element_properties", {}), with_clickable=True)}  

@ab.tool  
async def reset_env() -> str:  
    """Resets the environment with a random custom task and returns the initial observation."""  
    global env_thread, current_task_id, current_obs, current_info, step_count, final_reward
    step_count = 0
    final_reward = 0.0
    current_task_id = random.choice(CUSTOM_TASK_IDS)  
    if env_thread is None or not env_thread.is_alive():  
        env_thread = threading.Thread(target=_env_worker, daemon=True)  
        env_thread.start()  
    env_queue.put(("reset", current_task_id))  
    def _wait_result():  
        status, data = result_queue.get(timeout=60)  
        if status == "error": raise Exception(data)  
        return data  
    try:  
        result = await asyncio.to_thread(_wait_result)  
        current_obs, current_info = result["obs"], result["info"]
        return json.dumps(_get_observation_for_agent(current_obs), indent=2)  
    except Exception as e:  
        return json.dumps({"error": f"Failed to reset environment: {e}"})  

@ab.tool  
async def execute_browser_action(action: str) -> str:  
    """Executes a single browser action and returns the new state and result."""  
    global current_obs, current_info, step_count, final_reward
    if step_count >= MAX_STEPS:  
        final_reward = 0.0
        return json.dumps({"error": "Maximum step limit reached.", "reward": final_reward, "terminated": True})  
    step_count += 1  
    env_queue.put(("step", action))  
    def _wait_result():  
        status, data = result_queue.get(timeout=30)  
        if status == "error": raise Exception(data)  
        return data  
    try:  
        result = await asyncio.to_thread(_wait_result)  
        current_obs, current_info = result["obs"], result["info"]
        terminated = result["terminated"] or result["truncated"]  
        final_reward = result["reward"] # Update the global final_reward
        agent_observation = _get_observation_for_agent(current_obs)
          
        if any(keyword in agent_observation.get("axtree", "").lower() for keyword in ["recaptcha", "i'm not a robot"]):  
            final_reward = 0.0
            return json.dumps({"new_observation": agent_observation, "error": "reCAPTCHA detected.", "reward": final_reward, "terminated": True})
          
        return json.dumps({"new_observation": agent_observation, "reward": final_reward, "terminated": terminated}, indent=2)  
    except Exception as e:  
        final_reward = 0.0
        return json.dumps({"error": f"Failed to execute action: {e}", "reward": final_reward, "terminated": True})  

@ab.tool  
async def evaluate_task_completion() -> str:  
    """Call this after the task is terminated. Returns a final JSON report."""  
    global current_task_id, step_count, final_reward, current_obs
    provided_answer = "N/A (not submitted)"
    if current_obs and current_obs.get("chat_messages"):  
        for msg in reversed(current_obs.get("chat_messages", [])):  
            if msg["role"] == "assistant":  
                provided_answer = msg["message"]  
                break  
    evaluation = {  
        "task_id": current_task_id,  
        "total_steps": step_count,  
        "final_reward": final_reward,  
        "success": final_reward > 0.5,  
        "provided_answer": provided_answer,  
    }  
    return json.dumps(evaluation, ensure_ascii=False, indent=2)