import agentbeats as ab
import tools  # 必须 import 才能注册工具

if __name__ == "__main__":
    # Load agent card
    ab.load_agent_card("green_agent_card.toml")
    
    # Start MCP-based agent for cloud controller
    ab.start_green_agent()
