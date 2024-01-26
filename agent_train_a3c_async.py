from src.agent.run_mc_agent_epochs import run_mc_agent
import os
if __name__ == '__main__':
    old_config_file = "./used-config/train_config.json"
    if os.path.exists(old_config_file):
        os.remove(old_config_file)
    run_mc_agent()
