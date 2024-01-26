import time

from src.agent.actionExecution import ActionExecution
from src.common.countdown import countdown

action_exec = ActionExecution()
countdown(5, optional_text="Agent starting. Please focus Minecraft on screen")
timestep_length_sec = 0.33
for i in range(10):
    action_exec.execute_timestep(action_id=0, timestep_length_sec=timestep_length_sec)
    action_exec.release_all_keys()
    time.sleep(timestep_length_sec)
    action_exec.execute_timestep(action_id=2, timestep_length_sec=timestep_length_sec)
    action_exec.release_all_keys()
    time.sleep(timestep_length_sec)
