from src.agent.actionExecution import ActionExecution
from src.common.countdown import countdown

act_exec: ActionExecution = ActionExecution()
countdown(5, "Focus window of running Minecraft world and release keyboard and mouse until new world is created")
act_exec.agent_create_new_world_if_done()


