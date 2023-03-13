from agent import Agent

BASE_POSITION = 0

def check_if_reward(map: list) -> int:
    max_reward = max(map)
    if max_reward != 0:
        for i in range(len(map)):
            if map[i] == max_reward:
                return i
    return -1

def l_o_r(actual : int, obj: int) -> str:
    if actual == obj:
        return "none"
    elif obj > actual:
        return "right"
    else:
        return "left"

def ekip_policy(agent: Agent) -> str:
    actions = ["left", "right", "none"]
    action = ""

    if agent.position == 0:
        BASE_POSITION = 7
    else:
        BASE_POSITION = 0

    pos = check_if_reward(agent.known_rewards)

    if pos != -1:
        action = l_o_r(agent.position, pos)
    else:
        action = l_o_r(agent.position, BASE_POSITION)
    
    assert action in actions
    return action