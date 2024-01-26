from src.common.action_space import get_all_action_ids, get_random_action_dist


def print_dist(n):
    print(n)
    all_action_ids = get_all_action_ids()
    rel_frequency, abs_frequency, actions = get_random_action_dist(all_action_ids, n)
    print(rel_frequency)
    print("---")


for k in [100, 1000, 1000000, 10000000, 100000000]:
    print_dist(k)
