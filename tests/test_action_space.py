import unittest

from config.train_config import get_train_config
from src.agent.agent import McRlAgent
from src.common.action_space import get_all_action_ids, get_all_attack_action_ids, get_random_action_dist

train_config: dict = get_train_config()


class TestActionSpace(unittest.TestCase):

    def test_random_action_distribution(self):
        epochs = 100
        steps = train_config["steps_per_epoch"]

        n = 1000000
        all_action_ids = get_all_action_ids()
        rel_frequency, abs_frequency, actions = get_random_action_dist(all_action_ids, n)

        attack_action_ids = get_all_attack_action_ids()
        attack_action_sequence_counter = 0
        seq_len = 6
        for i in range(len(actions) - seq_len + 1):
            tmp = actions[i:(i + seq_len)]

            is_action_seq = True
            for a in tmp:
                if a not in attack_action_ids:
                    is_action_seq = False
                    break
            if is_action_seq:
                attack_action_sequence_counter += 1
        attack_action_sequences_in_agent_epochs = ((epochs*steps)/n) * attack_action_sequence_counter

        print("test_random_action_distribution")
        print("n: " + str(n))
        print(abs_frequency)
        print(rel_frequency)
        probability_attack_action = 0
        for attack_action in attack_action_ids:
            probability_attack_action += rel_frequency[str(attack_action)]
        print("probability attack action: " + str(probability_attack_action))
        print("attack action sequences: " + str(attack_action_sequence_counter))
        print("mean attack action sequences each agent: " + str(attack_action_sequences_in_agent_epochs))

