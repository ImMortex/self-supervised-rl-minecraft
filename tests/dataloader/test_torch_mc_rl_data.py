import unittest

from torch.utils.data import DataLoader

from src.agent.observation.agent_make_screenshot import agent_make_screenshot
from src.common.dummy_transitions import get_dummy_transition_seq
from src.common.persisted_memory import PersistedMemory, TIMESTEP_NAME_PREFIX
from src.dataloader.torch_mc_rl_data import HardDiskCustomDataset, AgentCustomDataset


class TestMcRlData(unittest.TestCase):

    def test_get_data_sequence_paths(self):
        test_storage_dir = "./tests/test_storage"  # Assume timestep 11 is missing (delete it)
        transition_abs_paths_grouped_agent, session_id, generation_id, agent_id = PersistedMemory.get_timestep_paths_from_filesystem_by_filter(
            local_filesystem_store_root_dir=test_storage_dir, session_id="2023.08.26.00.00.00", generation_id=None,
            agent_id=None,
            group_paths=True)

        self.assertEqual(219, len(transition_abs_paths_grouped_agent[0]))

        x_depth = 10
        dataset: HardDiskCustomDataset = HardDiskCustomDataset(transition_abs_paths_grouped_agent, x_depth=x_depth,
                                                               width_2d=640, height_2d=640)

        self.assertEqual(200, dataset.__len__())
        item_first = dataset.__getitem__(0)

        x = item_first
        self.assertEqual(x_depth, x["tensor_image"].shape[3], "first 3d x depth")
        self.assertEqual(x_depth, x["tensor_state_seq"].shape[0], "first state_seq x depth")

        last_id = dataset.length - 1
        self.assertEqual(199, last_id, "last idx of dataset")
        item_last = dataset.__getitem__(last_id)
        x = item_last
        self.assertEqual(x_depth, x["tensor_image"].shape[3], "last sequence x depth")
        self.assertEqual(x_depth, x["tensor_state_seq"].shape[0], "last state_seq x depth")

    def test_load_transitions_from_list(self):
        x_depth = 10
        transition_seq = get_dummy_transition_seq(x_depth, 2, screenshot = agent_make_screenshot())

        dataset: AgentCustomDataset = AgentCustomDataset(transition_seq=transition_seq, x_depth=x_depth,
                                                               width_2d=640, height_2d=640)

        item_first = dataset.__getitem__(0)
        x = item_first

        self.assertEqual(x_depth, x["tensor_image"].shape[3], "first 3d x depth")
        self.assertEqual(x_depth, x["tensor_state_seq"].shape[0], "first state_seq x depth")

        train_loader = DataLoader(dataset, batch_size=1,
                                  shuffle=False)

        batch = None
        for step, b in enumerate(train_loader):
            batch = b
            break

        self.assertIsNotNone(batch)
