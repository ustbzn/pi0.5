import unittest

from src.utils.bev_axes import get_bev_axis_config


class BevAxesConfigTest(unittest.TestCase):
    def test_bev_axis_config_uses_left_positive_horizontal_axis(self):
        config = get_bev_axis_config()
        self.assertEqual(config["xlabel"], "Y (left positive)")
        self.assertEqual(config["ylabel"], "X (up positive)")
        self.assertTrue(config["invert_x"])


if __name__ == "__main__":
    unittest.main()
