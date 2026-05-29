import unittest

from src.utils.collision_time_config import get_default_collision_interp_dt


class CollisionTimeConfigTest(unittest.TestCase):
    def test_default_collision_interp_dt_is_half_second(self):
        self.assertEqual(get_default_collision_interp_dt(), 0.5)


if __name__ == "__main__":
    unittest.main()
