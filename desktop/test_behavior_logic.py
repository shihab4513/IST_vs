import unittest

from desktop.behavior_logic import AlertEngine


class AlertEngineTests(unittest.TestCase):
    def test_alert_requires_sustained_behavior(self):
        engine = AlertEngine(window_size=4, required_frames=3)
        self.assertFalse(engine.update("Sleeping"))
        self.assertFalse(engine.update("Sleeping"))
        self.assertTrue(engine.update("Sleeping"))

    def test_normal_state_clears_after_safe_frames(self):
        engine = AlertEngine(window_size=3, required_frames=2)
        engine.update("Using Phone")
        engine.update("Using Phone")
        engine.update("Normal Focused")
        engine.update("Normal Focused")
        self.assertEqual(engine.active_behavior, "Normal Focused")

    def test_mixed_unsafe_behaviors_do_not_trigger_one_alert(self):
        engine = AlertEngine(window_size=4, required_frames=3)
        engine.update("Sleeping")
        engine.update("Using Phone")
        engine.update("Sleeping")
        engine.update("Using Phone")
        self.assertFalse(engine.update("Sleeping"))


if __name__ == "__main__":
    unittest.main()
