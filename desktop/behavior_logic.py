from collections import deque


UNSAFE_BEHAVIORS = {
    "Drinking",
    "Eating",
    "Normal Non-Focused",
    "Sleeping",
    "Using Phone",
    "Yawning/Drowsiness",
}


class AlertEngine:
    def __init__(self, window_size=12, required_frames=7):
        self.window = deque(maxlen=window_size)
        self.required_frames = required_frames
        self.active_behavior = "Normal Focused"
        self.active_count = 0
        self.safe_count = 0

    def update(self, behavior):
        self.window.append(behavior)
        if behavior in UNSAFE_BEHAVIORS:
            self.safe_count = 0
        else:
            self.safe_count += 1
            if self.safe_count >= self.required_frames:
                self.active_behavior = "Normal Focused"
                self.active_count = 0
        matching_behavior_count = sum(1 for item in self.window if item == behavior)
        if behavior in UNSAFE_BEHAVIORS and matching_behavior_count >= self.required_frames:
            self.active_behavior = behavior
            self.active_count = matching_behavior_count
            return True
        return self.active_behavior != "Normal Focused"

    def reset(self):
        self.window.clear()
        self.active_behavior = "Normal Focused"
        self.active_count = 0
        self.safe_count = 0
