import numpy as np
from datetime import datetime

# Threat assessment logic based on drone movement + acoustic cues
class ThreatEngine:
    def __init__(self):
        self.positions = []  # store recent (x, y) positions
        self.max_buffer = 30

    def update_position(self, x, y):
        self.positions.append((x, y, datetime.now()))
        if len(self.positions) > self.max_buffer:
            self.positions.pop(0)

    def calculate_velocity(self):
        if len(self.positions) < 2:
            return 0
        (x1, y1, t1), (x2, y2, t2) = self.positions[0], self.positions[-1]
        dt = (t2 - t1).total_seconds() + 1e-5
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance / dt

    def assess_threat(self, acoustic_label=None):
        velocity = self.calculate_velocity()

        # Acoustic label simulation
        acoustic_status = "SIMULATED"
        if acoustic_label is None:
            acoustic_label = np.random.choice([0, 1])  # 0=cruise, 1=hover

        if velocity > 50 or acoustic_label == 1:
            return "HIGH"
        elif 20 < velocity <= 50:
            return "MODERATE"
        else:
            return "LOW"
