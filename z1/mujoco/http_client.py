import requests

# Global base URL for the Z1 robot API
ROBOT_API_URL = "http://192.168.123.220:12000/unitree/z1"
DATABASE_TEMPLATE = {
    "func": "",
    "args": {},
}

class RobotController:
    def __init__(self, url=ROBOT_API_URL):
        self.url = url
        print(f"RobotController initialized for API: {self.url}")

    def _send_request(self, func_name, args=None):
        data = DATABASE_TEMPLATE.copy()
        data["func"] = func_name
        if args is not None:
            data["args"] = args

        try:
            response = requests.post(self.url, json=data)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"Error sending request for {func_name}: {e}")
            return None

    def label_run(self, label: str):
        assert len(label) < 10
        return self._send_request("labelRun", {"label": label})

    def label_save(self, label: str):
        assert len(label) < 10
        return self._send_request("labelSave", {"label": label})

    def back_to_start(self):
        return self._send_request("backToStart")

    def passive(self):
        return self._send_request("Passive")

    def get_q(self):
        response = self._send_request("getQ")
        if response:
            try:
                return response.json()
            except requests.exceptions.JSONDecodeError:
                print("Failed to decode JSON response for getQ.")
                return None
        return None

    def move_j(self, q: list, gripper_pos: float = 0, speed: float = 0.5):
        assert len(q) == 6
        args = {
            "q": q,
            "gripperPos": gripper_pos,
            "maxSpeed": speed,
        }
        return self._send_request("MoveJ", args)

    def set_gripper(self, position: int, speed: int = 128, force: int = 128):
        position = max(0, min(255, position))
        speed = max(0, min(255, speed))
        force = max(0, min(255, force))

        args = {
            "position": position,
            "speed": speed,
            "force": force,
        }
        return self._send_request("setGripper", args)