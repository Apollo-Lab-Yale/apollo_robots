import numpy as np
from planner import MujocoPlanner
import mujoco
import time

np.set_printoptions(precision=3)

def main():
    planner = MujocoPlanner()
    data = planner.data # Access MuJoCo data for initialization

    home_pose_qpos = np.array([0.0, 0.0, -0.1, 0.0, 0.0, 0.0, -1.5])
    data.qpos[:len(home_pose_qpos)] = home_pose_qpos

    # Step simulation to update positions
    mujoco.mj_step(planner.model, data)
    if planner.viewer: planner.viewer.sync()

    target_pos = [0.30, 0, 0.13]

    # Target orientation (quaternion: w, x, y, z)
    # This orientation is for the gripper facing downward (-Z axis of the end-effector)
    target_quat = np.array([0, -0.969, 0, 0.25])
    print(f"Target Position: {target_pos}")

    print("Solving IK...")
    q_goal = planner.solve_ik(target_pos, target_quat)

    if q_goal is None:
        print("Could not find IK solution. Exiting.")
        planner.close_viewer()
        return

    # Use the current 6-DOF joint position as the start state for planning
    q_start = data.qpos[:6].copy()
    path = planner.rrt_plan(q_start, q_goal)

    if path is None:
        print("Path planning failed. Exiting.")
        planner.close_viewer()
        return

    print(f"Executing Path with {len(path)} waypoints on the real robot...")

    # TODO: Open Gripper

    # Execute move to pick pose
    for i, waypoint in enumerate(path):
        for _ in range(50):
            data.ctrl[:6] = waypoint
            mujoco.mj_step(planner.model, data)
            if planner.viewer and planner.viewer.is_running():
                planner.viewer.sync()
                time.sleep(0.01)

        print(f"Reached Waypoint {i + 1}/{len(path)}")

    # TODO: Close Gripper
    print("Closing Gripper to pick object...")

    # Delay for gripping (real robot operation)
    time.sleep(1.0)

    # TODO: Return to start
    print("Returning to start position...")

    planner.close_viewer()


if __name__ == "__main__":
    main()