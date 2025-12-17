import mujoco
import mujoco.viewer
import numpy as np
import os
import time

SCENE_XML = """
<mujoco model="z1_scene">
    <include file="../z1_description/z1.xml"/>

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20"/>
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>

    <worldbody>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>

        <body name="target_object" pos="0.35 0.0 0.021">
            <joint type="free"/>
            <geom name="target_geom" type="box" size="0.02 0.02 0.02" rgba="1 0 0 1" mass="0.1"/>
        </body>
    </worldbody>

    </mujoco>
"""

class RRTNode:
    def __init__(self, q, parent=None):
        self.q = q
        self.parent = parent

class MujocoPlanner:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(SCENE_XML)
        self.data = mujoco.MjData(self.model)

        # Find the ID of the end-effector (gripper stator) for IK
        ee_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "z1_GripperStator"
        )
        self.ee_geom_id = ee_geom_id if ee_geom_id != -1 else self.model.ngeom - 1

        # Setup viewer
        self.viewer = None
        if os.environ.get('DISPLAY') is not None:
            try:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            except Exception:
                print("Warning: Could not launch MuJoCo viewer.")

        # Z1 Approximate Joint Limits
        self.joint_limits_min = np.array([-2.6, 0.0, -2.8, -1.5, -1.3, -2.7])
        self.joint_limits_max = np.array([2.6, 2.9, 0.0, 1.5, 1.3, 2.7])
        self.dof_indices = np.arange(6)

    def _update_sim(self, duration=0.005):
        mujoco.mj_step(self.model, self.data)
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
            time.sleep(duration)

    def solve_ik(self, target_pos: list, target_quat: list, max_steps=500, tol=1e-5):
        q0 = self.data.qpos.copy()
        temp_data = self.data  # Use self.data directly for iterative update

        target_pos = np.array(target_pos)
        target_quat = np.array(target_quat)

        success = False
        error_norm = np.inf

        for i in range(max_steps):
            mujoco.mj_kinematics(self.model, temp_data)
            mujoco.mj_comPos(self.model, temp_data) #TODO: what is this?

            current_pos = temp_data.geom_xpos[self.ee_geom_id]
            current_mat = temp_data.geom_xmat[self.ee_geom_id].reshape(3, 3)

            err_pos = target_pos - current_pos

            target_mat_flat = np.zeros(9)
            mujoco.mju_quat2Mat(target_mat_flat, target_quat)
            target_mat = target_mat_flat.reshape(3, 3)

            # Orientation error (half of the rotation vector)
            err_rot = 0.5 * (np.cross(current_mat[:, 0], target_mat[:, 0]) +
                             np.cross(current_mat[:, 1], target_mat[:, 1]) +
                             np.cross(current_mat[:, 2], target_mat[:, 2]))

            error = np.concatenate([err_pos, err_rot])
            error_norm = np.linalg.norm(error)

            if error_norm < tol:
                success = True
                break

            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacGeom(self.model, temp_data, jacp, jacr, self.ee_geom_id)

            jac = np.vstack([jacp, jacr])
            jac_arm = jac[:, self.dof_indices]

            # Damped Least Squares
            lambda_val = 0.1
            diag = lambda_val * np.eye(6)

            dq_arm = jac_arm.T @ np.linalg.solve(jac_arm @ jac_arm.T + diag, error)
            temp_data.qpos[self.dof_indices] += dq_arm

            # Apply Joint Limits
            temp_data.qpos[self.dof_indices] = np.clip(
                temp_data.qpos[self.dof_indices], self.joint_limits_min, self.joint_limits_max
            )

        result_q = temp_data.qpos.copy()[self.dof_indices]

        # Restore original state and forward kinematics
        temp_data.qpos[:] = q0
        mujoco.mj_forward(self.model, temp_data)

        if success:
            return result_q
        else:
            print(f"IK Failed to converge. Final error: {error_norm:.4f}")
            return None

    def is_collision(self, q_check: np.ndarray, debug=False):
        q_save = self.data.qpos.copy()

        # Temporarily set the joint positions to check
        self.data.qpos[:6] = q_check[:6]
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

        in_collision = False

        for i in range(self.data.ncon):
            # contact = self.data.contact[i]
            # No need to inspect details, just the fact that ncon > 0 is enough
            in_collision = True
            if debug:
                 contact = self.data.contact[i]
                 name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
                 name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
                 print(f"Collision Detected between: {name1} and {name2}")
            break

        # Restore original state
        self.data.qpos[:] = q_save
        return in_collision

    def rrt_plan(self, start_q: np.ndarray, goal_q: np.ndarray, max_iter=30000, step_size=0.5):
        if self.is_collision(start_q, debug=True):
            print("CRITICAL: Start configuration is in collision! Cannot plan.")
            return None

        start_node = RRTNode(start_q)
        goal_node = RRTNode(goal_q)

        tree = [start_node]

        for i in range(max_iter):
            # Bias sampling towards the goal (20% chance)
            if np.random.rand() < 0.2:
                sample_q = goal_node.q
            else:
                sample_q = np.random.uniform(self.joint_limits_min, self.joint_limits_max)

            # Find nearest node
            distances = [np.linalg.norm(node.q - sample_q) for node in tree]
            nearest_idx = np.argmin(distances)
            nearest_node = tree[nearest_idx]

            # Extend toward the sample
            direction = sample_q - nearest_node.q
            distance = np.linalg.norm(direction)

            direction = direction / distance
            move_dist = min(step_size, distance)
            new_q = nearest_node.q + direction * move_dist

            # Check collision for new point
            if not self.is_collision(new_q):
                new_node = RRTNode(new_q, nearest_node)
                tree.append(new_node)

                # Check if the new node is close to the goal
                if np.linalg.norm(new_q - goal_node.q) < 0.04:
                    if not self.is_collision(goal_node.q):
                        print(f"Path found in {i} iterations!")
                        path = [goal_node.q]
                        curr = new_node
                        while curr is not None:
                            path.append(curr.q)
                            curr = curr.parent
                        return path[::-1] # Reverse to go from start to goal

        print("RRT failed to find a path.")
        return None

    def close_viewer(self):
        if self.viewer:
            self.viewer.close()

    def run_simulation(self):
        if self.viewer:
            print("Keeping MuJoCo window open. Close to exit.")
            while self.viewer.is_running():
                self._update_sim(duration=0.01)