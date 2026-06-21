"""
xArm7 push demo — L-shape tool, objects on the floor.

Run from the xArm/ directory:
    python push_demo.py

Controls
--------
  W / S      move end-effector forward / back  (+/- X)
  A / D      move end-effector left  / right   (+/- Y)
  Q / E      move end-effector up    / down    (+/- Z)
  I / K      rotate around X  (+/-)
  J / L      rotate around Y  (+/-)
  U / O      rotate around Z  (+/-)
  T          sweep left → right to clear the table
  R          reset objects to OBJECT_INIT positions (also cancels sweep)
  Mouse      drag the red target box directly
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

# ── tunable parameters ──────────────────────────────────────────────────────
dt              = 0.002   # simulation timestep (s)
integration_dt  = 1.0     # IK velocity integration window
damping         = 1e-4    # Jacobian damping (singularity avoidance)
gravity_comp    = True    # enable gravity compensation on arm links
max_angvel      = 0.0     # joint-velocity cap (rad/s); 0 = disabled

pos_step        = 0.02    # m per keypress
rot_step        = 0.02    # rad per keypress
# ────────────────────────────────────────────────────────────────────────────

# ── Edit these to place objects anywhere on the floor ────────────────────────
# (x, y, z)  —  z=0 keeps each object's bottom flush with the floor.
# x is forward from the robot base, y is lateral.
OBJECT_INIT = {
    "game_controller": (0.42,  0.05, 0.0),
    "red_bowl":        (0.30, -0.10, 0.0),
    "hammer":          (0.40,  -0.28, 0.0),
    "spring_clamp":    (0.38, -0.22, 0.0),
    "painters_tape":   (0.50, -0.08, 0.0),
}
# ────────────────────────────────────────────────────────────────────────────

# ── Sweep parameters — adjust to match your scene layout ────────────────────
SWEEP_X           = 0.4   # forward reach during sweep (m)
SWEEP_Z           = 0.18   # height of L-shape during sweep (m)
SWEEP_Y_START     = 0.40   # left edge  (positive Y = robot's left)
SWEEP_Y_END       = -0.2  # right edge
SWEEP_APPROACH_T  = 1.5    # seconds to raise & move to sweep start
SWEEP_MOTION_T    = 3.0    # seconds for the left→right stroke
# ────────────────────────────────────────────────────────────────────────────


def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco >= 3.1.0"

    model = mujoco.MjModel.from_xml_path("xarm_description/push_scene.xml")
    data  = mujoco.MjData(model)
    model.opt.timestep = dt

    # ── IK setup ────────────────────────────────────────────────────────────
    site_id = model.site("attachment_site").id

    arm_bodies = ["link_base", "link1", "link2", "link3",
                  "link4", "link5", "link6", "link7"]
    body_ids   = [model.body(name).id for name in arm_bodies]
    if gravity_comp:
        model.body_gravcomp[body_ids] = 1.0

    joint_names    = ["joint1","joint2","joint3","joint4","joint5","joint6","joint7"]
    actuator_names = ["act1",  "act2",  "act3",  "act4",  "act5",  "act6",  "act7"]
    dof_ids      = np.array([model.joint(name).id    for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in actuator_names])

    key_id   = model.key("push_home").id
    mocap_id = model.body("target").mocapid[0]

    # Pre-allocate IK arrays.
    jac          = np.zeros((6, model.nv))
    diag         = damping * np.eye(6)
    error        = np.zeros(6)
    error_pos    = error[:3]
    error_ori    = error[3:]
    site_quat    = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat   = np.zeros(4)

    def _place_objects() -> None:
        """Apply OBJECT_INIT positions and zero velocities for all scene objects."""
        for name, (x, y, z) in OBJECT_INIT.items():
            body = model.body(name)
            qa = model.jnt_qposadr[body.jntadr[0]]
            da = model.jnt_dofadr[body.jntadr[0]]
            data.qpos[qa:qa+3]   = [x, y, z]
            data.qpos[qa+3:qa+7] = [1, 0, 0, 0]
            data.qvel[da:da+6]   = 0.0
        mujoco.mj_forward(model, data)

    # ── sweep state ──────────────────────────────────────────────────────────
    # p2_quat: EE orientation snapped at the first step of phase 2 and held fixed.
    sweep = {'active': False, 't0': 0.0, 'from_pos': np.zeros(3), 'p2_quat': None}

    def _start_sweep() -> None:
        sweep['active']   = True
        sweep['t0']       = data.time
        sweep['from_pos'] = data.mocap_pos[mocap_id].copy()
        sweep['p2_quat']  = None
        print("Sweep started.")

    def _tick_sweep() -> None:
        """Move mocap target along the sweep trajectory each sim step."""
        if not sweep['active']:
            return
        elapsed = data.time - sweep['t0']

        if elapsed < SWEEP_APPROACH_T:
            # Phase 1: move to sweep start; leave orientation for the IK to resolve.
            alpha = elapsed / SWEEP_APPROACH_T
            alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep
            data.mocap_pos[mocap_id] = (1 - alpha) * sweep['from_pos'] + alpha * np.array(
                [SWEEP_X, SWEEP_Y_START, SWEEP_Z]
            )

        elif elapsed < SWEEP_APPROACH_T + SWEEP_MOTION_T:
            # Phase 2: lock orientation from actual EE pose at first entry.
            if sweep['p2_quat'] is None:
                sweep['p2_quat'] = np.zeros(4)
                mujoco.mju_mat2Quat(sweep['p2_quat'], data.site(site_id).xmat)
            alpha = (elapsed - SWEEP_APPROACH_T) / SWEEP_MOTION_T
            alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep
            y = SWEEP_Y_START + alpha * (SWEEP_Y_END - SWEEP_Y_START)
            data.mocap_pos[mocap_id]  = [SWEEP_X, y, SWEEP_Z]
            data.mocap_quat[mocap_id] = sweep['p2_quat']

        else:
            sweep['active'] = False
            print("Sweep done.")

    # ── keyboard callback ────────────────────────────────────────────────────
    def key_callback(key: int) -> None:
        try:
            ch = chr(key)
        except (ValueError, OverflowError):
            return

        if ch == 'T':
            _start_sweep()
            return

        # Manual controls are ignored while a sweep is running.
        if sweep['active']:
            return

        if ch == 'W': data.mocap_pos[mocap_id][0] += pos_step
        if ch == 'S': data.mocap_pos[mocap_id][0] -= pos_step
        if ch == 'A': data.mocap_pos[mocap_id][1] += pos_step
        if ch == 'D': data.mocap_pos[mocap_id][1] -= pos_step
        if ch == 'Q': data.mocap_pos[mocap_id][2] += pos_step
        if ch == 'E': data.mocap_pos[mocap_id][2] -= pos_step

        quat = data.mocap_quat[mocap_id].copy()
        def apply_rot(axis, angle):
            dq = np.zeros(4)
            mujoco.mju_axisAngle2Quat(dq, axis, angle)
            mujoco.mju_mulQuat(quat, dq, quat)
        if ch == 'I': apply_rot([1, 0, 0],  rot_step)
        if ch == 'K': apply_rot([1, 0, 0], -rot_step)
        if ch == 'J': apply_rot([0, 1, 0],  rot_step)
        if ch == 'L': apply_rot([0, 1, 0], -rot_step)
        if ch == 'U': apply_rot([0, 0, 1],  rot_step)
        if ch == 'O': apply_rot([0, 0, 1], -rot_step)
        data.mocap_quat[mocap_id] = quat

        if ch == 'R':
            sweep['active'] = False
            _place_objects()
            print("Objects reset.")

    # ── viewer loop ─────────────────────────────────────────────────────────
    with mujoco.viewer.launch_passive(
        model=model, data=data,
        show_left_ui=False, show_right_ui=False,
        key_callback=key_callback,
    ) as viewer:

        # Reset robot to keyframe, place objects, align mocap to EE.
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        _place_objects()
        data.mocap_pos[mocap_id]  = data.site(site_id).xpos.copy()
        mujoco.mju_mat2Quat(data.mocap_quat[mocap_id], data.site(site_id).xmat)

        mujoco.mjv_defaultFreeCamera(model, viewer.cam)
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        print(__doc__)

        while viewer.is_running():
            step_start = time.time()

            _tick_sweep()   # updates mocap_pos when a sweep is active

            # ── diff-IK ──────────────────────────────────────────────────
            error_pos[:] = data.mocap_pos[mocap_id] - data.site(site_id).xpos

            mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
            mujoco.mju_negQuat(site_quat_conj, site_quat)
            mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
            mujoco.mju_quat2Vel(error_ori, error_quat, 1.0)

            mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            # During phase 2 of a sweep, weight orientation 4× to keep EE fixed.
            if sweep['p2_quat'] is not None:
                w_err = error.copy()
                w_err[3:] *= 4.0
                dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, w_err)
            else:
                dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

            if max_angvel > 0:
                dq_abs_max = np.abs(dq).max()
                if dq_abs_max > max_angvel:
                    dq *= max_angvel / dq_abs_max

            q = data.qpos.copy()
            mujoco.mj_integratePos(model, q, dq, integration_dt)
            q[dof_ids] = np.clip(q[dof_ids],
                                 model.jnt_range[dof_ids, 0],
                                 model.jnt_range[dof_ids, 1])
            data.ctrl[actuator_ids] = q[dof_ids]
            # ─────────────────────────────────────────────────────────────

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            if dt - elapsed > 0:
                time.sleep(dt - elapsed)


if __name__ == "__main__":
    main()
