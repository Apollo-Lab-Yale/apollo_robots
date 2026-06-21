"""
Real xArm7 sweep — faithfully mirrors the push_demo.py sweep action.

Run from the xArm/ directory:
    python sweep_real.py --ip 192.168.1.xxx

Sequence (matches push_demo.py):
  1. Move to push_home joint angles  (same as the simulation keyframe).
  2. Lock EE orientation from the home pose.
  3. Phase 1 — approach: move linearly to (SWEEP_X, SWEEP_Y_START, SWEEP_Z).
  4. Phase 2 — stroke:   sweep Y from SWEEP_Y_START → SWEEP_Y_END at fixed X, Z,
                         holding the orientation locked in step 2.

All positions are in mm, angles in degrees (xArm SDK convention).
Keep push_demo.py open and copy SWEEP_* constants from there (×1000 for mm).
"""

import argparse
import sys
import time

# ── Match these to push_demo.py (values × 1000: m → mm) ─────────────────────
SWEEP_X          =  400.0   # = push_demo SWEEP_X   × 1000
SWEEP_Z          =  120.0   # = push_demo SWEEP_Z   × 1000
SWEEP_Y_START    =  300.0   # = push_demo SWEEP_Y_START × 1000
SWEEP_Y_END      = -200.0   # = push_demo SWEEP_Y_END   × 1000

APPROACH_SPEED   =  100     # mm/s
SWEEP_SPEED      =   80     # mm/s for the stroke
MVACC            =  500     # mm/s²

# push_home joint angles from the simulation keyframe (radians)
HOME_ANGLES_RAD  = [0.0, -0.247, 0.0, 0.909, 0.0, 1.15644, 0.0]
# ─────────────────────────────────────────────────────────────────────────────


def check(code: int, label: str) -> None:
    if code != 0:
        print(f"[ERROR] {label} returned code {code}")
        sys.exit(1)


def main(ip: str) -> None:
    from xarm.wrapper import XArmAPI

    arm = XArmAPI(ip, baud_checkset=False)
    time.sleep(0.5)

    print(f"Connected to xArm7 at {ip}")
    print(f"State: {arm.state}  Mode: {arm.mode}  Error: {arm.error_code}")

    if arm.error_code != 0:
        print("Clearing errors...")
        arm.clean_error()
        time.sleep(0.5)

    check(arm.motion_enable(enable=True), "motion_enable")
    check(arm.set_mode(0),  "set_mode(position)")
    check(arm.set_state(0), "set_state(sport)")
    time.sleep(0.5)

    # ── Step 1: go to push_home joint configuration ───────────────────────────
    print("\nMoving to push_home joint angles...")
    code = arm.set_servo_angle(
        angle=HOME_ANGLES_RAD,
        is_radian=True,
        speed=0.3,       # rad/s — slow and safe
        mvacc=5.0,
        wait=True,
    )
    check(code, "set_servo_angle (home)")

    # ── Step 2: lock EE orientation from the home pose ────────────────────────
    code, pose = arm.get_position(is_radian=False)
    check(code, "get_position at home")
    _, _, _, roll, pitch, yaw = pose
    print(f"Home pose: x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f}  "
          f"roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}")
    print(f"Orientation locked: roll={roll:.1f} pitch={pitch:.1f} yaw={yaw:.1f}")

    # ── Phase 1: approach sweep start ─────────────────────────────────────────
    print(f"\nPhase 1 — approach  ({SWEEP_X}, {SWEEP_Y_START}, {SWEEP_Z}) mm ...")
    code = arm.set_position(
        x=SWEEP_X, y=SWEEP_Y_START, z=SWEEP_Z,
        roll=roll, pitch=pitch, yaw=yaw,
        speed=APPROACH_SPEED, mvacc=MVACC,
        wait=True,
    )
    check(code, "approach move")

    # ── Phase 2: sweep stroke ─────────────────────────────────────────────────
    print(f"\nPhase 2 — stroke  y: {SWEEP_Y_START} → {SWEEP_Y_END} mm ...")
    code = arm.set_position(
        x=SWEEP_X, y=SWEEP_Y_END, z=SWEEP_Z,
        roll=roll, pitch=pitch, yaw=yaw,
        speed=SWEEP_SPEED, mvacc=MVACC,
        wait=True,
    )
    check(code, "sweep stroke")

    print("\nSweep done.")
    arm.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="xArm7 real-robot sweep")
    parser.add_argument("--ip", required=True, help="Robot IP address (e.g. 192.168.1.220)")
    args = parser.parse_args()
    main(args.ip)
