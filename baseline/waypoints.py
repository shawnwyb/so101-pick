"""Teach or replay a fixed sequence; replay only moves with --execute."""
import argparse
import json
import math
import time
from pathlib import Path

DEFAULT_NAMES = ("approach", "lower", "close", "lift")
JOINTS = tuple(f"{name}.pos" for name in (
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"
))


def make_robot(port, robot_id):
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    return SO101Follower(SO101FollowerConfig(
        port=port, id=robot_id, use_degrees=True, cameras={}, max_relative_target=None,
    ))


def validate_pose(pose):
    if not isinstance(pose, dict) or set(pose) != set(JOINTS):
        raise ValueError(f"Each pose must contain exactly these joints: {JOINTS}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in pose.values()):
        raise ValueError("Joint positions must be finite numbers")
    return {key: float(pose[key]) for key in JOINTS}


def read_joints(robot):
    observation = robot.get_observation()
    return validate_pose({key: observation[key] for key in JOINTS})


def move(robot, goal, step=0.4, fps=15.0, cancelled=lambda: False, observe=False):
    """Interpolate from measured state; gripper increments use native units."""
    if not math.isfinite(step) or step <= 0 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("step and fps must be finite and positive")
    goal = validate_pose(goal)
    if cancelled():
        raise InterruptedError("Motion cancelled")
    start = read_joints(robot)
    steps = max(1, math.ceil(max(abs(goal[k] - start[k]) for k in JOINTS) / step))
    for i in range(1, steps + 1):
        if observe:
            robot.get_observation()  # Feed the session recorder on this same control thread.
        if cancelled():
            raise InterruptedError("Motion cancelled")
        robot.send_action({k: start[k] + (goal[k] - start[k]) * i / steps for k in JOINTS})
        time.sleep(1 / fps)


def commit_sequence(path, closed=0.0):
    """Select the original dive/close/lift, without silent index fallbacks."""
    payload = load(path)
    stages = []
    indices = []
    for needle in ("lower", "clos", "lift"):
        matches = [(i, w) for i, w in enumerate(payload["waypoints"])
                   if needle in w["name"].lower()]
        if len(matches) != 1:
            raise ValueError(f"Expected one waypoint containing {needle!r}, found {len(matches)}")
        index, waypoint = matches[0]
        indices.append(index)
        stages.append({"name": waypoint["name"], "joints": dict(waypoint["joints"])})
    if indices != sorted(indices):
        raise ValueError("Waypoints must be ordered lower, close, lift")
    for waypoint in stages[1:]:
        waypoint["joints"]["gripper.pos"] = closed
    return stages


def load(path):
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("use_degrees") is not True:
        raise ValueError("Waypoint file must declare use_degrees: true")
    waypoints = payload.get("waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError("Waypoint file must contain a nonempty waypoints list")
    for waypoint in waypoints:
        if not isinstance(waypoint, dict) or not isinstance(waypoint.get("name"), str) or not waypoint["name"].strip():
            raise ValueError("Every waypoint needs a nonempty name")
        waypoint["joints"] = validate_pose(waypoint.get("joints"))
    return payload


def teach(args):
    if args.file.exists():
        raise ValueError(f"File already exists; choose a new --file: {args.file}")
    robot = make_robot(args.port, args.id)
    print("Support the arm BEFORE connecting; teaching disables torque.", flush=True)
    try:
        robot.connect(calibrate=False)
        robot.bus.disable_torque()
        waypoints = []
        for name in args.names:
            while True:
                answer = input(f"Move to {name!r}; Enter=capture, q=cancel: ").strip().lower()
                if answer == "q":
                    print("Cancelled; no file written.")
                    return
                if not answer:
                    break
            waypoints.append({"name": name, "joints": read_joints(robot)})
        payload = dict(port=args.port, id=args.id, use_degrees=True,
                       joint_order=list(JOINTS), waypoints=waypoints)
        args.file.parent.mkdir(parents=True, exist_ok=True)
        with args.file.open("x") as stream:
            stream.write(json.dumps(payload, indent=2) + "\n")
        print(f"Saved {args.file}")
    finally:
        if robot.is_connected:
            robot.disconnect()


def replay(args):
    payload = load(args.file)
    for waypoint in payload["waypoints"]:
        print(f"{waypoint['name']}: {waypoint['joints']}")
    if not args.execute:
        print("Dry run: no hardware connection. Add --execute to move.")
        return
    port = args.port or payload.get("port")
    robot_id = args.id or payload.get("id")
    if not isinstance(port, str) or not port.strip() or not isinstance(robot_id, str) or not robot_id.strip():
        raise ValueError("Provide --port and --id, or include them in the waypoint file")
    robot = make_robot(port, robot_id)
    print("Executing every pose in file order; support the arm when torque releases.", flush=True)
    try:
        robot.connect(calibrate=False)
        robot.bus.enable_torque()
        for waypoint in payload["waypoints"]:
            print(f"Moving to {waypoint['name']}", flush=True)
            move(robot, waypoint["joints"], args.step, args.fps)
            time.sleep(args.pause)
        time.sleep(args.hold)
    finally:
        if robot.is_connected:
            robot.disconnect()


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("teach", help="capture poses by hand; disables torque")
    capture.add_argument("--names", nargs="+", default=list(DEFAULT_NAMES))
    playback = commands.add_parser("replay", help="validate and print poses; --execute moves")
    playback.add_argument("--execute", action="store_true")
    playback.add_argument("--step", type=positive, default=0.4,
                          help="max target increment per tick (degrees; native units for gripper)")
    playback.add_argument("--fps", type=positive, default=15.0)
    playback.add_argument("--pause", type=nonnegative, default=1.0)
    playback.add_argument("--hold", type=nonnegative, default=5.0)
    for command in (capture, playback):
        command.add_argument("--file", type=Path, required=True)
        command.add_argument("--port", required=command is capture)
        command.add_argument("--id", required=command is capture)
    args = parser.parse_args()
    try:
        (teach if args.command == "teach" else replay)(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
