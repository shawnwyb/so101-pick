"""Original timed coarse-to-fine experiment, integrated with LeRobot's session."""
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from waypoints import JOINTS, commit_sequence, read_joints


def move(robot, start, goal, step_deg=1.5, fps=30.0, cancelled=lambda: False, observe=False):
    """Original commit_session._move arithmetic; optional cancellation/recording hooks."""
    dist = max(abs(goal[k] - start[k]) for k in start)
    steps = max(1, int(dist / step_deg))
    dt = 1.0 / fps
    keys = [k for k in goal if not k.startswith("_")]
    for i in range(1, steps + 1):
        if observe:
            robot.get_observation()
        if cancelled():
            raise InterruptedError("Commit cancelled")
        robot.send_action({k: start[k] + (goal[k] - start[k]) * i / steps for k in keys})
        time.sleep(dt)
    return {k: goal[k] for k in keys}


class Trace:
    """Log existing observations/actions; no background hardware reads."""

    def __init__(self, robot, directory):
        self.robot, self.directory = robot, directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "frames").mkdir()
        self.stream = (directory / "actions.jsonl").open("w", buffering=1)
        self.frames = (directory / "frames.jsonl").open("w", buffering=1)
        self.original_observe, self.original_send = robot.get_observation, robot.send_action
        self.latest, self.observed_at, self.previous_command = {}, None, None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending = None
        self.last_snapshot = self.last_video = float("-inf")
        self.recording, self.writers = None, {}
        self.phase = "idle"
        robot.get_observation, robot.send_action = self.observe, self.send

    def observe(self):
        observation = self.original_observe()
        self.latest, self.observed_at = observation, time.monotonic()
        snapshot = self.observed_at - self.last_snapshot >= 0.5
        video = self.recording if self.observed_at - self.last_video >= 0.1 else None
        if self.pending is not None and self.pending.done():
            self.pending.result()  # Surface disk/codec failures instead of silently losing evidence.
        if (snapshot or video) and (self.pending is None or self.pending.done()):
            frames = {k: observation[k].copy() for k in ("cam0", "cam1") if k in observation}
            if frames:
                self.pending = self.executor.submit(self.save, frames, time.time_ns(), snapshot, video)
                if snapshot:
                    self.last_snapshot = self.observed_at
                if video:
                    self.last_video = self.observed_at
        return observation

    def save(self, frames, stamp, snapshot, video):
        from PIL import Image
        for key, frame in frames.items():
            if snapshot:
                Image.fromarray(frame).save(self.directory / "frames" / f"{stamp}_{key}.jpg")
            if video:
                import cv2
                writer_key = (str(video), key)
                if writer_key not in self.writers:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(video / f"{key}.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot create video for {key}")
                    self.writers[writer_key] = writer
                self.writers[writer_key].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frames.write(json.dumps({"time_ns": stamp, "snapshot": snapshot,
                                      "video": str(video) if video else None}) + "\n")

    def send(self, action):
        started = time.monotonic()
        sent = self.original_send(action)
        requested = {k: float(v) for k, v in action.items() if k in JOINTS}
        transmitted = {k: float(v) for k, v in sent.items() if k in JOINTS}
        self.stream.write(json.dumps({
            "schema_version": 2, "time": time.time(), "phase": self.phase,
            "observation_age_s": None if self.observed_at is None else started - self.observed_at,
            "command_interval_s": None if self.previous_command is None else started - self.previous_command,
            "measured_latest": {k: float(self.latest[k]) for k in JOINTS if k in self.latest},
            "requested": requested, "sent": transmitted,
            "clipped_joints": [k for k in transmitted if k in requested
                               and abs(transmitted[k] - requested[k]) > 1e-4],
        }) + "\n")
        self.previous_command = started
        return sent

    def start_recording(self):
        self.recording = self.directory / ("commit_" + datetime.now().strftime("%H%M%S-%f"))
        self.recording.mkdir()
        self.last_video = float("-inf")
        return self.recording

    def stop_recording(self):
        self.recording = None
        try:
            if self.pending:
                self.pending.result()
        finally:
            for writer in self.writers.values():
                writer.release()
            self.writers.clear()

    def close(self):
        self.robot.get_observation, self.robot.send_action = self.original_observe, self.original_send
        try:
            self.stop_recording()
        finally:
            self.executor.shutdown(wait=True)
            self.stream.close()
            self.frames.close()


class HybridStrategy:
    """Run policy and primitives on the same controller thread, in sequence."""

    def __init__(self, strategy, ctx, options):
        self.strategy, self.ctx, self.options = strategy, ctx, options
        self.pending = None
        self.lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.strategy, name)

    def cancelled(self):
        ctrl = self.controller
        # This adapter targets the pinned LeRobot controller; no public pause-only
        # method exists there. Its request flags also let primitives obey reset/stop.
        return (ctrl.stopped or ctrl.failed or ctrl._stop_requested.is_set()
                or ctrl._reset_requested.is_set() or self.ctx.runtime.shutdown_event.is_set())

    def wait(self, seconds, observe=False):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.cancelled():
                raise InterruptedError("Commit cancelled")
            if observe:
                self.ctx.hardware.robot_wrapper.get_observation()
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    def request(self, cmd, record=False):
        parts = cmd.args.lower().split()
        reset = "reset" in parts
        numbers = [p for p in parts if p != "reset"]
        try:
            seconds = float(numbers[0]) if numbers else self.options.coarse_seconds
            if len(numbers) > 1 or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError
            sequence = commit_sequence(self.options.grasp, self.options.closed_gripper)
        except (ValueError, OSError, KeyError) as exc:
            self.say(f"Usage: /commit [positive seconds] [reset]. Check grasp file: {exc}")
            return
        with self.lock:
            if self.pending or self.controller.running or self.controller._reset_requested.is_set():
                self.say("A run/reset is active. Wait for it to finish or /reset first.")
                return
            self.pending = (seconds, reset, record, sequence)
            if not self.controller.start():
                self.pending = None
                self.say("Cannot start: session is stopping or has failed.")
                return
        self.say(f"Commit queued: {seconds:g}s policy, then lower / clamp / lift / hold.")

    def run(self, ctx):
        with self.lock:
            request, self.pending = self.pending, None
        if request is None:
            self.trace.phase = "policy"
            try:
                return self.strategy.run(ctx)
            finally:
                self.trace.phase = "idle"
        seconds, reset, record, sequence = request
        duration = ctx.runtime.cfg.duration
        robot = ctx.hardware.robot_wrapper
        try:
            if record:
                self.say(f"Recording: {self.trace.start_recording()}")
            self.trace.phase = "coarse"
            ctx.runtime.cfg.duration = seconds
            self.strategy.run(ctx)
            # The policy loop has returned before any primitive can send a target.
            ctx.policy.inference.pause()
            if self.cancelled():
                raise InterruptedError("Commit cancelled before handoff")
            self.wait(0.3, record)  # Original post-policy settling pause.
            current = read_joints(robot)  # Original: measure once, then chain targets.
            self.say("Policy stopped; taught grasp taking over.")
            for stage, waypoint in zip(("lower", "clamp", "lift"), sequence, strict=True):
                self.trace.phase = stage
                self.say(f"{stage}: {waypoint['name']}")
                current = move(robot, current, waypoint["joints"],
                     step_deg=1.5 if stage == "clamp" else self.options.dive_step,
                     fps=30 if stage == "clamp" else self.options.dive_fps,
                     cancelled=self.cancelled, observe=record)
                if stage == "clamp":
                    self.wait(1, record)
                    if self.cancelled():
                        raise InterruptedError("Commit cancelled")
                    robot.send_action(waypoint["joints"])
                    self.wait(3, record)
                    current = move(robot, current, waypoint["joints"], step_deg=0.4, fps=15,
                         cancelled=self.cancelled, observe=record)
                    self.wait(1, record)
                else:
                    self.wait(0.8, record)
            self.trace.phase = "hold"
            if self.cancelled():
                raise InterruptedError("Commit cancelled")
            robot.send_action(current)
            self.wait(5 if record else 0, record)
            if reset:
                self.controller.reset()
            self.say("Commit finished. Holding position; /reset to return, /stop to disconnect.")
        except InterruptedError:
            self.say("Commit cancelled; pending reset/stop will now run.")
        finally:
            ctx.runtime.cfg.duration = duration
            self.trace.phase = "idle"
            self.trace.stop_recording()


def install(session_class, options, directory):
    """Register /commit and /commitandrecord; preserve LeRobot's other commands."""
    original_init, original_run = session_class.__init__, session_class.run

    def initialize(session, strategy, ctx, input_stream=None):
        hybrid = HybridStrategy(strategy, ctx, options)
        original_init(session, hybrid, ctx, input_stream=input_stream)
        hybrid.controller, hybrid.say = session.controller, session._print
        session._baseline_trace = hybrid.trace = Trace(ctx.hardware.robot_wrapper, directory)
        session._commands.update({
            "commit": (hybrid.request, " [seconds] [reset]", "policy coarse approach, then taught grasp"),
            "commitandrecord": (lambda cmd: hybrid.request(cmd, True), " [seconds] [reset]",
                                "commit plus top/wrist video through five seconds of hold"),
        })
        old_reset = session._commands["reset"]

        def reset(cmd):
            with hybrid.lock:
                hybrid.pending = None
            old_reset[0](cmd)

        session._commands["reset"] = (reset, old_reset[1], old_reset[2])

    def run(session):
        try:
            return original_run(session)
        finally:
            session._baseline_trace.close()

    session_class.__init__, session_class.run = initialize, run

    def restore():
        session_class.__init__, session_class.run = original_init, original_run

    return restore
