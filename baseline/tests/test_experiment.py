"""Hardware-free regression checks for launch, handoff and cancellation."""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from enum import Enum
from types import SimpleNamespace as NS
from unittest.mock import patch

BASELINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE))
import run
import session
import waypoints


class FakeRobot:
    def __init__(self):
        self.pose = {k: 0.0 for k in waypoints.JOINTS}
        self.sent = []

    def get_observation(self):
        return dict(self.pose)

    def send_action(self, action):
        self.sent.append(dict(action))
        self.pose = dict(action)
        return dict(action)


def hybrid_fixture():
    robot = FakeRobot()
    events = []
    shutdown = threading.Event()
    engine = NS(pause=lambda: events.append("paused"))
    ctx = NS(runtime=NS(cfg=NS(duration=30), shutdown_event=shutdown),
             hardware=NS(robot_wrapper=robot), policy=NS(inference=engine))
    strategy = NS(run=lambda ctx: events.append(("policy_return", ctx.runtime.cfg.duration)))
    options = run.parser().parse_args(["run", "--port", "/dev/test"])
    options.grasp = BASELINE / "reference_waypoints/tape_grasp.json"
    hybrid = session.HybridStrategy(strategy, ctx, options)
    hybrid.controller = NS(stopped=False, failed=False, running=False,
                           _stop_requested=threading.Event(), _reset_requested=threading.Event(),
                           start=lambda: True, reset=lambda: events.append("reset"))
    hybrid.say = lambda message: None
    hybrid.trace = NS(phase="idle", stop_recording=lambda: None)
    return hybrid, robot, ctx, events


class LaunchTests(unittest.TestCase):
    def test_defaults_and_original_modes(self):
        for mode, steps, inference in (("sync", 30, "sync"), ("replan", 8, "sync"), ("prefetch", 30, "rtc")):
            args = run.parser().parse_args(["run", "--port", "/dev/test", "--mode", mode])
            argv = run.rollout_args(args)
            self.assertIn(f"--policy.n_action_steps={steps}", argv)
            self.assertIn(f"--inference.type={inference}", argv)
            self.assertIn("--robot.max_relative_target=10", argv)
            self.assertIn("--interactive=true", argv)
            if mode == "prefetch":
                self.assertIn("--inference.rtc.enabled=false", argv)

    def test_dry_runs_without_dependencies_or_hardware(self):
        for argv in (("setup", "--dry-run"), ("run", "--port", "/dev/test", "--dry-run")):
            result = subprocess.run([sys.executable, str(BASELINE / "run.py"), *argv],
                                    capture_output=True, text=True, check=True)
            self.assertIn("--requirement" if argv[0] == "setup" else "Dry run", result.stdout)

    def test_setup_plan_uses_conda_without_venv_or_uv(self):
        result = subprocess.run([sys.executable, str(BASELINE / "run.py"), "setup", "--dry-run",
                                 "--env-name", "lerobot-baseline"],
                                capture_output=True, text=True, check=True)
        self.assertIn("conda create --yes --name lerobot-baseline", result.stdout)
        self.assertIn("conda run --no-capture-output --name lerobot-baseline python -m pip install", result.stdout)
        self.assertIn("requirements-lock.txt", result.stdout)
        self.assertNotIn("uv sync", result.stdout)
        self.assertNotIn(".venv", result.stdout)

    def test_inactive_environment_uses_conda_run_and_preserves_arguments(self):
        prefix = Path("/tmp/conda-envs/lerobot")
        with patch.object(run, "conda_prefix", return_value=prefix), \
             patch.object(run, "conda_executable", return_value="/opt/conda/bin/conda"), \
             patch.object(sys, "argv", ["run.py", "run", "--task", "Pick up the white block."]), \
             patch.object(run.os, "execv") as execute:
            run.managed_environment()
        command = execute.call_args.args[1]
        self.assertEqual(command[:7], ["/opt/conda/bin/conda", "run", "--no-capture-output",
                                       "--name", "lerobot", "python", str(BASELINE / "run.py")])
        self.assertEqual(command[-3:], ["run", "--task", "Pick up the white block."])

    def test_active_environment_is_not_reentered(self):
        prefix = Path(sys.prefix)
        with patch.object(run, "conda_prefix", return_value=prefix), \
             patch.dict(os.environ, {"CONDA_PREFIX": str(prefix)}), \
             patch.object(run.os, "execv") as execute:
            run.managed_environment()
        execute.assert_not_called()

    def test_setup_preserves_existing_packages_and_source_without_installing(self):
        with patch.object(run, "conda_prefix", return_value=Path("/tmp/envs/lerobot")), \
             patch.object(run, "conda_executable", return_value="conda"), \
             patch.object(run, "record_environment") as record, \
             patch.object(run, "shell") as shell, \
             patch.object(run.subprocess, "run") as process:
            run.setup()
        record.assert_called_once_with("lerobot", Path("/tmp/envs/lerobot"), "preserved")
        shell.assert_not_called()
        process.assert_not_called()

    def test_missing_environment_installs_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(run, "LEROBOT", Path(directory)), \
             patch.object(run, "conda_prefix", side_effect=[None, Path("/tmp/envs/lerobot")]), \
             patch.object(run, "conda_executable", return_value="conda"), \
             patch.object(run.shutil, "which", return_value="git"), \
             patch.object(run, "shell", return_value=NS(stdout=run.SOURCES["lerobot"])) as shell, \
             patch.object(run.subprocess, "run", return_value=NS(returncode=0)), \
             patch.object(run, "record_environment") as record:
            run.setup()
        commands = [call.args[0] for call in shell.call_args_list]
        self.assertTrue(any("create" in cmd for cmd in commands))
        self.assertTrue(any("install" in cmd for cmd in commands))
        record.assert_called_once_with("lerobot", Path("/tmp/envs/lerobot"), "reconstructed")

    def test_preserved_environment_keeps_existing_checkpoints_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            base, model = Path(directory) / "upstream", Path(directory) / "prepared"
            for folder in (base, model):
                folder.mkdir()
                (folder / "config.json").write_text('{"original": true}')
                (folder / "model.safetensors").touch()
            with patch.object(run, "BASE", base), patch.object(run, "MODEL", model), \
                 patch.object(run, "saved_runtime", return_value={"kind": "preserved"}), \
                 patch.object(run, "migrate") as migrate:
                run.prepare(NS(env_name="lerobot"))
            migrate.assert_not_called()
            self.assertEqual((model / "config.json").read_text(), '{"original": true}')

    def test_migration_preserves_joint_transform(self):
        source = dict(joint_signs=[1, -1, 1, 1, 1, 1], joint_offsets=[0, 90, 90, 0, 0, 0],
                      model_dtype="float32", enable_lora_vlm=False)
        config = run.migrate(source)
        self.assertEqual(config["joint_signs"], source["joint_signs"])
        self.assertEqual(config["joint_offsets"], source["joint_offsets"])
        self.assertNotIn("model_dtype", config)
        self.assertIn("model_dtype", source)  # Don't mutate the source config.


class WaypointTests(unittest.TestCase):
    def test_reject_bad_late_pose_before_any_connection(self):
        payload = json.loads((BASELINE / "reference_waypoints/tape_grasp.json").read_text())
        for invalid in (float("nan"), float("inf"), True, "10"):
            payload["waypoints"][-1]["joints"]["gripper.pos"] = invalid
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "grasp.json"
                path.write_text(json.dumps(payload))
                with patch.object(waypoints, "make_robot") as connect:
                    with self.assertRaises(ValueError):
                        waypoints.replay(NS(file=path, execute=True))
                    connect.assert_not_called()

    def test_interpolation_bounds_and_cancellation(self):
        for distance in (0, 0.1, 0.8, 1.01):
            robot = FakeRobot()
            goal = {k: distance for k in waypoints.JOINTS}
            with patch.object(waypoints.time, "sleep"):
                waypoints.move(robot, goal, step=0.4)
            previous = {k: 0 for k in waypoints.JOINTS}
            for action in robot.sent:
                self.assertTrue(all(abs(action[k] - previous[k]) <= 0.4 + 1e-9 for k in previous))
                previous = action
            self.assertEqual(robot.sent[-1], goal)
        robot = FakeRobot()
        with self.assertRaises(InterruptedError):
            waypoints.move(robot, goal, cancelled=lambda: True)
        self.assertEqual(robot.sent, [])

    def test_original_commit_forces_close_without_changing_file(self):
        path = BASELINE / "reference_waypoints/tape_grasp.json"
        before = path.read_bytes()
        stages = waypoints.commit_sequence(path)
        self.assertEqual(len(stages), 3)
        self.assertEqual(stages[1]["joints"]["gripper.pos"], 0)
        self.assertEqual(stages[2]["joints"]["gripper.pos"], 0)
        self.assertEqual(path.read_bytes(), before)


class HandoffTests(unittest.TestCase):
    def test_primitive_targets_and_pauses_match_historical_commit(self):
        """Compare against the actual old code, including a robot that misses targets."""
        source = subprocess.check_output(
            ["git", "show", "9cb398d:scripts/commit_session.py"], cwd=BASELINE.parent, text=True)
        original = types.ModuleType("original_commit_session")
        exec(compile(source, "9cb398d:scripts/commit_session.py", "exec"), original.__dict__)

        class LaggingRobot(FakeRobot):
            reads = 0

            def get_observation(self):
                self.reads += 1
                return {k: -2.0 for k in waypoints.JOINTS}

        old_robot, new_robot = LaggingRobot(), LaggingRobot()
        old_ctx = NS(hardware=NS(robot_wrapper=old_robot))
        old_ctrl = NS(stopped=False, start=lambda: True, _running=threading.Event(),
                      _start_requested=threading.Event(), _segment_stop=threading.Event(), _wake=threading.Event())
        old_session = NS(controller=old_ctrl, _ctx=old_ctx, _print=lambda message: None)
        path = BASELINE / "reference_waypoints/tape_grasp.json"
        old_pauses, new_pauses = [], []
        env = dict(COMMIT_GRASP=str(path), COMMIT_CLOSED="0", COMMIT_DIVE_STEP_DEG="0.4", COMMIT_DIVE_FPS="15")
        with patch.dict(os.environ, env), patch.object(original.time, "sleep", side_effect=old_pauses.append):
            original._run_commit(old_session, want_reset=False, seconds=0, record=False)
        hybrid, _, ctx, _ = hybrid_fixture()
        ctx.hardware.robot_wrapper = new_robot
        hybrid.pending = (15, False, False, waypoints.commit_sequence(path))

        def wait(seconds, observe=False):
            if seconds:
                new_pauses.append(seconds)

        with patch.object(hybrid, "wait", side_effect=wait), \
             patch.object(session.time, "sleep", side_effect=new_pauses.append):
            hybrid.run(ctx)
        self.assertEqual(new_robot.sent, old_robot.sent)
        self.assertEqual(new_pauses, old_pauses)
        self.assertEqual(old_robot.reads, 1)
        self.assertEqual(new_robot.reads, 1)

    def test_original_move_rounds_step_count_down(self):
        robot = FakeRobot()
        start = {k: 0.0 for k in waypoints.JOINTS}
        goal = {k: 1.01 for k in waypoints.JOINTS}
        with patch.object(session.time, "sleep"):
            result = session.move(robot, start, goal, step_deg=0.4, fps=15)
        self.assertEqual(len(robot.sent), 2)  # Same floor rounding as historical _move.
        self.assertEqual(result, goal)

    def test_policy_returns_and_pauses_before_primitive_then_resets(self):
        hybrid, robot, ctx, events = hybrid_fixture()
        hybrid.request(NS(args="20 reset"))
        def move(*args, **kwargs):
            self.assertEqual(events[:2], [("policy_return", 20), "paused"])
            events.append("primitive")
            return dict(args[2])
        with patch.object(session, "move", side_effect=move), patch.object(hybrid, "wait"):
            hybrid.run(ctx)
        self.assertEqual(events[-1], "reset")
        self.assertEqual(ctx.runtime.cfg.duration, 30)

    def test_reset_during_coarse_never_starts_primitives(self):
        hybrid, robot, ctx, events = hybrid_fixture()
        hybrid.strategy.run = lambda ctx: hybrid.controller._reset_requested.set()
        hybrid.request(NS(args=""))
        with patch.object(session, "move") as move:
            hybrid.run(ctx)
            move.assert_not_called()
        self.assertEqual(robot.sent, [])
        self.assertEqual(ctx.runtime.cfg.duration, 30)

    def test_stop_during_clamp_prevents_squeeze_and_lift(self):
        hybrid, robot, ctx, events = hybrid_fixture()
        hybrid.request(NS(args=""))
        def move(*args, **kwargs):
            if hybrid.trace.phase == "clamp":
                hybrid.controller._stop_requested.set()
            return dict(args[2])
        with patch.object(session, "move", side_effect=move) as motion:
            hybrid.run(ctx)
        self.assertEqual(motion.call_count, 2)
        self.assertEqual(robot.sent, [])

    def test_overlapping_commit_does_not_replace_request(self):
        hybrid, _, _, _ = hybrid_fixture()
        hybrid.request(NS(args="10"))
        first = hybrid.pending
        hybrid.request(NS(args="20"))
        self.assertIs(hybrid.pending, first)
        self.assertEqual(first[0], 10)

    def test_invalid_command_does_not_start(self):
        for value in ("nan", "-1", "inf", "20 unexpected", "0"):
            hybrid, _, _, _ = hybrid_fixture()
            with patch.object(hybrid.controller, "start") as start:
                hybrid.request(NS(args=value))
                start.assert_not_called()

    def test_trace_preserves_send_result_and_restores_methods(self):
        robot = FakeRobot()
        with tempfile.TemporaryDirectory() as directory:
            original_observe, original_send = robot.get_observation, robot.send_action
            trace = session.Trace(robot, Path(directory))
            robot.get_observation()
            goal = {k: 1 for k in waypoints.JOINTS}
            self.assertEqual(robot.send_action(goal), goal)
            trace.close()
            self.assertEqual(robot.get_observation, original_observe)
            self.assertEqual(robot.send_action, original_send)
            record = json.loads((Path(directory) / "actions.jsonl").read_text())
            self.assertEqual(record["requested"], goal)
            self.assertEqual(record["measured_latest"], {k: 0 for k in waypoints.JOINTS})

    def test_pinned_controller_integration(self):
        # After setup this exercises the real pinned controller with fake I/O.
        source = Path(os.environ.get("SO101_CONTROLLER_SOURCE", str(run.LEROBOT / "src/lerobot/rollout")))
        if not (source / "controller.py").exists():
            self.skipTest("Run setup to obtain the pinned controller source")
        package = types.ModuleType("pinned_rollout")
        package.__path__ = [str(source)]
        inference = types.ModuleType("pinned_rollout.inference")
        inference.QueryAnswer = NS
        inference.QueryKind = Enum("QueryKind", "VQA NEXT_SUBTASK")
        with patch.dict(sys.modules, {"pinned_rollout": package, inference.__name__: inference}):
            spec = importlib.util.spec_from_file_location("pinned_rollout.controller", source / "controller.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for cancel in (False, True):
                hybrid, robot, ctx, events = hybrid_fixture()
                ctx.runtime.shutdown_event = module.LinkedEvent(threading.Event())
                ctx.runtime.cfg.autosteer_interval_s = 1
                engine = ctx.policy.inference
                engine.failed, engine.task = False, "pick"
                for method in ("set_answer_observer", "stop_autosteer", "drop_pending_query",
                               "drop_ready_subtask_answers", "pump_query"):
                    setattr(engine, method, lambda *args: None)
                hybrid.strategy.config = NS(supports_interactive=True)
                hybrid.strategy.reset_control_state = lambda: None

                def policy(ctx):
                    events.append(("coarse", threading.get_ident()))
                    if cancel:
                        controller.stop()

                hybrid.strategy.run = policy

                def on_event(event, payload):
                    if event is module.RolloutEvent.SEGMENT_ENDED:
                        controller.stop()

                controller = module.RolloutController(hybrid, ctx, on_event=on_event)
                hybrid.controller = controller
                hybrid.request(NS(args="1"))

                def primitive(*args, **kwargs):
                    self.assertIn("paused", events)
                    self.assertEqual(events[0][1], threading.get_ident())
                    events.append("primitive")
                    return dict(args[2])

                with patch.object(session, "move", side_effect=primitive), patch.object(hybrid, "wait"):
                    controller.serve()
                self.assertTrue(controller.stopped)
                self.assertFalse(controller.failed)
                self.assertEqual(events.count("primitive"), 0 if cancel else 4)


if __name__ == "__main__":
    unittest.main()
