"""Guided setup and single entry point for the original MolmoAct-2 pick experiment."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from waypoints import commit_sequence, positive, nonnegative

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCES = json.loads((HERE / "sources.json").read_text())
LEROBOT = ROOT / "lerobot"
REQUIREMENTS = HERE / "requirements-lock.txt"
RUNTIMES = ROOT / ".cache/baseline-environments"
BASE = ROOT / "models/molmoact2-upstream"
MODEL = ROOT / "models/molmoact2-mps"
GRASP = ROOT / "outputs/waypoints/grasp.json"


def shell(command, cwd=ROOT, capture=False):
    print("+ " + shlex.join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), cwd=cwd, check=True,
                          text=True, capture_output=capture)


def conda_executable():
    executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not executable:
        raise ValueError("Install Conda/Miniforge first; see baseline/README.md.")
    return executable


def conda_prefix(name, required=True):
    result = shell([conda_executable(), "env", "list", "--json"], capture=True)
    matches = [Path(p) for p in json.loads(result.stdout)["envs"] if Path(p).name == name]
    if len(matches) > 1:
        raise ValueError(f"Multiple Conda environments named {name!r}; use a unique --env-name.")
    if matches:
        return matches[0]
    if required:
        raise ValueError(f"Missing Conda environment {name!r}; run the setup command first.")
    return None


def conda_command(name, *command):
    return [conda_executable(), "run", "--no-capture-output", "--name", name, *command]


def snapshot_environment(directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for filename, command in (
        ("baseline-environment.txt", conda_command(name, "python", "-m", "pip", "freeze")),
        ("baseline-environment.yml", [conda_executable(), "env", "export", "--name", name]),
        ("baseline-conda-explicit.txt", [conda_executable(), "list", "--name", name, "--explicit"]),
    ):
        (directory / filename).write_text(shell(command, capture=True).stdout)


def runtime_path(name):
    return RUNTIMES / f"{name}.json"


def record_environment(name, prefix, kind):
    directory = ROOT / "outputs/environments" / (name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    snapshot_environment(directory, name)
    probe = ("import sys,platform,importlib.util,json; s=importlib.util.find_spec('lerobot'); "
             "print(json.dumps(dict(python=sys.version,platform=platform.platform(),"
             "lerobot_file=s.origin if s else None)))")
    installed = json.loads(shell(conda_command(name, "python", "-c", probe), capture=True).stdout)
    if installed["lerobot_file"]:
        source = Path(installed["lerobot_file"]).parent
        revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                  text=True, capture_output=True)
        if revision.returncode == 0:
            installed["lerobot_git_commit"] = revision.stdout.strip()
            diff = shell(["git", "diff", "--binary", "HEAD"], cwd=source, capture=True).stdout
            (directory / "lerobot-local-changes.patch").write_text(diff)
    (directory / "installed-runtime.json").write_text(json.dumps(installed, indent=2) + "\n")
    runtime = dict(name=name, prefix=str(prefix), kind=kind, snapshot=str(directory), installed=installed)
    if kind == "reconstructed":
        runtime["requirements_sha256"] = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()
    RUNTIMES.mkdir(parents=True, exist_ok=True)
    runtime_path(name).write_text(json.dumps(runtime, indent=2) + "\n")
    print(f"Saved environment: {directory}")
    return runtime


def setup(dry_run=False, env_name="lerobot"):
    """Preserve an existing Conda environment; reconstruct dependencies only if missing."""
    conda = "conda" if dry_run else conda_executable()
    create = [conda, "create", "--yes", "--name", env_name, "--override-channels",
              "--channel", "conda-forge", f"python={SOURCES['python']}", "pip", "ffmpeg"]
    commands = [
        ["git", "init", LEROBOT],
        ["git", "-C", LEROBOT, "remote", "add", "origin", "https://github.com/huggingface/lerobot.git"],
        ["git", "-C", LEROBOT, "fetch", "--depth=1", "origin", SOURCES["lerobot"]],
        ["git", "-C", LEROBOT, "checkout", "--detach", "FETCH_HEAD"],
    ]
    install = [conda, "run", "--no-capture-output", "--name", env_name,
               "python", "-m", "pip", "install", "--requirement", REQUIREMENTS,
               "--editable", f"{LEROBOT}[molmoact2,feetech,core_scripts]"]
    if sys.platform == "linux":
        install += ["--extra-index-url", "https://download.pytorch.org/whl/cu128"]
    if dry_run:
        print("If the environment exists: export it and use installed packages unchanged; no install or source patch.")
        print("Only if missing, reconstruct an environment with:")
        print(shlex.join(map(str, create)))
        for command in commands:
            print(shlex.join(map(str, command)))
        print("Apply baseline/patches/lerobot-mps-loader.patch if not already applied")
        print(shlex.join(map(str, install)))
        return
    prefix = conda_prefix(env_name, required=False)
    if prefix is not None:
        record_environment(env_name, prefix, "preserved")
        print(f"Using existing Conda environment {env_name!r}; packages, Python and LeRobot source unchanged.")
        return
    if not shutil.which("git"):
        raise ValueError("Install Git first (macOS: brew install git), then retry.")
    shell(create)
    prefix = conda_prefix(env_name)
    if not LEROBOT.exists():
        for command in commands:
            shell(command)
    actual = shell(["git", "rev-parse", "HEAD"], cwd=LEROBOT, capture=True).stdout.strip()
    if actual != SOURCES["lerobot"]:
        raise ValueError(f"Existing {LEROBOT} is at {actual}. Move it aside; setup will not overwrite it.")
    patch = HERE / "patches/lerobot-mps-loader.patch"
    reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)],
                             cwd=LEROBOT, capture_output=True)
    if reverse.returncode:
        shell(["git", "apply", "--check", patch], cwd=LEROBOT)
        shell(["git", "apply", patch], cwd=LEROBOT)
    shell(install)
    record_environment(env_name, prefix, "reconstructed")
    print(f"Setup complete. Runtime: Conda environment {env_name!r}. Activate with: conda activate {env_name}")


def managed_environment(env_name="lerobot"):
    prefix = conda_prefix(env_name)
    if Path(sys.prefix).resolve() != prefix.resolve() or os.environ.get("CONDA_PREFIX") != str(prefix):
        command = list(map(str, conda_command(env_name, "python", HERE / "run.py", *sys.argv[1:])))
        os.execv(command[0], command)


def saved_runtime(name):
    path = runtime_path(name)
    if not path.exists():
        raise ValueError("Run setup first to record the Conda environment; existing packages will be preserved.")
    return json.loads(path.read_text())


def migrate(config, device="mps", dtype="bfloat16"):
    config = dict(config)
    if config.pop("enable_lora_action_expert", False):
        raise ValueError("Action-expert LoRA requires a separate migration")
    lora = config.pop("enable_lora_vlm", False)
    frozen = config.pop("train_action_expert_only", False)
    config["train_mode_vlm"] = "lora" if lora else ("freeze" if frozen else "fft")
    config.pop("model_dtype", None)
    config.update(device=device, dtype=dtype, checkpoint_path=str(BASE),
                  enable_inference_cuda_graph=False, compile_model=False)
    return config


def prepare(args):
    runtime = saved_runtime(args.env_name)
    if runtime["kind"] == "preserved" and all(
            (directory / "config.json").exists() and list(directory.glob("*.safetensors"))
            for directory in (BASE, MODEL)):
        print("Using existing local checkpoints and processors unchanged.")
        return
    from huggingface_hub import snapshot_download
    import draccus
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    for repo, directory in (("allenai/MolmoAct2-SO100_101", BASE),
                            ("lerobot/MolmoAct2-SO100_101-LeRobot", MODEL)):
        snapshot_download(repo, revision=SOURCES["models"][repo], local_dir=directory,
                          allow_patterns=["*.json", "*.safetensors", "*.jinja", "assets/*.png"],
                          max_workers=4)
    path = MODEL / "config.json"
    backup = MODEL / "config.original.json"
    if not backup.exists():
        backup.write_text(path.read_text())
    config = migrate(json.loads(backup.read_text()), args.device, args.dtype)
    draccus.decode(MolmoAct2Config, {k: v for k, v in config.items() if k != "type"})
    processor_path = MODEL / "policy_preprocessor.json"
    processor = json.loads(processor_path.read_text())
    for step in processor["steps"]:
        if step["registry_name"] == "molmoact2_pack_inputs":
            step["config"]["checkpoint_path"] = str(BASE)
        elif step["registry_name"] == "device_processor":
            step["config"]["device"] = args.device
    path.write_text(json.dumps(config, indent=2) + "\n")
    processor_path.write_text(json.dumps(processor, indent=2) + "\n")
    (MODEL / "source_revisions.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
    print(f"Prepared {MODEL}; the saved processors retain the joint-frame correction.")


def cameras(args):
    import cv2
    import threading
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    captures = []
    frames, errors = {}, []
    stop = threading.Event()
    server = worker = None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            key = self.path.split("?")[0].lstrip("/")
            if key in ("cam0", "cam1"):
                content, kind = frames.get(key, b""), "image/jpeg"
            elif not key:
                content = (b"<!doctype html><title>SO-101 camera check</title>"
                           b"<h2>cam0: top</h2><img id=cam0><h2>cam1: wrist</h2><img id=cam1>"
                           b"<script>setInterval(()=>{for(const k of ['cam0','cam1'])"
                           b"document.getElementById(k).src='/'+k+'?t='+Date.now()},300)</script>")
                kind = "text/html"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    def capture():
        try:
            while not stop.is_set():
                for cap, key in captures:
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f"Lost camera stream: {key}")
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if not ok:
                        raise ValueError(f"Cannot encode camera: {key}")
                    frames[key] = encoded.tobytes()
        except Exception as exc:
            errors.append(exc)
            print(f"Camera error: {exc}. Press Enter to close preview.", flush=True)

    try:
        for index, label in ((args.top, "cam0"), (args.wrist, "cam1")):
            cap = cv2.VideoCapture(index)
            captures.append((cap, label))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            if not cap.isOpened():
                raise ValueError(f"Cannot open {label} at index {index}")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        worker = threading.Thread(target=capture, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}"
        print(f"Camera check: {url} (cam0 top, cam1 wrist).", flush=True)
        webbrowser.open(url)
        input("Verify both views in your browser, then press Enter here to close preview: ")
        if errors:
            raise errors[0]
    finally:
        stop.set()
        if server:
            server.shutdown()
            server.server_close()
        if worker:
            worker.join(timeout=3)
        for cap, _ in captures:
            cap.release()


def rollout_args(args):
    cameras_config = {key: dict(type="opencv", index_or_path=index, width=640, height=480, fps=30)
                      for key, index in (("cam0", args.top), ("cam1", args.wrist))}
    inference = (["--inference.type=rtc", "--inference.rtc.enabled=false", "--inference.queue_threshold=30"]
                 if args.mode == "prefetch" else ["--inference.type=sync"])
    return ["--strategy.type=base", *inference,
            f"--policy.path={MODEL}", "--policy.inference_action_mode=continuous",
            f"--policy.device={args.device}", f"--policy.dtype={args.dtype}",
            "--policy.enable_inference_cuda_graph=false",
            f"--policy.n_action_steps={8 if args.mode == 'replan' else 30}",
            "--robot.type=so101_follower", f"--robot.port={args.port}", f"--robot.id={args.id}",
            "--robot.use_degrees=true", "--robot.max_relative_target=10",
            f"--robot.cameras={json.dumps(cameras_config)}", "--fps=30", f"--task={args.task}",
            f"--duration={args.duration}", "--interactive=true",
            "--return_to_initial_position=false", "--play_sounds=false"]


def check(args):
    import torch
    from lerobot.rollout.interactive import InteractiveSession
    from lerobot.scripts.lerobot_rollout import main as rollout_main
    from waypoints import load

    runtime = saved_runtime(args.env_name)
    if Path(runtime["prefix"]).resolve() != Path(sys.prefix).resolve():
        raise ValueError("Conda environment differs from setup; rerun setup with this --env-name.")
    if runtime["kind"] == "reconstructed":
        actual = shell(["git", "rev-parse", "HEAD"], cwd=LEROBOT, capture=True).stdout.strip()
        if (actual != SOURCES["lerobot"] or runtime["requirements_sha256"]
                != hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()):
            raise ValueError("Reconstruction source/dependency pins differ from the recorded setup.")

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable. The original experiment used an Apple Silicon Mac.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    if not Path(args.port).exists():
        raise ValueError(f"Arm port does not exist: {args.port}")
    for directory in (MODEL, BASE):
        if not (directory / "config.json").exists() or not list(directory.glob("*.safetensors")):
            raise ValueError(f"Incomplete checkpoint {directory}; run `prepare`.")
    if (runtime["kind"] == "reconstructed"
            and json.loads((MODEL / "source_revisions.json").read_text()) != SOURCES):
        raise ValueError("Checkpoint pins differ; run `prepare`.")
    config = json.loads((MODEL / "config.json").read_text())
    if config["device"] != args.device or config["dtype"] != args.dtype:
        raise ValueError("Device/dtype differ from prepared checkpoint; rerun `prepare` with these options.")
    payload = load(args.grasp)
    if payload.get("id") != args.id:
        raise ValueError("Waypoint robot ID differs. Re-teach for this calibrated arm.")
    commit_sequence(args.grasp, args.closed_gripper)
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS
    from lerobot.robots.so_follower import SO101Follower
    calibration = HF_LEROBOT_CALIBRATION / ROBOTS / SO101Follower.name / f"{args.id}.json"
    if not calibration.is_file():
        raise ValueError(f"Missing calibration {calibration}; run the calibrate command first.")
    if not callable(rollout_main) or not hasattr(InteractiveSession, "run"):
        raise ValueError("Incompatible LeRobot rollout API")
    print("Preflight passed. No robot connection or camera capture was opened.")
    print("Calibration file exists; live calibration consistency and model inference are checked on launch.")


def run(args):
    command = rollout_args(args)
    print("LeRobot options: " + shlex.join(command), flush=True)
    if args.dry_run:
        print("Dry run: setup, downloads, model loading, and hardware connection skipped.")
        return
    check(args)
    from lerobot.rollout.interactive import InteractiveSession
    from lerobot.scripts.lerobot_rollout import main as rollout_main
    from session import install

    directory = ROOT / "outputs/runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True)
    runtime = saved_runtime(args.env_name)
    manifest = dict(reconstruction_pins=SOURCES, runtime=runtime,
                    options={k: str(v) if isinstance(v, Path) else v
                                            for k, v in vars(args).items()}, argv=command,
                    python=sys.version, platform=platform.platform(),
                    conda_environment=args.env_name, conda_prefix=sys.prefix)
    manifest["packages"] = {k: importlib.metadata.version(k)
                            for k in ("lerobot", "torch", "transformers", "huggingface-hub")}
    manifest["files"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [*HERE.glob("*.py"), REQUIREMENTS, HERE / "sources.json", MODEL / "config.json",
                                   MODEL / "policy_preprocessor.json", args.grasp]}
    manifest["git_commit"] = shell(["git", "rev-parse", "HEAD"], capture=True).stdout.strip()
    manifest["git_dirty"] = bool(shell(["git", "status", "--porcelain"], capture=True).stdout)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(args.grasp, directory / "grasp.json")
    shutil.copy2(MODEL / "config.json", directory / "model_config.json")
    shutil.copy2(MODEL / "policy_preprocessor.json", directory / "policy_preprocessor.json")
    shutil.copytree(HERE, directory / "source", ignore=shutil.ignore_patterns("__pycache__", "assets", "tests"))
    shutil.copytree(runtime["snapshot"], directory / "environment")
    restore = install(InteractiveSession, args, directory)
    previous = sys.argv
    try:
        sys.argv = ["lerobot_rollout", *command]
        print(f"Run evidence: {directory}\n/commit, /commitandrecord, /reset, /start, /subtask, /stop", flush=True)
        rollout_main()
    finally:
        sys.argv = previous
        restore()


def wizard(args):
    if args.device == "mps" and (sys.platform != "darwin" or platform.machine() != "arm64"):
        raise ValueError("The original MPS experiment needs an Apple Silicon Mac. "
                         "This host cannot run that backend. See baseline/README.md; "
                         "CUDA/CPU adaptations require an explicit --device selection.")
    runtime_file = runtime_path(args.env_name)
    if not runtime_file.exists() or json.loads(runtime_file.read_text())["name"] != args.env_name:
        print("Step 1: preserve the existing Conda environment, or install a reconstruction if missing.")
        setup(env_name=args.env_name)
    managed_environment(args.env_name)
    print("Original experiment: Apple Silicon/MPS, calibrated SO-101, top + wrist cameras.")
    args.port = input("Follower serial port: ").strip()
    args.id = input(f"Robot calibration ID [{args.id}]: ").strip() or args.id
    args.task = input(f"Task [{args.task}]: ").strip() or args.task
    args.top = int(input("Top camera index [0]: ").strip() or "0")
    args.wrist = int(input("Wrist camera index [1]: ").strip() or "1")
    if args.top == args.wrist:
        raise ValueError("Camera indices must differ")
    if input("Is this arm already calibrated under that ID? [y/N]: ").lower() != "y":
        calibrate(args)
    print("Step 2: reuse existing checkpoints, or download/prepare them if missing.")
    prepare(args)
    print("Step 3: verify camera assignment in the browser, then Enter to finish.")
    cameras(args)
    print("Step 4: support the arm; teaching disables torque. Stop other robot programs.")
    if args.grasp.exists():
        answer = input(f"Reuse {args.grasp} with the same arm/base/object placement? [y/N]: ").lower()
        if answer != "y":
            raise ValueError("Choose a new --grasp path and rerun to teach without overwriting existing poses.")
    else:
        from waypoints import teach
        args.file, args.names = args.grasp, ["approach", "lower", "close", "lift"]
        teach(args)
    check(args)
    print("Step 5: model rollout plus taught grasp. Disconnect releases torque; support the arm.")
    if input("Ready to connect and enter the interactive session? [y/N]: ").lower() == "y":
        run(args)


def calibrate(args):
    shell([sys.executable, "-m", "lerobot.scripts.lerobot_calibrate",
           "--robot.type=so101_follower", f"--robot.port={args.port}", f"--robot.id={args.id}"])


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", nargs="?", default="wizard",
                    choices=("wizard", "setup", "prepare", "calibrate", "cameras", "teach", "replay", "check", "run"))
    ap.add_argument("--port")
    ap.add_argument("--env-name", default="lerobot", help="Conda environment name (default: lerobot)")
    ap.add_argument("--id", default="my_awesome_follower_arm")
    ap.add_argument("--task", default="Pick up the white block.")
    ap.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--wrist", type=int, default=1)
    ap.add_argument("--mode", choices=("sync", "prefetch", "replan"), default="sync")
    ap.add_argument("--duration", type=positive, default=30)
    ap.add_argument("--coarse-seconds", type=positive, default=15)
    ap.add_argument("--closed-gripper", type=nonnegative, default=0)
    ap.add_argument("--dive-step", type=positive, default=0.4)
    ap.add_argument("--dive-fps", type=positive, default=15)
    ap.add_argument("--grasp", type=Path, default=GRASP)
    ap.add_argument("--dry-run", action="store_true", help="setup plan or rollout options; no changes or hardware")
    ap.add_argument("--execute", action="store_true", help="execute waypoint replay (otherwise only validates)")
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    args.grasp = args.grasp.resolve()
    if args.env_name == "base" or not args.env_name or any(
            not (c.isalnum() or c in "-_") for c in args.env_name):
        ap.error("--env-name must be a named Conda environment using letters, digits, '-' or '_'; not base")
    if args.top == args.wrist:
        ap.error("--top and --wrist must differ")
    if args.command in ("check", "run", "teach", "calibrate") and not args.port:
        ap.error("--port is required for this command")
    if args.dry_run and args.command not in ("setup", "run"):
        ap.error("--dry-run is supported for setup and run; replay is dry by default")
    try:
        if args.command == "setup":
            setup(args.dry_run, args.env_name)
        elif args.command == "wizard":
            wizard(args)
        elif args.command == "run" and args.dry_run:
            run(args)
        else:
            if args.command != "replay" or args.execute:
                managed_environment(args.env_name)
            if args.command in ("teach", "replay"):
                from waypoints import teach, replay
                args.file = args.grasp
                args.names = ["approach", "lower", "close", "lift"]
                args.step, args.fps, args.pause, args.hold = 0.4, 15, 1, 5
                (teach if args.command == "teach" else replay)(args)
            else:
                {"prepare": prepare, "calibrate": calibrate, "cameras": cameras,
                 "check": check, "run": run}[args.command](args)
    except (ValueError, OSError, KeyError, ImportError, subprocess.CalledProcessError) as exc:
        ap.exit(1, f"Error: {exc}\n")
    except KeyboardInterrupt:
        ap.exit(130, "Interrupted. Support the arm when torque releases.\n")


if __name__ == "__main__":
    main()
