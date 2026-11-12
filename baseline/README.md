# MolmoAct-2 coarse-to-fine pick experiment

**The model approached the object but did not complete the grasp.**

The arm repeatedly moved up and down above the object without picking it up.
This motivated a coarse-to-fine handoff: run MolmoAct-2 for the approach, then
stop the policy and execute hand-taught lower, clamp, and lift poses.

The handoff uses a timer (15 seconds by default), not alignment or contact
feedback. It requires fixed arm-base and object placement. The demo shows the
hybrid approach; the cause of the model's stalled grasp remains unresolved.

![Original experiment demo, played at 2× speed](assets/pick-demo.gif)

## Related reports and possible causes

Related failures have been reported elsewhere. These sources suggest checks,
but do not diagnose our repeated up/down motion:

| Source | Finding and relevance |
|---|---|
| [Independent MolmoAct-2 SO-101 evaluation](https://huggingface.co/datasets/abdul004/molmoact2-so101-zero-shot-eval) | Reports initial movement followed by stalling or near-current-pose targets, even after removing a client clamp. Similar failure to progress, in a different setup. |
| [Official MolmoAct-2 deployment guide](https://huggingface.co/docs/lerobot/molmoact2#joint-frame-transform-so-100101-zero-shot) | Documents a joint-frame mismatch with newer LeRobot calibration. The converted checkpoint already corrects it; preserve its processors and check camera mapping. |
| [ACT paper, §V-A](https://roboticsproceedings.org/rss19/p016.pdf) | Abrupt action-chunk transitions can cause jerky motion—a possible contributor to investigate. |
| [LeRobot RTC guide](https://huggingface.co/docs/lerobot/rtc) | Slow inference and disagreeing chunks can cause pauses and strategy changes. Compare command timing across the preserved inference modes. |

**Another hypothesis to test:** `replan` mode executes only the first 8 of 30
predicted actions. If closure occurs later in each chunk, repeated replanning
could discard it. Compare full predictions with executed targets; this has not
been established. The original chunk-analysis script is recoverable from Git
(see below).

## Setup and run

The original setup used an Apple Silicon Mac with 48 GB memory, an SO-101 follower,
and two USB cameras. Follow [SO-101 hardware setup](https://huggingface.co/docs/lerobot/so101)
for assembly and motor configuration. Allow terminal/Python camera access on macOS.
Close other robot/camera programs before running.

**Support the arm during teaching and disconnection: torque releases.**

With Homebrew installed, run these commands from the repository root:

```bash
brew install python@3.12 git ffmpeg
python3.12 baseline/run.py
```

The guide installs an isolated environment, asks for the port and robot ID, offers
calibration, downloads/prepares the model, previews both cameras in a local browser,
and teaches grasp poses if needed. Enter captures a pose; `q` cancels teaching.
It asks before opening the model session.

Alternatively, run each stage explicitly:

```bash
# Install the pinned environment and prepare model weights.
python3.12 baseline/run.py setup
python3.12 baseline/run.py prepare

# Find the follower port; replace YOUR_PORT below.
ls /dev/tty.usbmodem*
python3.12 baseline/run.py calibrate --port /dev/tty.usbmodemYOUR_PORT

# Verify cam0=top, cam1=wrist; press Enter in the terminal to close preview.
python3.12 baseline/run.py cameras --top 0 --wrist 1

# Teach approach, lower, close, lift; then validate/print without moving.
python3.12 baseline/run.py teach --port /dev/tty.usbmodemYOUR_PORT
python3.12 baseline/run.py replay

# Preflight runs automatically before model loading and connection.
python3.12 baseline/run.py run --port /dev/tty.usbmodemYOUR_PORT
```

Use the same `--id` for calibration, teaching and running (default
`my_awesome_follower_arm`). Skip calibration if it already exists for that arm/ID.
After setup, the launcher selects its managed Python automatically.

Waypoints default to `outputs/waypoints/grasp.json`. Teaching never overwrites a
file; use `--grasp outputs/waypoints/another_grasp.json` consistently for a new
sequence. `replay --execute` physically replays every saved pose; inspect the path
first. Historical files in `reference_waypoints/` belong to the old arm/scene and
must be reviewed before reuse.

## Interactive commands

The arm starts idle after loading and connecting.

| Command | Behavior |
|---|---|
| `/commit` or `/commit 20` | Run policy for 15 or 20 seconds, then lower/clamp/lift and hold. |
| `/commit reset` | Commit, then return to the session's initial pose. |
| `/commitandrecord` | Commit plus top/wrist MP4s through five seconds of hold; accepts seconds/reset. |
| `/start` | Run the policy alone for 30 seconds by default, to compare its behavior. |
| `/subtask <text>` | Change the model's instruction. |
| `/reset` | Interrupt motion and return to the pose captured at session startup. |
| `/stop` | End the session, disconnect, release torque. |
| `/help` | Show available commands. |

Wait for the current run to finish or reset before another commit. Reset/stop
requests are checked between primitive commands; blocking inference or hardware
calls can delay them.

Defaults preserve `run_molmo.sh`: sync inference, 30 actions per observation,
MPS bf16, 640×480 cameras at requested 30 fps, joint degrees, relative target limit
10, and the white-block task. `--mode prefetch` queues chunks with RTC guidance
disabled; `--mode replan` executes eight actions per observation.

Use `--help` for options, including task, cameras, duration and handoff timing.
Clamp/lift retain the original forced gripper target of **0**; change it with
`--closed-gripper` in native gripper units. Standalone replay preserves taught
gripper targets.

## Reproducibility and retained files

Three core files keep setup and control separate behind one entry point:

- `run.py`: environment/model setup, calibration, cameras, preflight and launch.
- `session.py`: interactive handoff, recording and action traces.
- `waypoints.py`: pose capture, validation, replay and motion helpers.

[`sources.json`](sources.json) pins the model revisions, uv, Python 3.12.12 and a
LeRobot snapshot from 2026-09-26. Setup uses that snapshot's `uv.lock` and the
[MPS loading patch](patches/lerobot-mps-loader.patch). The original LeRobot revision
and complete environment were not recorded, so these pins reconstruct the setup.
**This refactor has not been validated with model inference or physical grasping.**
CUDA/CPU adaptations are also unvalidated; rerun `prepare --device ... --dtype ...`
before switching backend.

The refactor runs primitives on the controller thread after the policy loop
returns, rejects overlapping commits, and checks reset/stop requests. It validates
poses, re-reads joints per segment, and reuses observations for recording.
The timed handoff, forced closure, settling/re-squeeze, and hold are retained.

Sessions save source/config snapshots, environment details, launch options,
action traces and camera snapshots (up to 2 Hz) in `outputs/runs/<timestamp>/`.
`/commitandrecord` adds MP4s sampled at up to 10 fps. Dropped frames can compress
playback time; `frames.jsonl` records capture times. In prefetch mode, logged
observation age measures dispatch state, not the model's input age.

Models, environments and outputs are ignored by Git; back them up along with
robot calibration files. The original GIF, waypoint files and loading patch are
retained here. Superseded scripts remain at commit `9cb398d`, for example:

```bash
git show 9cb398d:scripts/analyze_molmo_chunks.py
```

Hardware-free checks (the controller integration test requires setup first):

```bash
python3.12 baseline/run.py setup --dry-run
python3.12 baseline/run.py run --port /dev/example --dry-run
python3.12 -m unittest discover -s baseline/tests -v
```
