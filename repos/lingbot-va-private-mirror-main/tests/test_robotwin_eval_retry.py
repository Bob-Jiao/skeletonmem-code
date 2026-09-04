import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = REPO_ROOT / "evaluation/robotwin/check_eval_completeness.py"
RUNNER_PATH = REPO_ROOT / "evaluation/robotwin/run_local_eval.sh"
WORKER_PATH = REPO_ROOT / "evaluation/robotwin/_run_local_client_worker.sh"


def load_checker_module():
    spec = importlib.util.spec_from_file_location(
        "robotwin_eval_completeness", CHECKER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_result(root: Path, task: str, total: int, success: int) -> None:
    result_path = root / "stseed-10000/metrics" / task / "res.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "succ_num": float(success),
                "total_num": float(total),
                "succ_rate": success / total,
            }
        ),
        encoding="utf-8",
    )


def available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_fake_launchers(tmp_path: Path) -> tuple[Path, Path]:
    server = tmp_path / "fake_server.py"
    server.write_text(
        """#!/usr/bin/env python3
import select
import socket
import sys

listeners = []
for raw_port in (sys.argv[3], sys.argv[4]):
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", int(raw_port)))
    listener.listen()
    listeners.append(listener)

while True:
    readable, _, _ = select.select(listeners, [], [], 1.0)
    for listener in readable:
        connection, _ = listener.accept()
        connection.close()
""",
        encoding="utf-8",
    )
    server.chmod(0o755)

    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        """#!/usr/bin/env python3
import json
from pathlib import Path
import sys

_, _, save_root, raw_test_num, raw_seed, *tasks = sys.argv[1:]
test_num = int(raw_test_num)
start_seed = 10000 * (1 + int(raw_seed))
save_root = Path(save_root)
attempt_root = save_root / ".fake_attempts"
attempt_root.mkdir(parents=True, exist_ok=True)
failed = False

for task in tasks:
    attempt_path = attempt_root / task
    attempt = int(attempt_path.read_text() or "0") + 1 if attempt_path.exists() else 1
    attempt_path.write_text(str(attempt))
    total = test_num
    if task == "never_complete" or (task == "retry_me" and attempt == 1):
        total = test_num - 1
        failed = True
    result_path = save_root / f"stseed-{start_seed}" / "metrics" / task / "res.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({
        "succ_num": float(total),
        "total_num": float(total),
        "succ_rate": 1.0,
    }))

raise SystemExit(1 if failed else 0)
""",
        encoding="utf-8",
    )
    worker.chmod(0o755)
    return server, worker


def test_checker_rejects_partial_missing_and_inconsistent_results(tmp_path):
    checker = load_checker_module()
    write_result(tmp_path, "complete", total=3, success=2)
    write_result(tmp_path, "partial", total=2, success=1)
    write_result(tmp_path, "bad_rate", total=3, success=2)
    bad_rate = tmp_path / "stseed-10000/metrics/bad_rate/res.json"
    payload = json.loads(bad_rate.read_text())
    payload["succ_rate"] = 0.1
    bad_rate.write_text(json.dumps(payload))

    report = checker.build_report(
        tmp_path.resolve(),
        test_num=3,
        seed=0,
        tasks=["complete", "partial", "bad_rate", "missing"],
    )

    assert report["complete_task_count"] == 1
    assert report["incomplete_task_count"] == 3
    assert report["tasks"]["complete"]["complete"] is True
    assert "expected exactly 3" in report["tasks"]["partial"]["reason"]
    assert "inconsistent" in report["tasks"]["bad_rate"]["reason"]
    assert report["tasks"]["missing"]["reason"] == "missing res.json"


def test_runner_retries_only_incomplete_tasks_until_complete(tmp_path):
    server, worker = make_fake_launchers(tmp_path)
    transformer = tmp_path / "transformer"
    save_root = tmp_path / "results"
    transformer.mkdir()
    policy_port = available_port()
    master_port = available_port()
    while master_port == policy_port:
        master_port = available_port()

    env = os.environ.copy()
    env.update(
        CHECK_PYTHON=sys.executable,
        GPU_IDS="0",
        MAX_RETRY_ROUNDS="2",
        ROBOTWIN_SERVER_LAUNCHER=str(server),
        ROBOTWIN_WORKER_LAUNCHER=str(worker),
        SAVE_VIDEOS="0",
        SEED="0",
        SERVER_TIMEOUT="10",
        START_MASTER_PORT=str(master_port),
        START_PORT=str(policy_port),
        TASKS="complete_now retry_me",
        TEST_NUM="3",
    )
    completed = subprocess.run(
        ["bash", str(RUNNER_PATH), str(transformer), str(save_root)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    attempts = save_root / ".fake_attempts"
    assert (attempts / "complete_now").read_text() == "1"
    assert (attempts / "retry_me").read_text() == "2"
    report = json.loads((save_root / "completeness.json").read_text())
    assert report["complete"] is True
    assert report["complete_task_count"] == 2
    assert (save_root / "logs/client_round00_worker0_gpu0.log").is_file()
    assert (save_root / "logs/client_round01_worker0_gpu0.log").is_file()


def test_runner_stops_after_configured_retry_limit(tmp_path):
    server, worker = make_fake_launchers(tmp_path)
    transformer = tmp_path / "transformer"
    save_root = tmp_path / "results"
    transformer.mkdir()
    policy_port = available_port()
    master_port = available_port()
    while master_port == policy_port:
        master_port = available_port()

    env = os.environ.copy()
    env.update(
        CHECK_PYTHON=sys.executable,
        GPU_IDS="0",
        MAX_RETRY_ROUNDS="1",
        ROBOTWIN_SERVER_LAUNCHER=str(server),
        ROBOTWIN_WORKER_LAUNCHER=str(worker),
        SAVE_VIDEOS="0",
        SEED="0",
        SERVER_TIMEOUT="10",
        START_MASTER_PORT=str(master_port),
        START_PORT=str(policy_port),
        TASKS="never_complete",
        TEST_NUM="3",
    )
    completed = subprocess.run(
        ["bash", str(RUNNER_PATH), str(transformer), str(save_root)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 1
    assert "Exhausted 1 automatic retry round" in completed.stderr
    assert (save_root / ".fake_attempts/never_complete").read_text() == "2"
    report = json.loads((save_root / "completeness.json").read_text())
    assert report["complete"] is False
    assert report["incomplete_task_count"] == 1


def test_runner_skips_previously_complete_tasks_without_starting_server(tmp_path):
    transformer = tmp_path / "transformer"
    save_root = tmp_path / "results"
    transformer.mkdir()
    write_result(save_root, "already_done", total=3, success=2)

    env = os.environ.copy()
    env.update(
        CHECK_PYTHON=sys.executable,
        GPU_IDS="0",
        ROBOTWIN_SERVER_LAUNCHER="/usr/bin/false",
        ROBOTWIN_WORKER_LAUNCHER="/usr/bin/false",
        SAVE_VIDEOS="0",
        SEED="0",
        TASKS="already_done",
        TEST_NUM="3",
    )
    completed = subprocess.run(
        ["bash", str(RUNNER_PATH), str(transformer), str(save_root)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "selected tasks were already complete" in completed.stdout
    assert not list((save_root / "logs").glob("server_*.log"))


def test_worker_continues_after_one_task_process_fails(tmp_path):
    task_log = tmp_path / "tasks.log"
    client = tmp_path / "fake_client.py"
    client.write_text(
        """#!/usr/bin/env python3
import os
from pathlib import Path
import sys

task = sys.argv[1]
with Path(os.environ["TASK_LOG"]).open("a", encoding="utf-8") as handle:
    handle.write(f"{task}\\n")
raise SystemExit(1 if task == "fails" else 0)
""",
        encoding="utf-8",
    )
    client.chmod(0o755)

    env = os.environ.copy()
    env.update(
        ROBOTWIN_CLIENT_LAUNCHER=str(client),
        TASK_LOG=str(task_log),
    )
    completed = subprocess.run(
        [
            "bash",
            str(WORKER_PATH),
            "0",
            "29556",
            str(tmp_path / "results"),
            "3",
            "0",
            "fails",
            "still_runs",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 1
    assert task_log.read_text(encoding="utf-8").splitlines() == [
        "fails",
        "still_runs",
    ]
    assert "continuing" in completed.stderr
