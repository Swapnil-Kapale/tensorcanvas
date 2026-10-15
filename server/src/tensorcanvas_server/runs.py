"""Training run lifecycle: a RunHandle per run (status, live event fan-out,
checkpoints, pause/stop), a registry, and the SSE stream generator.

Events published from the training thread are marshalled onto the asyncio loop
via ``loop.call_soon_threadsafe`` and fanned out to per-subscriber bounded
queues. Durable events (hello/epoch/status/done/error) are also kept in
``history`` for reconnect replay; sampled events (metrics/frame) are droppable.
"""

from __future__ import annotations

import asyncio
import copy
import json
import threading
import uuid
from typing import Optional

CHECKPOINT_KEEP = 60
DURABLE = {"hello", "epoch", "status", "done", "error"}


class RunHandle:
    def __init__(self, model_id: str, config, loop: asyncio.AbstractEventLoop):
        self.run_id = "run_" + uuid.uuid4().hex[:10]
        self.model_id = model_id
        self.config = config
        self.loop = loop
        self.status = "running"
        self.seq = 0
        self.history: list[dict] = []
        self.latest: dict = {}
        self.subs: list[asyncio.Queue] = []
        self.checkpoints: list[dict] = []
        self._pause = threading.Event()
        self._pause.set()  # set == running
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # ── publish / fan-out ────────────────────────────────────────────────
    def publish(self, etype: str, data: dict, droppable: bool = False) -> None:
        with self._lock:
            self.seq += 1
            frame = {"type": etype, "id": self.seq, "data": {"seq": self.seq, **data}}
            if etype in DURABLE:
                self.history.append(frame)
                if etype == "epoch":
                    self.latest = frame["data"]
        try:
            self.loop.call_soon_threadsafe(self._fanout, frame, droppable)
        except RuntimeError:
            pass  # loop gone (server shutting down)

    def _fanout(self, frame: dict, droppable: bool) -> None:
        for q in list(self.subs):
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                if not droppable:  # terminal/durable: evict oldest, then enqueue
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        q.put_nowait(frame)
                    except asyncio.QueueFull:
                        pass

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self.subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if q in self.subs:
            self.subs.remove(q)

    # ── control ──────────────────────────────────────────────────────────
    def stopped(self) -> bool:
        return self._stop.is_set()

    def wait_if_paused(self) -> None:
        self._pause.wait()

    def pause(self) -> None:
        if self.status == "running":
            self._pause.clear()
            self.note_status("paused")

    def resume(self) -> None:
        if self.status == "paused":
            self._pause.set()
            self.note_status("running")

    def stop(self) -> None:
        self._stop.set()
        self._pause.set()

    def note_status(self, s: str) -> None:
        self.status = s
        self.publish("status", {"status": s})

    # ── checkpoints ──────────────────────────────────────────────────────
    def checkpoint(self, module, epoch: int, val_loss: float, val_acc: Optional[float]) -> str:
        ckpt_id = f"ck{epoch}"
        state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        best = not self.checkpoints or val_loss <= min(c["val_loss"] for c in self.checkpoints)
        self.checkpoints.append(
            {"ckpt_id": ckpt_id, "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc,
             "is_best": best, "state": state}
        )
        for c in self.checkpoints:
            c["is_best"] = c["val_loss"] == min(x["val_loss"] for x in self.checkpoints)
        if len(self.checkpoints) > CHECKPOINT_KEEP:
            # drop the oldest non-best checkpoint
            for i, c in enumerate(self.checkpoints):
                if not c["is_best"]:
                    self.checkpoints.pop(i)
                    break
        return ckpt_id

    def restore(self, ckpt_id: str, record) -> bool:
        for c in self.checkpoints:
            if c["ckpt_id"] == ckpt_id:
                with record.train_lock:
                    record.built.module.load_state_dict(c["state"])
                return True
        return False

    def summary(self) -> dict:
        if not self.checkpoints:
            return {"epochs": 0}
        best = min(self.checkpoints, key=lambda c: c["val_loss"])
        return {"epochs": len(self.checkpoints), "best_ckpt": best["ckpt_id"],
                "best_val_loss": best["val_loss"], "best_val_acc": best["val_acc"]}

    def finish(self) -> None:
        if self.status != "error":
            self.status = "stopped" if self._stop.is_set() else "completed"
        self.publish("done", {"status": self.status, "summary": self.summary()})

    def error(self, msg: str) -> None:
        self.status = "error"
        self.publish("error", {"message": msg})

    def snapshot(self) -> dict:
        return {
            "run_id": self.run_id, "model_id": self.model_id, "status": self.status,
            "latest": self.latest,
            "checkpoints": [
                {k: c[k] for k in ("ckpt_id", "epoch", "val_loss", "val_acc", "is_best")}
                for c in self.checkpoints
            ],
            "last_seq": self.seq,
        }


class RunRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, RunHandle] = {}
        self._active: dict[str, str] = {}  # model_id -> run_id (only if running/paused)
        self._lock = threading.Lock()

    def active_for(self, model_id: str) -> Optional[RunHandle]:
        with self._lock:
            rid = self._active.get(model_id)
            run = self._runs.get(rid) if rid else None
        if run and run.status in ("running", "paused"):
            return run
        return None

    def create(self, model_id: str, config, loop) -> RunHandle:
        run = RunHandle(model_id, config, loop)
        with self._lock:
            self._runs[run.run_id] = run
            self._active[model_id] = run.run_id
        return run

    def get(self, run_id: str) -> Optional[RunHandle]:
        with self._lock:
            return self._runs.get(run_id)


runs = RunRegistry()


def _format(frame: dict) -> str:
    return f"event: {frame['type']}\nid: {frame['id']}\ndata: {json.dumps(frame['data'])}\n\n"


async def sse_stream(handle: RunHandle, last_id: int):
    """Async generator of SSE text; replays history after last_id, then streams live."""
    q = handle.subscribe()
    emitted = last_id
    try:
        for f in list(handle.history):
            if f["id"] > emitted:
                emitted = f["id"]
                yield _format(f)
        if handle.status in ("completed", "stopped", "error"):
            return
        while True:
            try:
                f = await asyncio.wait_for(q.get(), timeout=10.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            if f["id"] <= emitted:
                continue
            emitted = f["id"]
            yield _format(f)
            if f["type"] in ("done", "error"):
                break
    finally:
        handle.unsubscribe(q)
