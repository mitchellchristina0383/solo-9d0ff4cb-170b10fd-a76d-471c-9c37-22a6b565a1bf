"""校时轮次与探测证据的登记库（线程安全，JSON 文件持久化）。

登记语义：

- 探测按“设备 + 递增校时轮次”分批上报；轮次必须严格大于该设备当前轮次，
  登记新轮次立即停用旧轮次的全部证据（旧证据不再参与试算，也不再保留）。
- 新轮次可空报（``probes=[]``）：仅登记轮次并停用旧证据。
- 已被停用的旧轮次若迟来补报，一律拒收（:class:`StaleRoundError`），由路由层转 409。
- 每次提交成功后原子落盘，服务重启后仍用已登记的轮次和证据试算。
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass

from .timing_engine import DeviceSnapshot
from .timing_models import ProbeRecord

_STORE_VERSION = 1


class StaleRoundError(Exception):
    """上报轮次不大于设备当前轮次（旧轮次迟报），拒收。"""

    def __init__(self, device_id: str, submitted_round: int, current_round: int):
        self.device_id = device_id
        self.submitted_round = submitted_round
        self.current_round = current_round
        super().__init__(
            f"设备 {device_id} 拒收轮次 {submitted_round} 的迟报：当前已登记轮次为 "
            f"{current_round}，旧轮次证据已停用"
        )


@dataclass(frozen=True)
class StoredProbe:
    """落盘的一条探测（与 :class:`ProbeRecord` 字段对齐，便于 JSON 序列化）。"""

    t1_controller_send_ms: int
    t2_device_receive_ms: int
    t3_device_send_ms: int
    t4_controller_receive_ms: int

    def to_record(self) -> ProbeRecord:
        return ProbeRecord(
            t1_controller_send_ms=self.t1_controller_send_ms,
            t2_device_receive_ms=self.t2_device_receive_ms,
            t3_device_send_ms=self.t3_device_send_ms,
            t4_controller_receive_ms=self.t4_controller_receive_ms,
        )

    @classmethod
    def from_record(cls, p: ProbeRecord) -> "StoredProbe":
        return cls(
            t1_controller_send_ms=p.t1_controller_send_ms,
            t2_device_receive_ms=p.t2_device_receive_ms,
            t3_device_send_ms=p.t3_device_send_ms,
            t4_controller_receive_ms=p.t4_controller_receive_ms,
        )


@dataclass
class _DeviceEvidence:
    current_round: int
    probes: list[StoredProbe]


class TimingStore:
    """内存登记 + JSON 原子落盘；进程内以可重入锁串行化所有写操作。"""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self._devices: dict[str, _DeviceEvidence] = {}
        self._load()

    # ------------------------------------------------------------ 持久化
    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if raw.get("version") != _STORE_VERSION:
            raise ValueError(f"校时登记文件版本不受支持：{raw.get('version')!r}")
        devices: dict[str, _DeviceEvidence] = {}
        for dev_id, item in raw.get("devices", {}).items():
            devices[dev_id] = _DeviceEvidence(
                current_round=int(item["current_round"]),
                probes=[StoredProbe(**probe) for probe in item.get("probes", [])],
            )
        self._devices = devices

    def _persist_locked(self) -> None:
        """调用方须持有锁；先写同目录临时文件再原子替换，避免半写文件。"""
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        payload = {
            "version": _STORE_VERSION,
            "devices": {
                dev_id: {
                    "current_round": ev.current_round,
                    "probes": [probe.__dict__ for probe in ev.probes],
                }
                for dev_id, ev in sorted(self._devices.items())
            },
        }
        fd, tmp_path = tempfile.mkstemp(prefix=".timing-store-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.path)
        except BaseException:
            os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------ 写接口
    def register_batch(
        self, device_id: str, round_no: int, probes: list[ProbeRecord]
    ) -> tuple[bool, int | None, int]:
        """登记一批探测。

        返回 ``(是否新登记轮次, 被停用的旧轮次（无则 None）, 当前轮次在册探测总数)``。
        轮次不大于当前轮次时抛 :class:`StaleRoundError`，状态与文件均不变。
        """
        with self._lock:
            existing = self._devices.get(device_id)
            if existing is not None and round_no <= existing.current_round:
                raise StaleRoundError(device_id, round_no, existing.current_round)

            superseded: int | None = None
            if existing is None:
                evidence = _DeviceEvidence(current_round=round_no, probes=[])
                self._devices[device_id] = evidence
                registered = True
            else:
                superseded = existing.current_round
                existing.current_round = round_no
                existing.probes = []  # 登记即停用旧证据
                evidence = existing
                registered = True

            evidence.probes.extend(StoredProbe.from_record(p) for p in probes)
            self._persist_locked()
            return registered, superseded, len(evidence.probes)

    # ------------------------------------------------------------ 读接口
    def snapshot(self, device_id: str) -> DeviceSnapshot | None:
        """取某设备当前轮次的证据快照（深拷贝语义，调用方在锁外使用也安全）。"""
        with self._lock:
            ev = self._devices.get(device_id)
            if ev is None:
                return None
            return DeviceSnapshot(
                device_id=device_id,
                round=ev.current_round,
                probes=[p.to_record() for p in ev.probes],
            )

    def snapshots(self, device_ids: list[str]) -> list[DeviceSnapshot | None]:
        """在同一次加锁内取多台设备的快照，保证试算基于同一已提交时点。

        未登记设备对应位置返回 None，由调用方转 404。
        """
        with self._lock:
            return [
                (
                    DeviceSnapshot(
                        device_id=dev_id,
                        round=ev.current_round,
                        probes=[p.to_record() for p in ev.probes],
                    )
                    if (ev := self._devices.get(dev_id)) is not None
                    else None
                )
                for dev_id in device_ids
            ]

    def list_devices(self) -> list[tuple[str, int, list[StoredProbe]]]:
        with self._lock:
            return [
                (dev_id, ev.current_round, list(ev.probes))
                for dev_id, ev in sorted(self._devices.items())
            ]
