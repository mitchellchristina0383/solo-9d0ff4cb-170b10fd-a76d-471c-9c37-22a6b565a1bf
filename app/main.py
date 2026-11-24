"""FastAPI 入口：通过 /docs 可直接发起重放与校时试算。"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from .models import ReplayRequest, ReplayResponse
from .replay import simulate
from .timing_engine import analyze_probes, solve_trial
from .timing_models import (
    DeviceTimingState,
    DeviceTrialInfo,
    EffectiveRange,
    OffsetBounds,
    ProbeBatchRequest,
    ProbeBatchResponse,
    TimingStateResponse,
    TrialRequest,
    TrialResponse,
)
from .timing_store import StaleRoundError, TimingStore

app = FastAPI(
    title="交换机出口 告警/相机图像 重放试算 + 校时生效时刻试算 API",
    description=(
        "产线告警与相机图像共用交换机出口的重放仿真，以及两台串联交换机改配窗口的校时试算。\n\n"
        "## 重放\n"
        "- 时间一律整数毫秒，时间轴从 0 开始，仿真区间 `[0, replay_end_ms)`。\n"
        "- 每周期前段（长度 reserve_ms）只发告警，其余时间只发图像。\n"
        "- 出口一次只发一帧，整帧必须完全落在本类时段内才会开始发送。\n"
        "- 同类帧按到达顺序发送（同时到达按请求中的先后）。\n"
        "- 可在指定时刻换一套周期/预留：到点若正在发帧则等其发完再切换，排队帧保留，"
        "新周期从实际生效时刻起算。\n"
        "- 告警从到达计时，发完即停止；未发完算到重放终点；耗时严格大于告警时限即超时。\n\n"
        "## 校时试算\n"
        "- 探测按设备和递增的校时轮次分批上报，每条含 t1 控制器发出、t2 设备接收、"
        "t3 设备发回、t4 控制器收到四个整数毫秒时刻。\n"
        "- 偏移约定：设备本地时刻 = 控制器时刻 + 偏移 o；单条探测约束 `t3-t4 <= o <= t2-t1`，"
        "同轮多条取交集。\n"
        "- 新轮次可空报，登记即停用旧证据；旧轮次迟报拒收（409）。\n"
        "- 试算用同一已提交时点下两台设备各自当前轮次的全部探测，返回一对可下发的本地整数毫秒时刻，"
        "保证任意偏移组合下都在窗口内生效且时差不超限。\n"
        "- 登记与证据持久化到 JSON 文件，服务重启后仍可试算。"
    ),
    version="1.1.0",
)

_STORE_PATH = os.environ.get("TIMING_STORE_PATH", "data/timing_store.json")
store = TimingStore(_STORE_PATH)


_EXAMPLE_PAYLOAD = {
    "period_ms": 100,
    "reserve_ms": 40,
    "alarm_deadline_ms": 80,
    "replay_end_ms": 320,
    "frames": [
        {"id": "A1", "type": "alarm", "arrival_ms": 0, "busy_ms": 30},
        {"id": "A2", "type": "alarm", "arrival_ms": 5, "busy_ms": 60},
        {"id": "I1", "type": "image", "arrival_ms": 10, "busy_ms": 50},
        {"id": "A3", "type": "alarm", "arrival_ms": 120, "busy_ms": 25},
        {"id": "I2", "type": "image", "arrival_ms": 130, "busy_ms": 70},
    ],
    "switch_at_ms": 150,
    "new_period_ms": 120,
    "new_reserve_ms": 60,
}

_PROBE_EXAMPLE = {
    "device_id": "SW-A",
    "round": 0,
    "probes": [
        {
            "t1_controller_send_ms": 1000,
            "t2_device_receive_ms": 1030,
            "t3_device_send_ms": 1050,
            "t4_controller_receive_ms": 1060,
        },
        {
            "t1_controller_send_ms": 2000,
            "t2_device_receive_ms": 2025,
            "t3_device_send_ms": 2040,
            "t4_controller_receive_ms": 2055,
        },
    ],
}

_TRIAL_EXAMPLE = {
    "device_ids": ["SW-A", "SW-B"],
    "window_start_ms": 10000,
    "window_end_ms": 10200,
    "max_effective_skew_ms": 30,
}


@app.get("/health", tags=["运维"], summary="健康检查")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/replay",
    response_model=ReplayResponse,
    tags=["重放"],
    summary="执行一次重放试算",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "含换配的完整示例": {"value": _EXAMPLE_PAYLOAD},
                        "不换配（最简）": {
                            "value": {
                                "period_ms": 100,
                                "reserve_ms": 40,
                                "alarm_deadline_ms": 80,
                                "replay_end_ms": 200,
                                "frames": [
                                    {"type": "alarm", "arrival_ms": 0, "busy_ms": 30},
                                    {"type": "image", "arrival_ms": 0, "busy_ms": 50},
                                ],
                            }
                        },
                    }
                }
            }
        }
    },
)
def replay(req: ReplayRequest) -> ReplayResponse:
    """按请求参数仿真，返回实际切换时刻、各帧发送起止、终点状态与超时告警。"""
    return simulate(req)


# ---------------------------------------------------------------- 校时试算
@app.post(
    "/timing/probes",
    response_model=ProbeBatchResponse,
    tags=["校时试算"],
    summary="上报一批校时探测（新轮次登记即停用旧证据，旧轮次迟报 409）",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "带探测的新轮次": {"value": _PROBE_EXAMPLE},
                        "新轮次空报": {
                            "value": {"device_id": "SW-A", "round": 1, "probes": []},
                            "description": "仅登记轮次 1 并停用轮次 0 的全部证据",
                        },
                    }
                }
            }
        }
    },
)
def submit_probes(req: ProbeBatchRequest) -> ProbeBatchResponse:
    """按设备 + 轮次登记探测；轮次必须严格递增，旧轮次迟来补报返回 409。"""
    try:
        registered, superseded, total = store.register_batch(
            req.device_id, req.round, req.probes
        )
    except StaleRoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return ProbeBatchResponse(
        device_id=req.device_id,
        round=req.round,
        registered=registered,
        superseded_round=superseded,
        accepted_probe_count=len(req.probes),
        current_round_probe_count=total,
    )


@app.get(
    "/timing/devices",
    response_model=TimingStateResponse,
    tags=["校时试算"],
    summary="查看全部设备的当前轮次、在册探测与偏移推算",
)
def list_timing_devices() -> TimingStateResponse:
    devices: list[DeviceTimingState] = []
    for dev_id, current_round, stored_probes in store.list_devices():
        records = [p.to_record() for p in stored_probes]
        analysis = analyze_probes(records)
        devices.append(
            DeviceTimingState(
                device_id=dev_id,
                current_round=current_round,
                probe_count=analysis.probe_count,
                offset_bounds=(
                    OffsetBounds(lower=analysis.lower, upper=analysis.upper)
                    if analysis.lower is not None and analysis.upper is not None
                    else None
                ),
                contradiction=analysis.contradiction,
            )
        )
    return TimingStateResponse(store_path=store.path, devices=devices)


@app.get(
    "/timing/devices/{device_id}",
    response_model=DeviceTimingState,
    tags=["校时试算"],
    summary="查看单台设备的校时登记状态",
)
def get_timing_device(device_id: str) -> DeviceTimingState:
    snap = store.snapshot(device_id)
    if snap is None:
        raise HTTPException(status_code=404, detail=f"设备 {device_id} 从未登记过校时轮次")
    analysis = analyze_probes(snap.probes)
    return DeviceTimingState(
        device_id=device_id,
        current_round=snap.round,
        probe_count=analysis.probe_count,
        offset_bounds=(
            OffsetBounds(lower=analysis.lower, upper=analysis.upper)
            if analysis.lower is not None and analysis.upper is not None
            else None
        ),
        contradiction=analysis.contradiction,
    )


@app.post(
    "/timing/trial",
    response_model=TrialResponse,
    tags=["校时试算"],
    summary="试算两台设备可下发的本地生效时刻（窗口含两端，按最坏偏移组合保证）",
    openapi_extra={
        "requestBody": {
            "content": {"application/json": {"examples": {"试算示例": {"value": _TRIAL_EXAMPLE}}}}
        }
    },
)
def run_trial(req: TrialRequest) -> TrialResponse:
    """用同一已提交时点下两台设备各自当前轮次的全部探测推算并求解。

    缺探测、探测矛盾或约束无解时仍返回 200，``feasible=false`` 并在 ``reason/message``
    说明原因；设备从未登记返回 404。
    """
    # 同一已提交时点：单锁内一次取两台设备的快照，避免两次读取之间混入并发提交
    snapshots = store.snapshots(req.device_ids)
    for dev_id, snap in zip(req.device_ids, snapshots):
        if snap is None:
            raise HTTPException(status_code=404, detail=f"设备 {dev_id} 从未登记过校时轮次")
    assert snapshots[0] is not None and snapshots[1] is not None

    result = solve_trial(
        snapshots[0],
        snapshots[1],
        req.window_start_ms,
        req.window_end_ms,
        req.max_effective_skew_ms,
    )
    infos = [
        DeviceTrialInfo(
            device_id=sol.device_id,
            round=sol.round,
            probe_count=sol.probe_count,
            offset_bounds=(
                OffsetBounds(lower=sol.lower, upper=sol.upper)
                if sol.lower is not None and sol.upper is not None
                else None
            ),
            local_time_ms=sol.local_time,
            controller_effective_range_ms=(
                EffectiveRange(min_ms=sol.effective_min, max_ms=sol.effective_max)
                if sol.effective_min is not None and sol.effective_max is not None
                else None
            ),
        )
        for sol in result.devices
    ]
    return TrialResponse(
        feasible=result.feasible,
        reason=result.reason,  # type: ignore[arg-type]
        message=result.message,
        window_start_ms=req.window_start_ms,
        window_end_ms=req.window_end_ms,
        max_effective_skew_ms=req.max_effective_skew_ms,
        devices=infos,
        worst_case_skew_ms=result.worst_case_skew,
    )
