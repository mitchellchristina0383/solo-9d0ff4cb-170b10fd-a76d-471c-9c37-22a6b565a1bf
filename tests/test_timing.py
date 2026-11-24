"""校时偏移推算、轮次登记持久化与试算 API 的自测。

重点覆盖：
1) 两向传输耗时不同（非对称时延）下的偏移区间推算与窗口试算；
2) 新轮次登记即停用旧证据、旧轮次迟报拒收（含 HTTP 409 与重启恢复）；
3) 试算结果对偏移区间内所有整数组合做暴力枚举校验；
4) 缺探测、探测矛盾、窗口过窄、时差超限等无解路径与参数校验。
"""

from __future__ import annotations

import itertools
import random

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.main import app
from app.timing_engine import DeviceSnapshot, analyze_probes, solve_trial
from app.timing_models import ProbeRecord, ProbeBatchRequest, TrialRequest
from app.timing_store import StaleRoundError, TimingStore


def _probe(t1: int, t2: int, t3: int, t4: int) -> ProbeRecord:
    return ProbeRecord(
        t1_controller_send_ms=t1,
        t2_device_receive_ms=t2,
        t3_device_send_ms=t3,
        t4_controller_receive_ms=t4,
    )


def _bruteforce_feasible(snap_a, snap_b, w0: int, w1: int, d: int) -> bool:
    """枚举理论可行域内所有本地点对与全部整数偏移组合，判断是否存在可行下发。"""
    aa, ab = analyze_probes(snap_a.probes), analyze_probes(snap_b.probes)
    if aa.lower is None or ab.lower is None or aa.contradiction or ab.contradiction:
        return False
    for x1 in range(w0 + aa.upper, w1 + aa.lower + 1):
        for x2 in range(w0 + ab.upper, w1 + ab.lower + 1):
            ok = True
            for o1, o2 in itertools.product(
                range(aa.lower, aa.upper + 1), range(ab.lower, ab.upper + 1)
            ):
                t1, t2 = x1 - o1, x2 - o2
                if not (w0 <= t1 <= w1 and w0 <= t2 <= w1 and abs(t1 - t2) <= d):
                    ok = False
                    break
            if ok:
                return True
    return False


def _assert_trial_covers_all_offsets(result, w0: int, w1: int, d: int) -> None:
    """可行结果必须对两台设备偏移区间内的全部整数组合成立。"""
    assert result.feasible
    (s0, s1) = result.devices
    for o0, o1 in itertools.product(
        range(s0.lower, s0.upper + 1), range(s1.lower, s1.upper + 1)
    ):
        t0 = s0.local_time - o0
        t1 = s1.local_time - o1
        assert w0 <= t0 <= w1
        assert w0 <= t1 <= w1
        assert abs(t0 - t1) <= d
    # 响应中的生效浮动范围与枚举极值一致
    assert s0.effective_min == s0.local_time - s0.upper
    assert s0.effective_max == s0.local_time - s0.lower
    assert s1.effective_min == s1.local_time - s1.upper
    assert s1.effective_max == s1.local_time - s1.lower
    # 最坏时差不超限且与公式一致：max(|-Δx+(L2-U1)|, |-Δx+(U2-L1)|)
    assert result.worst_case_skew <= d
    s0, s1 = result.devices
    dx = s1.local_time - s0.local_time
    assert result.worst_case_skew == max(
        abs(-dx + (s1.lower - s0.upper)),
        abs(-dx + (s1.upper - s0.lower)),
    )


# ================================================================ 偏移推算
def test_single_probe_bounds_from_four_timestamps():
    # t3-t4 = 1050-1060 = -10；t2-t1 = 1030-1000 = 30 ⇒ o ∈ [-10, 30]
    analysis = analyze_probes([_probe(1000, 1030, 1050, 1060)])
    assert (analysis.lower, analysis.upper, analysis.contradiction) == (-10, 30, None)


def test_asymmetric_two_way_delays_intersect_bounds():
    """两向时延不同：探测1 上行 10/下行 20，探测2 上行 40/下行 5，区间仍须正确取交。"""
    # 探测1：t1=0,t2=10+o；t3=20+o,t4=40。取 o=5 ⇒ t2=15,t3=25,t4=40
    #   下界 t3-t4=-15，上界 t2-t1=15
    # 探测2：上行 40、下行 5，o=5 ⇒ t1=100,t2=145,t3=160,t4=165
    #   下界 -5，上界 45
    analysis = analyze_probes(
        [_probe(0, 15, 25, 40), _probe(100, 145, 160, 165)]
    )
    assert analysis.contradiction is None
    assert (analysis.lower, analysis.upper) == (-5, 15)  # max(-15,-5), min(15,45)


def test_self_contradicting_probe_negative_delays():
    # t3-t4=100 > t2-t1=10：两向时延不可能同时非负
    analysis = analyze_probes([_probe(0, 10, 200, 100)])
    assert analysis.lower is None and analysis.upper is None
    assert analysis.contradiction is not None and "自身矛盾" in analysis.contradiction


def test_empty_intersection_between_probes_is_contradiction():
    # 探测1 ⇒ o ∈ [0, 10]，探测2 ⇒ o ∈ [20, 30]，交集为空
    analysis = analyze_probes([_probe(0, 10, 0, 0), _probe(0, 30, 20, 0)])
    assert analysis.contradiction is not None and "交集为空" in analysis.contradiction


def test_no_probes_yields_no_bounds():
    analysis = analyze_probes([])
    assert analysis.probe_count == 0 and analysis.lower is None and analysis.upper is None


# ================================================================ 试算求解
def _snapshot(device_id: str, round_no: int, probes: list[ProbeRecord]) -> DeviceSnapshot:
    return DeviceSnapshot(device_id=device_id, round=round_no, probes=probes)


def test_trial_feasible_with_asymmetric_delays_and_bruteforce():
    """两向时延不同的完整试算：结果对所有整数偏移组合成立，且与暴力枚举结论一致。"""
    # SW-A：两探测定出 oA ∈ [-5, 15]
    snap_a = _snapshot(
        "SW-A", 2, [_probe(0, 15, 25, 40), _probe(100, 145, 160, 165)]
    )
    # SW-B：单探测 oB ∈ [-20, 10]（上行 10、下行 30，o=-10 时 t2=0,t3=10,t4=40）
    snap_b = _snapshot("SW-B", 1, [_probe(0, 0, 10, 40)])
    w0, w1, d = 10000, 10200, 60

    result = solve_trial(snap_a, snap_b, w0, w1, d)
    _assert_trial_covers_all_offsets(result, w0, w1, d)
    assert result.reason is None
    assert result.devices[0].round == 2 and result.devices[1].round == 1
    assert _bruteforce_feasible(snap_a, snap_b, w0, w1, d) is True


def test_trial_tight_window_and_zero_skew_exact_solution():
    """偏移被探测唯一确定（区间宽度 0）、窗口单点、时差限值 0：存在精确解。"""
    # t3-t4 = 5-0 = 5，t2-t1 = 5-0 = 5 ⇒ o 恒为 5
    snap_a = _snapshot("A", 0, [_probe(0, 5, 5, 0)])
    snap_b = _snapshot("B", 0, [_probe(0, 5, 5, 0)])
    result = solve_trial(snap_a, snap_b, 1000, 1000, 0)
    assert result.feasible
    # x = W0+U = 1005，实际生效 T = x-o = 1000，恰为窗口单点
    assert result.devices[0].local_time == 1005
    assert result.devices[1].local_time == 1005
    assert result.worst_case_skew == 0
    _assert_trial_covers_all_offsets(result, 1000, 1000, 0)


def test_trial_no_solution_when_window_narrower_than_uncertainty():
    # o ∈ [0, 50] 宽 50，窗口宽 10 ⇒ 无本地点可保证落窗
    snap_a = _snapshot("A", 0, [_probe(0, 50, 0, 0)])
    snap_b = _snapshot("B", 0, [_probe(0, 50, 0, 0)])
    result = solve_trial(snap_a, snap_b, 1000, 1010, 1000)
    assert not result.feasible and result.reason == "no_solution"
    assert "窄于" in result.message
    assert _bruteforce_feasible(snap_a, snap_b, 1000, 1010, 1000) is False


def test_trial_skew_feasible_even_when_uncertainty_widths_are_large():
    """偏移不确定宽度之和 > D 不必然无解：Δx 可抵消同向偏差，结论以暴力枚举为准。"""
    snap_a = _snapshot("A", 0, [_probe(0, 20, 0, 0)])   # oA ∈ [0,20]，宽 20
    snap_b = _snapshot("B", 0, [_probe(0, 10, 0, 0)])   # oB ∈ [0,10]，宽 10
    result = solve_trial(snap_a, snap_b, 1000, 2000, 10)
    expected = _bruteforce_feasible(snap_a, snap_b, 1000, 2000, 10)
    assert result.feasible is expected
    if expected:
        _assert_trial_covers_all_offsets(result, 1000, 2000, 10)
    else:
        assert result.reason == "no_solution"


def test_trial_skew_infeasible_when_required_delta_out_of_window_reach():
    """各自能落窗，但时差要求的 Δx 与落窗允许的 Δx 区间无交集时无解；放宽限值后恢复可行。"""
    # oA∈[2,7]、oB∈[-4,-1]，窗口 [4,9]，D=0（严格同时生效）无解，D=5 可行。
    snap_a = _snapshot("A", 0, [_probe(0, 7, 2, 0)])
    snap_b = _snapshot("B", 0, [_probe(0, -1, -4, 0)])
    result = solve_trial(snap_a, snap_b, 4, 9, 0)
    assert not result.feasible and result.reason == "no_solution"
    assert _bruteforce_feasible(snap_a, snap_b, 4, 9, 0) is False
    wide = solve_trial(snap_a, snap_b, 4, 9, 5)
    assert wide.feasible
    _assert_trial_covers_all_offsets(wide, 4, 9, 5)


def test_trial_single_point_window_with_offset_widths():
    """窗口恰好容纳偏移宽度时每台本地点唯一；时差可行性与暴力枚举一致。"""
    # oA∈[0,10]、oB∈[0,10]：窗口 [1000,1010] 下 xA=xB=1010 唯一
    snap_a = _snapshot("A", 0, [_probe(0, 10, 0, 0)])
    snap_b = _snapshot("B", 0, [_probe(0, 10, 0, 0)])
    for d in (5, 20):
        r = solve_trial(snap_a, snap_b, 1000, 1010, d)
        assert r.feasible is _bruteforce_feasible(snap_a, snap_b, 1000, 1010, d)
        if r.feasible:
            _assert_trial_covers_all_offsets(r, 1000, 1010, d)


def test_trial_missing_probes_and_contradiction_paths():
    snap_empty = _snapshot("A", 1, [])
    snap_ok = _snapshot("B", 0, [_probe(0, 10, 0, 0)])
    result = solve_trial(snap_empty, snap_ok, 0, 100, 100)
    assert not result.feasible and result.reason == "missing_probes"
    assert result.devices[0].local_time is None and result.worst_case_skew is None

    snap_bad = _snapshot("A", 0, [_probe(0, 10, 200, 100)])
    result2 = solve_trial(snap_bad, snap_ok, 0, 100, 100)
    assert not result2.feasible and result2.reason == "contradictory_probes"
    assert result2.devices[0].contradiction is not None


def test_trial_solution_is_deterministic():
    snap_a = _snapshot("A", 0, [_probe(0, 15, -5, 5)])
    snap_b = _snapshot("B", 0, [_probe(0, 5, 0, 10)])
    r1 = solve_trial(snap_a, snap_b, 5000, 5100, 40)
    r2 = solve_trial(snap_a, snap_b, 5000, 5100, 40)
    assert r1.devices[0].local_time == r2.devices[0].local_time
    assert r1.devices[1].local_time == r2.devices[1].local_time


def test_trial_solver_matches_bruteforce_on_random_cases():
    """随机生成偏移区间（含多探测交集）/窗口/时差限值，求解器必须与独立暴力枚举完全一致。"""
    rng = random.Random(20261124)
    for _ in range(150):
        snaps = []
        for k in range(2):
            lo = rng.randint(-25, 25)
            width = rng.randint(0, 15)
            hi = lo + width
            probes = [_probe(0, hi, lo, 0)]  # 边界探测正好表示区间 [lo, hi]
            # 再随机加一条更宽或等宽的探测，交集应仍为 [lo, hi]，验证多探测取交参与试算
            if rng.random() < 0.5:
                probes.append(_probe(0, hi + rng.randint(0, 8), lo - rng.randint(0, 8), 0))
            snaps.append(_snapshot(f"D{k}", rng.randint(0, 9), probes))
        w0 = rng.randint(-30, 50)
        w1 = w0 + rng.randint(0, 45)
        d = rng.choice([0, 0, 1, 2, 5, 10, 20, 40, 80])  # 加权覆盖 D=0 边界
        result = solve_trial(snaps[0], snaps[1], w0, w1, d)
        expected = _bruteforce_feasible(snaps[0], snaps[1], w0, w1, d)
        assert result.feasible is expected, (w0, w1, d, result.reason)
        if expected:
            _assert_trial_covers_all_offsets(result, w0, w1, d)
            # 最坏时差字段不得超过限值，且不得为负
            assert 0 <= result.worst_case_skew <= d


def test_trial_huge_window_is_o1_and_matches_bruteforce():
    """窗口跨 10^9 ms：求解不得随窗口宽度扫描，结论与暴力枚举一致。"""
    snap_a = _snapshot("A", 7, [_probe(0, 15, 25, 40)])   # oA ∈ [-15, 15]
    snap_b = _snapshot("B", 3, [_probe(0, 5, -20, 10)])  # oB ∈ [-30, 5]
    w0, w1, d = 0, 1_000_000_000, 100
    result = solve_trial(snap_a, snap_b, w0, w1, d)
    assert result.feasible
    _assert_trial_covers_all_offsets(result, w0, w1, d)
    assert result.devices[0].round == 7 and result.devices[1].round == 3
    # 暴力枚举器在缩小但同形的窗口上结论一致（可行性只取决于约束宽度）
    assert _bruteforce_feasible(snap_a, snap_b, 0, 200, d) is True
    # 更紧限值下求解器与暴力枚举同结论
    tight = solve_trial(snap_a, snap_b, 0, 200, 10)
    assert tight.feasible is _bruteforce_feasible(snap_a, snap_b, 0, 200, 10)


# ================================================================ 登记库与持久化
def test_store_register_supersede_and_reject_late_round(tmp_path):
    store = TimingStore(str(tmp_path / "timing.json"))
    registered, superseded, total = store.register_batch(
        "SW-A", 0, [_probe(0, 10, 0, 0), _probe(100, 110, 100, 100)]
    )
    assert registered and superseded is None and total == 2

    # 同轮补报 = 旧轮次迟报，拒收且状态不变
    with pytest.raises(StaleRoundError) as exc:
        store.register_batch("SW-A", 0, [_probe(200, 210, 200, 200)])
    assert exc.value.current_round == 0
    assert len(store.snapshot("SW-A").probes) == 2

    # 更小轮次同样拒收
    with pytest.raises(StaleRoundError):
        store.register_batch("SW-A", -1, [])

    # 新轮次空报：登记即停用旧证据
    registered, superseded, total = store.register_batch("SW-A", 1, [])
    assert registered and superseded == 0 and total == 0
    snap = store.snapshot("SW-A")
    assert snap.round == 1 and snap.probes == []

    # 轮次可跳跃（只要严格递增）
    _, superseded, total = store.register_batch("SW-A", 5, [_probe(0, 10, 0, 0)])
    assert superseded == 1 and total == 1
    with pytest.raises(StaleRoundError):
        store.register_batch("SW-A", 3, [])


def test_store_persists_across_restart(tmp_path):
    path = str(tmp_path / "timing.json")
    store = TimingStore(path)
    store.register_batch("SW-A", 0, [_probe(0, 15, 25, 40)])
    store.register_batch("SW-B", 2, [_probe(100, 145, 160, 165)])
    store.register_batch("SW-A", 1, [])  # 旧证据停用后落盘

    # 模拟服务重启：新建同名登记库
    revived = TimingStore(path)
    snap_a = revived.snapshot("SW-A")
    snap_b = revived.snapshot("SW-B")
    assert snap_a is not None and snap_a.round == 1 and snap_a.probes == []
    assert snap_b is not None and snap_b.round == 2 and len(snap_b.probes) == 1
    assert snap_b.probes[0].t1_controller_send_ms == 100
    assert revived.snapshot("unknown") is None

    # 重启后旧轮次迟报仍拒收
    with pytest.raises(StaleRoundError):
        revived.register_batch("SW-B", 2, [_probe(0, 1, 0, 1)])
    with pytest.raises(StaleRoundError):
        revived.register_batch("SW-B", 0, [])


def test_store_snapshots_share_one_commit_point_and_unknown_is_none(tmp_path):
    store = TimingStore(str(tmp_path / "timing.json"))
    store.register_batch("SW-A", 2, [_probe(0, 10, 0, 0)])
    store.register_batch("SW-B", 4, [_probe(0, 20, 5, 0)])
    snaps = store.snapshots(["SW-B", "SW-A", "GHOST"])
    assert snaps[0].device_id == "SW-B" and snaps[0].round == 4
    assert snaps[1].device_id == "SW-A" and snaps[1].round == 2
    assert snaps[2] is None
    # 快照为拷贝：改快照不影响登记库
    snaps[1].probes.append(_probe(1, 1, 1, 1))
    assert len(store.snapshot("SW-A").probes) == 1


def test_store_rejected_batch_does_not_touch_persistence(tmp_path):
    path = str(tmp_path / "timing.json")
    store = TimingStore(path)
    store.register_batch("SW-A", 3, [_probe(0, 10, 0, 0)])
    mtime_before = (tmp_path / "timing.json").stat().st_mtime_ns
    with pytest.raises(StaleRoundError):
        store.register_batch("SW-A", 3, [_probe(9, 9, 9, 9)])
    assert (tmp_path / "timing.json").stat().st_mtime_ns == mtime_before
    assert len(store.snapshot("SW-A").probes) == 1


# ================================================================ Pydantic 校验
def test_trial_request_validation():
    with pytest.raises(Exception):
        TrialRequest(device_ids=["A", "A"], window_start_ms=0, window_end_ms=10,
                     max_effective_skew_ms=1)
    with pytest.raises(Exception):
        TrialRequest(device_ids=["A", "B"], window_start_ms=10, window_end_ms=5,
                     max_effective_skew_ms=1)
    with pytest.raises(Exception):
        TrialRequest(device_ids=["A", "B"], window_start_ms=0, window_end_ms=10,
                     max_effective_skew_ms=-1)
    with pytest.raises(Exception):
        TrialRequest(device_ids=["A"], window_start_ms=0, window_end_ms=10,
                     max_effective_skew_ms=1)
    # 探测批次字段约束
    with pytest.raises(Exception):
        ProbeBatchRequest(device_id="", round=0, probes=[])
    with pytest.raises(Exception):
        ProbeBatchRequest(device_id="A", round=-1, probes=[])


# ================================================================ HTTP 端到端
@pytest.fixture
def client(tmp_path, monkeypatch):
    test_store = TimingStore(str(tmp_path / "timing.json"))
    monkeypatch.setattr(main_module, "store", test_store)
    with TestClient(app) as c:
        yield c


def _batch(device_id, round_no, probes):
    return {"device_id": device_id, "round": round_no, "probes": probes}


def test_http_docs_health_and_unknown_device_404(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200
    assert client.get("/timing/devices/GHOST").status_code == 404


def test_http_probe_lifecycle_and_stale_late_report(client):
    # 轮次 0 带两条探测
    r = client.post(
        "/timing/probes",
        json=_batch(
            "SW-A",
            0,
            [
                {"t1_controller_send_ms": 0, "t2_device_receive_ms": 15,
                 "t3_device_send_ms": 25, "t4_controller_receive_ms": 40},
                {"t1_controller_send_ms": 100, "t2_device_receive_ms": 145,
                 "t3_device_send_ms": 160, "t4_controller_receive_ms": 165},
            ],
        ),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["registered"] is True and body["superseded_round"] is None
    assert body["current_round_probe_count"] == 2

    # 旧轮次迟报：拒收 409，证据条数不变
    late = client.post(
        "/timing/probes",
        json=_batch("SW-A", 0, [
            {"t1_controller_send_ms": 9, "t2_device_receive_ms": 9,
             "t3_device_send_ms": 9, "t4_controller_receive_ms": 9}
        ]),
    )
    assert late.status_code == 409
    assert "迟报" in late.json()["detail"]
    assert client.get("/timing/devices/SW-A").json()["probe_count"] == 2

    # 新轮次空报：停用旧证据
    empty = client.post("/timing/probes", json=_batch("SW-A", 1, []))
    assert empty.status_code == 200
    state = empty.json()
    assert state["superseded_round"] == 0 and state["current_round_probe_count"] == 0
    listing = client.get("/timing/devices").json()
    assert listing["devices"][0]["current_round"] == 1
    assert listing["devices"][0]["offset_bounds"] is None

    # 非法轮次（负数）422
    bad = client.post("/timing/probes", json=_batch("SW-A", -1, []))
    assert bad.status_code == 422


def test_http_trial_full_flow_with_asymmetric_delays(client):
    # SW-A：非对称两向时延，oA ∈ [-5, 15]
    client.post(
        "/timing/probes",
        json=_batch(
            "SW-A", 0,
            [
                {"t1_controller_send_ms": 0, "t2_device_receive_ms": 15,
                 "t3_device_send_ms": 25, "t4_controller_receive_ms": 40},
                {"t1_controller_send_ms": 100, "t2_device_receive_ms": 145,
                 "t3_device_send_ms": 160, "t4_controller_receive_ms": 165},
            ],
        ),
    )
    # SW-B：oB ∈ [-20, 10]
    client.post(
        "/timing/probes",
        json=_batch(
            "SW-B", 0,
            [{"t1_controller_send_ms": 0, "t2_device_receive_ms": 0,
              "t3_device_send_ms": 10, "t4_controller_receive_ms": 40}],
        ),
    )

    r = client.post(
        "/timing/trial",
        json={"device_ids": ["SW-A", "SW-B"], "window_start_ms": 10000,
              "window_end_ms": 10200, "max_effective_skew_ms": 60},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["feasible"] is True and data["reason"] is None
    assert [d["device_id"] for d in data["devices"]] == ["SW-A", "SW-B"]
    assert all(d["round"] == 0 and d["probe_count"] >= 1 for d in data["devices"])
    # 对所有偏移组合暴力复核
    for d in data["devices"]:
        lo, hi = d["offset_bounds"]["lower"], d["offset_bounds"]["upper"]
        for o in range(lo, hi + 1):
            t = d["local_time_ms"] - o
            assert 10000 <= t <= 10200
    da, db = data["devices"]
    for oa in range(da["offset_bounds"]["lower"], da["offset_bounds"]["upper"] + 1):
        for ob in range(db["offset_bounds"]["lower"], db["offset_bounds"]["upper"] + 1):
            assert abs((da["local_time_ms"] - oa) - (db["local_time_ms"] - ob)) <= 60
    assert data["worst_case_skew_ms"] <= 60

    # 时差限值收紧：可行性以求解结果为准（200 + feasible 字段），收紧到极端负值非法走 422
    tight = client.post(
        "/timing/trial",
        json={"device_ids": ["SW-A", "SW-B"], "window_start_ms": 10000,
              "window_end_ms": 10200, "max_effective_skew_ms": 5},
    )
    assert tight.status_code == 200
    tight_body = tight.json()
    if tight_body["feasible"]:
        # 若声称可行，逐偏移组合复核时差确实 <=5
        da, db = tight_body["devices"]
        for oa in range(da["offset_bounds"]["lower"], da["offset_bounds"]["upper"] + 1):
            for ob in range(db["offset_bounds"]["lower"], db["offset_bounds"]["upper"] + 1):
                assert abs((da["local_time_ms"] - oa) - (db["local_time_ms"] - ob)) <= 5
    else:
        assert tight_body["reason"] == "no_solution"

    # 设备未登记 404；请求体非法 422
    assert client.post(
        "/timing/trial",
        json={"device_ids": ["SW-A", "GHOST"], "window_start_ms": 1,
              "window_end_ms": 2, "max_effective_skew_ms": 1},
    ).status_code == 404
    assert client.post(
        "/timing/trial",
        json={"device_ids": ["SW-A", "SW-A"], "window_start_ms": 1,
              "window_end_ms": 2, "max_effective_skew_ms": 1},
    ).status_code == 422


def test_http_trial_uses_only_current_round_evidence(client):
    # 轮次 0 证据很宽，轮次 1 证据更紧；试算必须只使用轮次 1
    client.post(
        "/timing/probes",
        json=_batch("SW-A", 0, [
            {"t1_controller_send_ms": 0, "t2_device_receive_ms": 100,
             "t3_device_send_ms": 0, "t4_controller_receive_ms": 0}
        ]),
    )
    client.post(
        "/timing/probes",
        json=_batch("SW-A", 1, [
            {"t1_controller_send_ms": 0, "t2_device_receive_ms": 10,
             "t3_device_send_ms": 0, "t4_controller_receive_ms": 0}
        ]),
    )
    client.post(
        "/timing/probes",
        json=_batch("SW-B", 0, [
            {"t1_controller_send_ms": 0, "t2_device_receive_ms": 10,
             "t3_device_send_ms": 0, "t4_controller_receive_ms": 0}
        ]),
    )
    r = client.post(
        "/timing/trial",
        json={"device_ids": ["SW-A", "SW-B"], "window_start_ms": 1000,
              "window_end_ms": 1010, "max_effective_skew_ms": 20},
    )
    data = r.json()
    assert data["feasible"] is True
    assert data["devices"][0]["round"] == 1
    assert data["devices"][0]["offset_bounds"] == {"lower": 0, "upper": 10}
