"""校时偏移推算与改配生效时刻试算的纯函数引擎。

偏移约定（o 为设备时钟相对控制器时钟的固定偏移，整数，可为负）::

    设备本地时刻 = 控制器时刻 + o

单条四时刻探测 (t1, t2, t3, t4)：

- 正向（控制器 -> 设备）：t2 = t1 + d_up + o，正向时延 d_up >= 0 ⇒ o <= t2 - t1；
- 反向（设备 -> 控制器）：t4 = t3 + d_down - o，反向时延 d_down >= 0 ⇒ o >= t3 - t4。

故每条探测给出 ``[t3-t4, t2-t1]``；同轮设备对同一控制器的偏移固定，
多条探测的可行区间取交集，交集为空即探测矛盾。

试算：控制器窗口 [W0, W1]（含两端）、允许最大时差 D（含等号）。
对设备 i 与本地下发时刻 x_i，实际在控制器时钟上的生效时刻为 T_i = x_i - o_i，
需对偏移区间内的**任意整数组合**同时满足：

- W0 <= T_i <= W1（两台各自落在窗口内）；
- |T_1 - T_2| <= D。
"""

from __future__ import annotations

from dataclasses import dataclass

from .timing_models import ProbeRecord


@dataclass(frozen=True)
class ProbeAnalysis:
    """一轮探测的偏移推算结果。"""

    probe_count: int
    lower: int | None  # 有探测且一致时为整数，否则 None
    upper: int | None
    contradiction: str | None  # 矛盾说明；不矛盾为 None


def analyze_probes(probes: list[ProbeRecord]) -> ProbeAnalysis:
    """用同一轮次的全部探测推算偏移区间。

    返回区间 ``[lower, upper]``（含两端，整数）；无探测返回全 None；
    任一探测自身不可行（负向时延上界 < 下界）或探测间交集为空，判为矛盾。
    """
    if not probes:
        return ProbeAnalysis(0, None, None, None)

    lower: int | None = None
    upper: int | None = None
    for idx, p in enumerate(probes, start=1):
        lo = p.t3_device_send_ms - p.t4_controller_receive_ms
        hi = p.t2_device_receive_ms - p.t1_controller_send_ms
        if lo > hi:
            return ProbeAnalysis(
                len(probes),
                None,
                None,
                (
                    f"第 {idx} 条探测自身矛盾：t3-t4={lo} > t2-t1={hi}，"
                    "两向传输耗时不可能同时非负"
                ),
            )
        lower = lo if lower is None else max(lower, lo)
        upper = hi if upper is None else min(upper, hi)

    assert lower is not None and upper is not None
    if lower > upper:
        return ProbeAnalysis(
            len(probes),
            None,
            None,
            (
                f"各探测的偏移可行区间交集为空：下界 max(t3-t4)={lower} > "
                f"上界 min(t2-t1)={upper}，同轮偏移固定的前提不成立"
            ),
        )
    return ProbeAnalysis(len(probes), lower, upper, None)


@dataclass(frozen=True)
class DeviceSnapshot:
    """试算时点某设备当前轮次的在册证据快照。"""

    device_id: str
    round: int
    probes: list[ProbeRecord]


@dataclass(frozen=True)
class DeviceSolution:
    """单台设备的试算明细。"""

    device_id: str
    round: int
    probe_count: int
    lower: int | None
    upper: int | None
    contradiction: str | None
    local_time: int | None
    effective_min: int | None  # 本地时刻对应的控制器生效浮动范围 [min, max]
    effective_max: int | None


@dataclass(frozen=True)
class TrialResult:
    """两台设备的试算结果。"""

    feasible: bool
    reason: str | None  # missing_probes / contradictory_probes / no_solution
    message: str | None
    devices: tuple[DeviceSolution, DeviceSolution]
    worst_case_skew: int | None


def solve_trial(
    snap_a: DeviceSnapshot,
    snap_b: DeviceSnapshot,
    window_start: int,
    window_end: int,
    max_skew: int,
) -> TrialResult:
    """在同一已提交快照上试算两台设备的可下发本地整数毫秒时刻。

    严格按“无论偏移取哪种可能组合都成立”的最坏情形求解，全程整数运算。
    """

    analyses = (
        analyze_probes(snap_a.probes),
        analyze_probes(snap_b.probes),
    )
    snaps = (snap_a, snap_b)

    def _device_solution(i: int, local: int | None) -> DeviceSolution:
        a = analyses[i]
        eff_min = eff_max = None
        if local is not None and a.lower is not None and a.upper is not None:
            # T = x - o；o 取最大时 T 最小，o 取最小时 T 最大
            eff_min = local - a.upper
            eff_max = local - a.lower
        return DeviceSolution(
            device_id=snaps[i].device_id,
            round=snaps[i].round,
            probe_count=a.probe_count,
            lower=a.lower,
            upper=a.upper,
            contradiction=a.contradiction,
            local_time=local,
            effective_min=eff_min,
            effective_max=eff_max,
        )

    # 1) 缺少探测
    missing = [snaps[i].device_id for i in range(2) if analyses[i].probe_count == 0]
    if missing:
        return TrialResult(
            False,
            "missing_probes",
            f"设备 {', '.join(missing)} 当前轮次没有任何探测，无法推算偏移",
            (_device_solution(0, None), _device_solution(1, None)),
            None,
        )

    # 2) 探测矛盾
    contradicted = [
        (snaps[i].device_id, analyses[i].contradiction)
        for i in range(2)
        if analyses[i].contradiction is not None
    ]
    if contradicted:
        detail = "；".join(f"{dev}：{msg}" for dev, msg in contradicted)
        return TrialResult(
            False,
            "contradictory_probes",
            f"探测矛盾，偏移区间不存在：{detail}",
            (_device_solution(0, None), _device_solution(1, None)),
            None,
        )

    bounds = [(a.lower, a.upper) for a in analyses]  # type: ignore[arg-type]

    # 3) 单台设备的窗口可行性：本地时刻 x 须使 [x-U, x-L] 整体落在 [W0, W1]
    #    ⇒ x ∈ [W0+U, W1+L]（整数闭区间）
    x_ranges: list[tuple[int, int]] = []
    for lo, u in bounds:
        x_lo = window_start + u
        x_hi = window_end + lo
        if x_lo > x_hi:
            return TrialResult(
                False,
                "no_solution",
                (
                    f"窗口 [{window_start}, {window_end}]（含两端）窄于设备偏移不确定宽度 "
                    f"{u - lo}ms，不存在能保证落在窗口内的本地时刻"
                ),
                (_device_solution(0, None), _device_solution(1, None)),
                None,
            )
        x_ranges.append((x_lo, x_hi))

    # 4) 时差约束。令 Δx = x2 - x1，则生效时差（控制器时钟）：
    #      T1-T2 = (x1-o1)-(x2-o2) = -Δx + (o2-o1)
    #    两独立偏移之差 o2-o1 的整数取值范围为 [L2-U1, U2-L1]。
    #    要求对全部偏移组合 |T1-T2| <= D，即整个区间 [-Δx+(L2-U1), -Δx+(U2-L1)]
    #    都落在 [-D, D] 内，归并为（整数闭区间，已用穷举对照验证）：
    #      Δx ∈ [(U2-L1)-D, (L2-U1)+D]
    #    再叠加 x2=x1+Δx 的落窗约束 Δx ∈ [lo2-x1, hi2-x1] 与 x1∈[lo1,hi1]，
    #    全程整数区间运算，O(1)，不随窗口宽度扫描。
    l1, u1 = bounds[0]
    l2, u2 = bounds[1]
    delta_skew_lo = (u2 - l1) - max_skew
    delta_skew_hi = (l2 - u1) + max_skew
    if delta_skew_lo > delta_skew_hi:
        # 时差可行 Δx 区间为空：两偏移不确定宽度之和 > 2D 时，无论怎么选下发时差，
        # 残余的最坏生效时差都超过限值。
        total_width = (u1 - l1) + (u2 - l2)
        return TrialResult(
            False,
            "no_solution",
            (
                f"两台设备偏移不确定宽度之和为 {total_width}ms，大于两倍时差限值 "
                f"2×{max_skew}={2 * max_skew}ms，任何下发时刻都无法把最坏生效时差压到 "
                f"{max_skew}ms 以内"
            ),
            (_device_solution(0, None), _device_solution(1, None)),
            None,
        )

    lo1, hi1 = x_ranges[0]
    lo2, hi2 = x_ranges[1]
    # 对给定 x1，Δx 需落在 [max(delta_skew_lo, lo2-x1), min(delta_skew_hi, hi2-x1)]，
    # 该区间非空 ⇔ x1 ∈ [lo2-delta_skew_hi, hi2-delta_skew_lo]，与 [lo1,hi1] 相交。
    x1_lo = max(lo1, lo2 - delta_skew_hi)
    x1_hi = min(hi1, hi2 - delta_skew_lo)
    if x1_lo > x1_hi:
        return TrialResult(
            False,
            "no_solution",
            (
                "窗口与时差约束的整数可行域为空：两台设备各自存在可落窗的本地时刻，"
                "但窗口内选不出满足生效时差上限的下发时刻组合"
            ),
            (_device_solution(0, None), _device_solution(1, None)),
            None,
        )

    chosen_x1 = x1_lo  # 取下界，结果确定且可复现
    dx_lo = max(delta_skew_lo, lo2 - chosen_x1)
    dx_hi = min(delta_skew_hi, hi2 - chosen_x1)
    assert dx_lo <= dx_hi  # 由候选域非空保证
    chosen_dx = dx_lo
    x1, x2 = chosen_x1, chosen_x1 + chosen_dx
    sol0 = _device_solution(0, x1)
    sol1 = _device_solution(1, x2)
    # 最坏时差（全部整数偏移组合下的最大绝对时差）：偏差区间两端取极值
    worst_case = max(
        abs(-chosen_dx + (l2 - u1)),
        abs(-chosen_dx + (u2 - l1)),
    )
    return TrialResult(
        True,
        None,
        None,
        (sol0, sol1),
        worst_case,
    )
