"""校时探测与生效时刻试算的请求/响应模型（时间单位均为整数毫秒）。

偏移约定：``设备本地时刻 = 控制器时刻 + 偏移 o``（o 可为负整数）。
一条四时刻探测 (t1, t2, t3, t4) 对偏移的约束为::

    t3 - t4 <= o <= t2 - t1

其中 t1=控制器发出、t2=设备接收、t3=设备发回、t4=控制器收到。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# 试算失败原因：缺探测 / 探测矛盾 / 约束无解
TrialReason = Literal["missing_probes", "contradictory_probes", "no_solution"]


class ProbeRecord(BaseModel):
    """一次四时刻校时探测，全部为整数毫秒。"""

    t1_controller_send_ms: int = Field(description="t1：控制器发出时刻（控制器时钟）")
    t2_device_receive_ms: int = Field(description="t2：设备接收时刻（设备时钟）")
    t3_device_send_ms: int = Field(description="t3：设备发回时刻（设备时钟）")
    t4_controller_receive_ms: int = Field(description="t4：控制器收到时刻（控制器时钟）")


class ProbeBatchRequest(BaseModel):
    """按设备 + 校时轮次分批上报探测；``probes=[]`` 表示空报，仅登记新轮次。"""

    device_id: str = Field(min_length=1, description="设备标识，非空字符串")
    round: int = Field(ge=0, description="校时轮次，非负整数，逐次递增")
    probes: list[ProbeRecord] = Field(
        default_factory=list, description="本批探测；空列表表示新轮次空报（登记即停用旧证据）"
    )


class ProbeBatchResponse(BaseModel):
    """探测批次接收结果。"""

    device_id: str
    round: int = Field(description="登记后该设备的当前轮次")
    registered: bool = Field(description="本次上报是否开启了一个新轮次（同时停用旧轮次证据）")
    superseded_round: int | None = Field(description="被停用的旧轮次号；无则 null")
    accepted_probe_count: int = Field(description="本批接收的探测条数")
    current_round_probe_count: int = Field(description="当前轮次累计在册探测条数")


class OffsetBounds(BaseModel):
    """偏移 o 的整数可行区间（含两端）。"""

    lower: int = Field(description="偏移下界 L = max(t3 - t4)")
    upper: int = Field(description="偏移上界 U = min(t2 - t1)")


class DeviceTimingState(BaseModel):
    """单台设备的在册校时状态。"""

    device_id: str
    current_round: int = Field(description="当前已登记轮次")
    probe_count: int = Field(description="当前轮次在册探测条数")
    offset_bounds: OffsetBounds | None = Field(
        description="当前轮次全部探测推出的偏移区间；无探测或矛盾时为 null"
    )
    contradiction: str | None = Field(description="探测矛盾说明；不矛盾时为 null")


class TimingStateResponse(BaseModel):
    """全部设备的校时登记状态。"""

    store_path: str = Field(description="持久化文件路径（重启后从该文件恢复）")
    devices: list[DeviceTimingState]


class TrialRequest(BaseModel):
    """两台设备的改配生效时刻试算请求（窗口以控制器时钟计，含两端）。"""

    device_ids: list[str] = Field(
        min_length=2, max_length=2, description="参与试算的两台设备 id，恰好两个且互不相同"
    )
    window_start_ms: int = Field(description="改配窗口起点（控制器时钟，含）")
    window_end_ms: int = Field(description="改配窗口终点（控制器时钟，含）")
    max_effective_skew_ms: int = Field(
        ge=0, description="两台设备实际生效时刻之差的允许上限（含等号），非负整数毫秒"
    )

    @model_validator(mode="after")
    def _check_cross_fields(self) -> "TrialRequest":
        if self.device_ids[0] == self.device_ids[1]:
            raise ValueError("两台设备 id 不能相同")
        if self.window_end_ms < self.window_start_ms:
            raise ValueError("window_end_ms 不能小于 window_start_ms（窗口含两端）")
        return self


class EffectiveRange(BaseModel):
    """给定本地下发时刻后，控制器时钟上实际生效时刻的浮动范围（含两端）。"""

    min_ms: int
    max_ms: int


class DeviceTrialInfo(BaseModel):
    """试算响应中单台设备的推算明细。"""

    device_id: str
    round: int = Field(description="试算所使用的轮次（该设备当前轮次）")
    probe_count: int = Field(description="该轮次使用的全部探测条数")
    offset_bounds: OffsetBounds | None = Field(description="偏移可行区间；缺探测/矛盾时为 null")
    local_time_ms: int | None = Field(description="可下发的本地整数毫秒时刻；无解时为 null")
    controller_effective_range_ms: EffectiveRange | None = Field(
        description="该本地时刻在控制器时钟上的实际生效浮动范围；无解时为 null"
    )


class TrialResponse(BaseModel):
    """试算结果：业务可行与否都返回 200，失败原因见 reason/message。"""

    feasible: bool = Field(description="是否得到一对可下发的本地时刻")
    reason: TrialReason | None = Field(
        description="不可行原因：missing_probes=缺少探测；contradictory_probes=探测矛盾；no_solution=约束无解"
    )
    message: str | None = Field(description="不可行的中文说明；可行时为 null")
    window_start_ms: int
    window_end_ms: int
    max_effective_skew_ms: int
    devices: list[DeviceTrialInfo] = Field(description="按请求顺序排列的两台设备明细")
    worst_case_skew_ms: int | None = Field(
        description="取返回的本地时刻时，所有偏移组合下两台设备生效时差的最坏值（毫秒）；无解时为 null"
    )
