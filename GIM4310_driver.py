#!/usr/bin/env python3
"""持续读取并显示电机位置信息。按 Ctrl-C 退出。

RS485 自定义协议 V3.03b3 — 波特率 115200, 8N1, 小端字节序。
"""

import math
import os
import struct
import threading
import time

import serial

# ── CRC16_MODBUS ────────────────────────────────────────────────────────────

_CRC16_TABLE = [
    0x0000, 0xC0C1, 0xC181, 0x0140, 0xC301, 0x03C0, 0x0280, 0xC241,
    0xC601, 0x06C0, 0x0780, 0xC741, 0x0500, 0xC5C1, 0xC481, 0x0440,
    0xCC01, 0x0CC0, 0x0D80, 0xCD41, 0x0F00, 0xCFC1, 0xCE81, 0x0E40,
    0x0A00, 0xCAC1, 0xCB81, 0x0B40, 0xC901, 0x09C0, 0x0880, 0xC841,
    0xD801, 0x18C0, 0x1980, 0xD941, 0x1B00, 0xDBC1, 0xDA81, 0x1A40,
    0x1E00, 0xDEC1, 0xDF81, 0x1F40, 0xDD01, 0x1DC0, 0x1C80, 0xDC41,
    0x1400, 0xD4C1, 0xD581, 0x1540, 0xD701, 0x17C0, 0x1680, 0xD641,
    0xD201, 0x12C0, 0x1380, 0xD341, 0x1100, 0xD1C1, 0xD081, 0x1040,
    0xF001, 0x30C0, 0x3180, 0xF141, 0x3300, 0xF3C1, 0xF281, 0x3240,
    0x3600, 0xF6C1, 0xF781, 0x3740, 0xF501, 0x35C0, 0x3480, 0xF441,
    0x3C00, 0xFCC1, 0xFD81, 0x3D40, 0xFF01, 0x3FC0, 0x3E80, 0xFE41,
    0xFA01, 0x3AC0, 0x3B80, 0xFB41, 0x3900, 0xF9C1, 0xF881, 0x3840,
    0x2800, 0xE8C1, 0xE981, 0x2940, 0xEB01, 0x2BC0, 0x2A80, 0xEA41,
    0xEE01, 0x2EC0, 0x2F80, 0xEF41, 0x2D00, 0xEDC1, 0xEC81, 0x2C40,
    0xE401, 0x24C0, 0x2580, 0xE541, 0x2700, 0xE7C1, 0xE681, 0x2640,
    0x2200, 0xE2C1, 0xE381, 0x2340, 0xE101, 0x21C0, 0x2080, 0xE041,
    0xA001, 0x60C0, 0x6180, 0xA141, 0x6300, 0xA3C1, 0xA281, 0x6240,
    0x6600, 0xA6C1, 0xA781, 0x6740, 0xA501, 0x65C0, 0x6480, 0xA441,
    0x6C00, 0xACC1, 0xAD81, 0x6D40, 0xAF01, 0x6FC0, 0x6E80, 0xAE41,
    0xAA01, 0x6AC0, 0x6B80, 0xAB41, 0x6900, 0xA9C1, 0xA881, 0x6840,
    0x7800, 0xB8C1, 0xB981, 0x7940, 0xBB01, 0x7BC0, 0x7A80, 0xBA41,
    0xBE01, 0x7EC0, 0x7F80, 0xBF41, 0x7D00, 0xBDC1, 0xBC81, 0x7C40,
    0xB401, 0x74C0, 0x7580, 0xB541, 0x7700, 0xB7C1, 0xB681, 0x7640,
    0x7200, 0xB2C1, 0xB381, 0x7340, 0xB101, 0x71C0, 0x7080, 0xB041,
    0x5000, 0x90C1, 0x9181, 0x5140, 0x9301, 0x53C0, 0x5280, 0x9241,
    0x9601, 0x56C0, 0x5780, 0x9741, 0x5500, 0x95C1, 0x9481, 0x5440,
    0x9C01, 0x5CC0, 0x5D80, 0x9D41, 0x5F00, 0x9FC1, 0x9E81, 0x5E40,
    0x5A00, 0x9AC1, 0x9B81, 0x5B40, 0x9901, 0x59C0, 0x5880, 0x9841,
    0x8801, 0x48C0, 0x4980, 0x8941, 0x4B00, 0x8BC1, 0x8A81, 0x4A40,
    0x4E00, 0x8EC1, 0x8F81, 0x4F40, 0x8D01, 0x4DC0, 0x4C80, 0x8C41,
    0x4400, 0x84C1, 0x8581, 0x4540, 0x8701, 0x47C0, 0x4680, 0x8641,
    0x8201, 0x42C0, 0x4380, 0x8341, 0x4100, 0x81C1, 0x8081, 0x4040,
]


def _crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc = (crc >> 8) ^ _CRC16_TABLE[(crc ^ b) & 0xFF]
    return crc


def _require_non_negative(value: float, name: str) -> None:
    """参数校验: 值不能为负数。"""
    if value < 0:
        raise ValueError(f"{name} 不能为负数: {value}")


def _require_finite(value: float, name: str) -> None:
    """参数校验: 值必须为有限数 (禁止 NaN/Inf)。"""
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值: {value}")


def _require_uint16(value: int, name: str) -> None:
    """参数校验: 整数必须在 uint16 范围 [0, 65535] 内。"""
    if not 0 <= value <= 0xFFFF:
        raise ValueError(f"{name} 超出 uint16 范围: {value} (须 0~65535)")


def _require_int32(value: int, name: str) -> None:
    """参数校验: 整数必须在 int32 范围 [-2^31, 2^31-1] 内。"""
    if not -0x80000000 <= value <= 0x7FFFFFFF:
        raise ValueError(f"{name} 超出 int32 范围: {value}")


def _clamp(value: int, lo: int, hi: int) -> int:
    """将整数钳制在 [lo, hi] 范围内。"""
    return lo if value < lo else hi if value > hi else value


def _deg_to_encoder(angle_deg: float) -> int:
    """角度 (度) → 编码器计数值。"""
    return int(angle_deg * ENCODER_SCALE)


def _rpm_to_raw(speed_rpm: float) -> int:
    """转速 (rpm) → 协议原始值 (0.01 rpm)。"""
    return int(speed_rpm * 100)


def _scale_bipolar_to_raw(value: float, max_abs: float, raw_max: int) -> int:
    """将 [-max_abs, +max_abs] 映射到 [0, raw_max], 钳制越界值。"""
    if max_abs <= 0:
        return 0
    raw = round((value + max_abs) / (2.0 * max_abs) * raw_max)
    return _clamp(raw, 0, raw_max)


def _scale_unipolar_to_raw(value: float, max_val: float, raw_max: int) -> int:
    """将 [0, max_val] 映射到 [0, raw_max], 钳制越界值。"""
    if max_val <= 0:
        return 0
    raw = round(value / max_val * raw_max)
    return _clamp(raw, 0, raw_max)


# ── 协议常量 ─────────────────────────────────────────────────────────────────

HEADER_HOST = 0xAE
HEADER_SLAVE = 0xAC
FRAME_HEADER_LEN = 5
FRAME_CRC_LEN = 2
MAX_PAYLOAD_LEN = 64

ENCODER_RESOLUTION = 16384
ANGLE_SCALE = 360.0 / ENCODER_RESOLUTION
ENCODER_SCALE = 1.0 / ANGLE_SCALE  # 度数 → 编码器值 (精确互为倒数)

# ── MIT 运控默认参数 ────────────────────────────────────────────────────────

MIT_DEFAULT_POS_MAX_RAD = 95.5        # 位置最大值 (rad)
MIT_DEFAULT_VEL_MAX_RAD_S = 45.0      # 速度最大值 (rad/s)
MIT_DEFAULT_T_MAX_NM = 18.0           # 力矩最大值 (N·m)
MIT_DEFAULT_POS_KP_MAX = 500          # 位置 KP 最大值
MIT_DEFAULT_VEL_KD_MAX = 5            # 速度 KD 最大值

MODE_NAMES = {0: "关闭", 1: "电压控制", 2: "Q轴电流控制", 3: "速度控制", 4: "位置控制"}

# ── 配置 (按需修改) ──────────────────────────────────────────────────────────

SERIAL_PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBVQPDL-if00-port0"
MOTOR_ADDR = 0x01
BAUDRATE = 921600
READ_INTERVAL = 0.02


# ── 电机通信 ─────────────────────────────────────────────────────────────────

class Motor:
    """RS485 电机驱动, 支持协议 V3.03b3"""

    def __init__(self, port=SERIAL_PORT, address=MOTOR_ADDR, baudrate=BAUDRATE,
                 timeout=0.01, _ser=None):
        self.address = address
        self._seq = 0
        self._lock = threading.Lock()
        self._external_ser = _ser is not None
        if _ser is not None:
            self._ser = _ser
        else:
            self._ser = serial.Serial(
                port=port, baudrate=baudrate,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=timeout,
            )
            self._set_serial_low_latency(self._ser)

        # MIT 运控参数缓存 — 尝试从电机读取实际值, 失败则用默认值
        try:
            self._mit_pos_max = MIT_DEFAULT_POS_MAX_RAD
            self._mit_vel_max = MIT_DEFAULT_VEL_MAX_RAD_S
            self._mit_t_max = MIT_DEFAULT_T_MAX_NM
            self._mit_pos_kp_max = MIT_DEFAULT_POS_KP_MAX
            self._mit_vel_kd_max = MIT_DEFAULT_VEL_KD_MAX
            self.read_mit_params()  # 尽力同步电机实际参数 (成功后 _cache_mit_params 更新 scale)
        except Exception:
            pass  # 电机未上电或无应答, 保留默认值
        # 确保 scale 已初始化 (read_mit_params 失败时 _cache_mit_params 不会被调用)
        self._mit_pos_scale = (self._mit_pos_max / 32768.0
                               if self._mit_pos_max > 0 else 0.0)
        self._mit_vel_scale = (self._mit_vel_max / 2048.0
                               if self._mit_vel_max > 0 else 0.0)
        self._mit_t_scale = (self._mit_t_max / 2048.0
                             if self._mit_t_max > 0 else 0.0)

    @staticmethod
    def _set_serial_low_latency(ser: serial.Serial) -> None:
        """尝试将 USB 串口延迟设为 1ms。"""
        # pyserial 3.5+ 内置方法 (Linux FTDI/CP210x)
        try:
            ser.set_low_latency_mode(True)
            return
        except (AttributeError, NotImplementedError):
            pass  # 方法不存在或不支持, 尝试 sysfs 备选方案
        # 备选: 直接写 sysfs (需 root 权限)
        try:
            port_name = os.path.basename(ser.port)
            latency_path = f"/sys/bus/usb-serial/devices/{port_name}/latency_timer"
            if os.path.exists(latency_path):
                with open(latency_path, "w") as f:
                    f.write("1")
        except (OSError, PermissionError):
            pass  # 非 USB 串口或权限不足, 静默跳过

    def _build_frame(self, cmd: int, data: bytes = b"") -> bytes:
        if not 0 <= cmd <= 0xFF:
            raise ValueError(f"命令码超出范围: {cmd} (须 0~255)")
        if len(data) > 0xFF:
            raise ValueError(
                f"数据过长: {len(data)} 字节 (单帧最大 255 字节)"
            )
        self._seq = (self._seq + 1) & 0xFF
        payload = bytes([HEADER_HOST, self._seq, self.address, cmd, len(data)]) + data
        return payload + struct.pack("<H", _crc16(payload))

    def _write_only(self, cmd: int, data: bytes = b"") -> None:
        """发送指令, 不等待应答 (用于高频实时控制, 避免阻塞)。

        仅写入串口即返回, 适用于不需要即时确认的场景,
        例如高频更新目标位置。
        调用 flush() 确保帧完整发出后再返回 — 这对共享 RS485 总线
        至关重要: 否则连续 _write_only 会把多帧合并成连续数据流,
        部分电机可能因缺少帧间隔而错过指令。
        """
        with self._lock:
            frame = self._build_frame(cmd, data)
            self._ser.write(frame)
            self._ser.flush()

    def _command(self, cmd: int, data: bytes, parser, *, wait: bool = True):
        """发送指令并根据 wait 决定是否等待应答。

        wait=True  → 调用 _send, 用 parser 解析应答后返回。
        wait=False → 调用 _write_only, fire-and-forget, 返回 None。
        """
        if wait:
            raw = self._send(cmd, data)
            return parser(raw)
        else:
            self._write_only(cmd, data)
            return None

    def _send(self, cmd: int, data: bytes = b"") -> bytes:
        with self._lock:
            frame = self._build_frame(cmd, data)
            self._ser.reset_input_buffer()
            self._ser.write(frame)

            # 直接用串口超时阻塞读取，不忙等待
            head = self._ser.read(FRAME_HEADER_LEN)
            if len(head) < FRAME_HEADER_LEN:
                raise TimeoutError(f"命令 0x{cmd:02X}: 无应答")

            # ── 校验响应帧头 ──
            if head[0] != HEADER_SLAVE:
                raise ValueError(
                    f"无效响应头: 期望 0x{HEADER_SLAVE:02X}, 收到 0x{head[0]:02X}"
                )
            if head[1] != self._seq:
                raise ValueError(
                    f"序列号不匹配: 期望 {self._seq:02X}, 收到 {head[1]:02X}"
                )
            if head[2] != self.address:
                raise ValueError(
                    f"地址不匹配: 期望 0x{self.address:02X}, 收到 0x{head[2]:02X}"
                )
            if head[3] != cmd:
                raise ValueError(
                    f"命令码不匹配: 期望 0x{cmd:02X}, 收到 0x{head[3]:02X}"
                )

            dlen = head[4]
            if dlen > MAX_PAYLOAD_LEN:
                raise ValueError(
                    f"响应数据长度异常: {dlen} (最大 {MAX_PAYLOAD_LEN})"
                )

            tail = self._ser.read(dlen + FRAME_CRC_LEN)
            if len(tail) < dlen + FRAME_CRC_LEN:
                raise TimeoutError(f"命令 0x{cmd:02X}: 应答不完整")

            raw = head + tail
            expected_crc = struct.unpack("<H", raw[FRAME_HEADER_LEN + dlen:
                                                     FRAME_HEADER_LEN + dlen + FRAME_CRC_LEN])[0]
            if _crc16(raw[:FRAME_HEADER_LEN + dlen]) != expected_crc:
                raise ValueError("CRC16 校验失败")

            return raw[FRAME_HEADER_LEN:FRAME_HEADER_LEN + dlen]

    def read_status(self):
        """读取电机状态, 返回包含各字段的字典 (值已转换为物理单位)"""
        data = self._send(0x0B)
        return self._parse_status(data)

    # ── 用户参数读写 ──────────────────────────────────────────────────────────

    _BAUD_MAP = {0: 921600, 1: 460800, 2: 115200, 3: 57600,
                 4: 38400, 5: 19200, 6: 9600}
    _BAUD_REV = {v: k for k, v in _BAUD_MAP.items()}

    def read_user_params(self) -> bytes:
        """读取用户参数 (0x10, V3 协议), 返回原始参数字节串"""
        data = struct.pack("<BB", 0x03, 0x00)
        return self._send(0x10, data)

    def write_user_params(self, params: bytes) -> bytes:
        """写入并保存用户参数 (0x11), 返回应答参数字节串"""
        return self._send(0x11, params)

    def get_rs485_baudrate(self) -> int:
        """读取当前电机 RS485 波特率 (返回值如 921600, 115200 等)"""
        params = self.read_user_params()
        code = params[15]
        return self._BAUD_MAP.get(code, 115200)

    def set_rs485_baudrate(self, baudrate: int) -> None:
        """修改并保存 RS485 波特率 (如 921600, 115200 等)。

        注意: 波特率变更后电机可能需要重启才能生效。
        """
        code = self._BAUD_REV.get(baudrate)
        if code is None:
            supported = sorted(self._BAUD_REV.keys())
            raise ValueError(
                f"不支持的波特率 {baudrate}, 可选值: {supported}"
            )
        params = bytearray(self.read_user_params())
        params[15] = code
        self.write_user_params(bytes(params))

    # ── 位置控制 ──────────────────────────────────────────────────────────────

    def set_target_position(self, angle_deg: float) -> dict:
        """绝对值位置控制 (0x22)。返回解析后的电机状态 dict。"""
        encoder_val = _deg_to_encoder(angle_deg)
        data = struct.pack("<i", encoder_val)
        raw = self._send(0x22, data)
        return self._parse_status(raw)

    def set_target_position_trapezoidal(
        self, angle_deg: float,
        max_speed_rpm: float = 100.0,
        max_accel_rpm_s: float = 200.0,
        max_decel_rpm_s: float = 200.0,
        *,
        wait: bool = True,
    ) -> dict:
        """梯形曲线绝对值位置控制 (0x26) — 平滑加减速。

        参数:
            angle_deg:       目标多圈角度 (度)
            max_speed_rpm:   最大速度 (rpm), 单位 0.01rpm
            max_accel_rpm_s: 最大加速度 (rpm/s), 单位 0.01rpm/s
            max_decel_rpm_s: 最大减速度 (rpm/s), 单位 0.01rpm/s
            wait:            是否等待应答 (默认 True), False → fire-and-forget

        返回解析后的电机状态 dict (wait=True 时)。
        """
        _require_non_negative(max_speed_rpm, "max_speed_rpm")
        _require_non_negative(max_accel_rpm_s, "max_accel_rpm_s")
        _require_non_negative(max_decel_rpm_s, "max_decel_rpm_s")
        encoder_val = _deg_to_encoder(angle_deg)
        speed = _rpm_to_raw(max_speed_rpm)
        accel = _rpm_to_raw(max_accel_rpm_s)
        decel = _rpm_to_raw(max_decel_rpm_s)
        data = struct.pack("<BiIII", 0x00, encoder_val, speed, accel, decel)
        return self._command(0x26, data, self._parse_status, wait=wait)

    # ── 位置+速度控制 ────────────────────────────────────────────────────────

    def set_target_position_speed(
        self, angle_deg: float, speed_rpm: float = 10.0,
        *, wait: bool = True,
    ):
        """位置+速度控制 (0x25) — 绝对位置 + 速度前馈。

        同时指定目标位置和到达该位置时应有的速度,
        电机在目标之间不会减速到零, 实现连续丝滑运动。
        适合正弦轨迹等需要平滑连续运动的场景。

        参数:
            angle_deg:  目标绝对角度 (度)
            speed_rpm:  目标速度 (rpm), 速度前馈
            wait:       是否等待应答, 默认 True 返回状态 dict
        """
        _require_finite(speed_rpm, "speed_rpm")
        encoder_val = _deg_to_encoder(angle_deg)
        spd = _rpm_to_raw(abs(speed_rpm))
        data = struct.pack("<BiI", 0x00, encoder_val, spd)
        return self._command(0x25, data, self._parse_status, wait=wait)

    # ── 位置滤波控制 ──────────────────────────────────────────────────────────

    def configure_position_filter(
        self,
        bandwidth_hz: float = 50.0,
        inertia_nm_per_turn_s2: float = 0.001,
        ff_current_limit_a: float = 1.0,
    ) -> None:
        """配置位置滤波控制参数 (0x17), 断电不保存。

        参数:
            bandwidth_hz:           位置滤波带宽 (Hz), 默认 50
            inertia_nm_per_turn_s2: 转动惯量 (N·m/(turn/s²)), 0=禁用电流前馈
            ff_current_limit_a:     前馈电流上限 (A)
        """
        _require_non_negative(bandwidth_hz, "bandwidth_hz")
        _require_finite(inertia_nm_per_turn_s2, "inertia_nm_per_turn_s2")
        _require_non_negative(ff_current_limit_a, "ff_current_limit_a")
        bw = int(bandwidth_hz)
        _require_uint16(bw, "bandwidth_hz (原始值)")
        inr = inertia_nm_per_turn_s2
        lim = int(ff_current_limit_a * 1000)  # A → 0.001A
        data = struct.pack("<HfI", bw, inr, lim)
        self._send(0x17, data)

    def set_target_position_filtered(
        self, angle_deg: float, max_speed_rpm: float = 20.0,
        *, wait: bool = True,
    ):
        """位置滤波控制 (0x27) — 绝对位置 + 速度限制 + 低通滤波 + 惯性前馈。

        位置指令先经过低通滤波器, 再输出电流前馈, 最终由 PID 闭环。
        适合相机云台等需要平滑运动 + 精准定位的场景。

        参数:
            angle_deg:      目标多圈角度 (度)
            max_speed_rpm:  最大速度 (rpm)
            wait:           是否等待应答 (默认 True).
                            True → 返回解析后的状态 dict (与 read_status 相同)
                            False → fire-and-forget, 返回 None
        """
        _require_non_negative(max_speed_rpm, "max_speed_rpm")
        encoder_val = _deg_to_encoder(angle_deg)
        speed = _rpm_to_raw(max_speed_rpm)
        data = struct.pack("<BiI", 0x00, encoder_val, speed)
        return self._command(0x27, data, self._parse_status, wait=wait)

    def _parse_status(self, data: bytes) -> dict:
        """解析 0x0B 格式的状态数据 (也用于 0x22/0x26/0x27 等控制指令的应答)。"""
        if len(data) < 22:
            raise ValueError(
                f"状态响应数据过短: 期望≥22字节, 实际{len(data)}字节"
            )
        return dict(
            single_turn_deg=struct.unpack_from("<H", data, 0)[0] * ANGLE_SCALE,
            multi_turn_deg=struct.unpack_from("<i", data, 2)[0] * ANGLE_SCALE,
            speed_rpm=struct.unpack_from("<i", data, 6)[0] * 0.01,
            q_current_a=struct.unpack_from("<i", data, 10)[0] * 0.001,
            bus_voltage_v=struct.unpack_from("<H", data, 14)[0] * 0.01,
            bus_current_a=struct.unpack_from("<H", data, 16)[0] * 0.01,
            temperature=data[18],
            run_mode=data[19],
            motor_enabled=data[20],
            fault_code=data[21],
        )

    # ── 基础位置控制 ──────────────────────────────────────────────────────────

    def move_relative(self, delta_deg: float) -> None:
        """相对位置控制 (0x23)。

        协议 0x23 数据字段为相对位置 (4s, 单位 Count)，固件基于当
        前位置累加该偏移量。
        """
        encoder_val = _deg_to_encoder(delta_deg)
        _require_int32(encoder_val, "delta_deg (编码器值)")
        data = struct.pack("<i", encoder_val)
        self._send(0x23, data)

    # ── MIT 运控模式 ──────────────────────────────────────────────────────────

    # ── 0x30: MIT 参数 读/写 ──────────────────────────────────────────────────

    def read_mit_params(self) -> dict:
        """读取 MIT 运控参数 (0x30), 并更新本地缓存。

        返回:
            dict: pos_max_rad, vel_max_rad_s, t_max_nm, pos_kp_max, vel_kd_max
        """
        data = self._send(0x30)
        params = self._parse_mit_params(data)
        self._cache_mit_params(params)
        return params

    def configure_mit_params(
        self,
        pos_max_rad: float = MIT_DEFAULT_POS_MAX_RAD,
        vel_max_rad_s: float = MIT_DEFAULT_VEL_MAX_RAD_S,
        t_max_nm: float = MIT_DEFAULT_T_MAX_NM,
        pos_kp_max: int = MIT_DEFAULT_POS_KP_MAX,
        vel_kd_max: int = MIT_DEFAULT_VEL_KD_MAX,
    ) -> dict:
        """配置 MIT 运控参数 (0x30), 断电保存, 同时更新本地缓存。

        参数:
            pos_max_rad:   位置最大值 (rad), 默认 95.5
            vel_max_rad_s: 速度最大值 (rad/s), 默认 45.0
            t_max_nm:      力矩最大值 (N·m), 默认 18.0
            pos_kp_max:    位置 KP 最大值, 默认 500
            vel_kd_max:    速度 KD 最大值, 默认 5
        """
        _require_non_negative(pos_max_rad, "pos_max_rad")
        _require_non_negative(vel_max_rad_s, "vel_max_rad_s")
        _require_non_negative(t_max_nm, "t_max_nm")
        _require_non_negative(pos_kp_max, "pos_kp_max")
        _require_non_negative(vel_kd_max, "vel_kd_max")
        pos_raw = int(pos_max_rad * 10)          # 0.1 rad
        vel_raw = int(vel_max_rad_s * 100)       # 0.01 rad/s
        t_raw = int(t_max_nm * 100)              # 0.01 N·m
        kp_raw = int(pos_kp_max)
        kd_raw = int(vel_kd_max)
        for name, val in [("pos_max_rad", pos_raw), ("vel_max_rad_s", vel_raw),
                           ("t_max_nm", t_raw), ("pos_kp_max", kp_raw),
                           ("vel_kd_max", kd_raw)]:
            _require_uint16(val, f"{name} (原始值)")
        raw = struct.pack("<HHHHH", pos_raw, vel_raw, t_raw, kp_raw, kd_raw)
        data = self._send(0x30, raw)
        params = self._parse_mit_params(data)
        self._cache_mit_params(params)
        return params

    def _parse_mit_params(self, data: bytes) -> dict:
        """解析 0x30 应答数据 (10 字节, 5 个 2u 小端字段)。"""
        if len(data) < 10:
            raise ValueError(
                f"MIT 参数数据过短: 期望 10 字节, 实际 {len(data)} 字节"
            )
        pos_raw, vel_raw, t_raw, kp_raw, kd_raw = struct.unpack_from(
            "<HHHHH", data, 0
        )
        return dict(
            pos_max_rad=pos_raw * 0.1,
            vel_max_rad_s=vel_raw * 0.01,
            t_max_nm=t_raw * 0.01,
            pos_kp_max=kp_raw,
            vel_kd_max=kd_raw,
        )

    def _cache_mit_params(self, params: dict) -> None:
        """将 MIT 参数写入实例缓存 (用于值域到物理量的换算)。"""
        with self._lock:
            self._mit_pos_max = params["pos_max_rad"]
            self._mit_vel_max = params["vel_max_rad_s"]
            self._mit_t_max = params["t_max_nm"]
            self._mit_pos_kp_max = params["pos_kp_max"]
            self._mit_vel_kd_max = params["vel_kd_max"]
            # 预计算解码比例因子 (原始值 → 物理量), 避免热路径上的重复除法
            self._mit_pos_scale = (self._mit_pos_max / 32768.0
                                   if self._mit_pos_max > 0 else 0.0)
            self._mit_vel_scale = (self._mit_vel_max / 2048.0
                                   if self._mit_vel_max > 0 else 0.0)
            self._mit_t_scale = (self._mit_t_max / 2048.0
                                 if self._mit_t_max > 0 else 0.0)

    def _get_mit_params_snapshot(self) -> dict:
        """原子读取 MIT 参数缓存, 返回一致性快照 (用于编码/解码)。"""
        with self._lock:
            return dict(
                pos_max=self._mit_pos_max,
                vel_max=self._mit_vel_max,
                t_max=self._mit_t_max,
                pos_kp_max=self._mit_pos_kp_max,
                vel_kd_max=self._mit_vel_kd_max,
                pos_scale=self._mit_pos_scale,
                vel_scale=self._mit_vel_scale,
                t_scale=self._mit_t_scale,
            )

    # ── 0x31: 读取 MIT 状态 ───────────────────────────────────────────────────

    def read_mit_status(self) -> dict:
        """读取 MIT 运控模式实时状态 (0x31)。

        返回 dict:
            position_rad:     机械位置 (rad), 范围 [-pos_max_rad, +pos_max_rad]
            velocity_rad_s:   机械速度 (rad/s)
            torque_nm:        输出力矩 (N·m)
            mit_mode_active:  是否处于运控模式
            fault:            是否有故障
            temperature:      工作温度 (℃)
        """
        data = self._send(0x31)
        return self._parse_mit_state(data)

    def _parse_mit_state(self, data: bytes) -> dict:
        """解析 0x31/0x32 应答数据 (9 字节)。"""
        if len(data) < 9:
            raise ValueError(
                f"MIT 状态数据过短: 期望 ≥9 字节, 实际 {len(data)} 字节"
            )

        # 位置 (16bit, [0]高8位 [1]低8位): 0~65535 ↔ -Pos_Max ~ +Pos_Max
        pos_raw = (data[0] << 8) | data[1]
        # 速度 (12bit, [2]高8位 [3]高4位): 0~4095 ↔ -Vel_Max ~ +Vel_Max
        vel_raw = (data[2] << 4) | (data[3] >> 4)
        # 力矩 (12bit, [3]低4位 [4]低8位): 0~4095 ↔ -T_Max ~ +T_Max
        t_raw = ((data[3] & 0x0F) << 8) | data[4]

        status = data[5]
        temperature = data[6]

        # 换算为物理量 (中点偏移 + 预计算比例因子)
        mit = self._get_mit_params_snapshot()
        pos_rad = (pos_raw - 32768) * mit["pos_scale"]
        vel_rad_s = (vel_raw - 2048) * mit["vel_scale"]
        torque_nm = (t_raw - 2048) * mit["t_scale"]

        return dict(
            position_rad=pos_rad,
            velocity_rad_s=vel_rad_s,
            torque_nm=torque_nm,
            mit_mode_active=bool(status & 0x01),
            fault=bool(status & 0x02),
            temperature=temperature,
        )

    # ── 0x32: MIT 控制指令 ────────────────────────────────────────────────────

    def set_mit_control(
        self,
        pos_rad: float,
        vel_rad_s: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        t_ff_nm: float = 0.0,
        *,
        wait: bool = True,
    ):
        """MIT 运控指令 (0x32) — 同时控制位置、速度、KP、KD、力矩前馈。

        参考 MIT Cheetah 控制架构:
            τ = kp * (θ_des - θ) + kd * (ω_des - ω) + τ_ff

        参数:
            pos_rad:   目标位置 (rad)
            vel_rad_s: 目标速度前馈 (rad/s), 范围 [-Vel_Max, +Vel_Max]
            kp:        位置增益 KP, 范围 [0, Pos_KP_Max]
            kd:        速度阻尼 KD, 范围 [0, Vel_KD_Max]
            t_ff_nm:   力矩前馈 (N·m), 范围 [-T_Max, +T_Max]
            wait:       是否等待应答 (默认 True)
                        True → 返回解析后的 MIT 状态 dict
                        False → fire-and-forget, 返回 None
        """

        # 原子读取 MIT 参数快照, 防止并发修改导致缩放不一致
        mit = self._get_mit_params_snapshot()

        # 物理量 → 原始值
        pos_raw = _scale_bipolar_to_raw(pos_rad, mit["pos_max"], 65535)
        vel_raw = _scale_bipolar_to_raw(vel_rad_s, mit["vel_max"], 4095)
        kp_raw = _scale_unipolar_to_raw(kp, mit["pos_kp_max"], 4095)
        kd_raw = _scale_unipolar_to_raw(kd, mit["vel_kd_max"], 4095)
        t_raw = _scale_bipolar_to_raw(t_ff_nm, mit["t_max"], 4095)

        # 按协议格式打包 8 字节
        packed = bytes([
            (pos_raw >> 8) & 0xFF,                                   # [0] 位置高8位
            pos_raw & 0xFF,                                          # [1] 位置低8位
            (vel_raw >> 4) & 0xFF,                                   # [2] 速度高8位
            ((vel_raw & 0x0F) << 4) | ((kp_raw >> 8) & 0x0F),       # [3] 速度低4位 | KP高4位
            kp_raw & 0xFF,                                           # [4] KP低8位
            (kd_raw >> 4) & 0xFF,                                    # [5] KD高8位
            ((kd_raw & 0x0F) << 4) | ((t_raw >> 8) & 0x0F),         # [6] KD低4位 | 力矩高4位
            t_raw & 0xFF,                                            # [7] 力矩低8位
        ])

        return self._command(0x32, packed, self._parse_mit_state, wait=wait)

    def enable(self) -> None:
        """电机使能 (0x2E), 进入闭环控制 (fire-and-forget, 不等待应答)。"""
        self._write_only(0x2E)

    def disable(self) -> None:
        """电机失能 (0x2F), 进入自由态。"""
        self._send(0x2F)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.disable()
        except Exception:
            pass
        self.close()
        return False

    def disable_and_close(self) -> None:
        try:
            self.disable()
        except Exception:
            pass
        self.close()

    def __repr__(self):
        return (f"Motor(port={self._ser.port!r}, address=0x{self.address:02X}, "
                f"open={self._ser.is_open})")

    def move(self, angle_deg: float,
             max_speed_rpm: float = 50.0,
             max_accel_rpm_s: float = 50.0,
             max_decel_rpm_s: float = 50.0) -> dict:
        """梯形曲线绝对位置控制 (0x26) 快捷方法。"""
        return self.set_target_position_trapezoidal(
            angle_deg,
            max_speed_rpm=max_speed_rpm,
            max_accel_rpm_s=max_accel_rpm_s,
            max_decel_rpm_s=max_decel_rpm_s,
        )

    def close(self):
        if self._external_ser:
            return
        with self._lock:
            if self._ser.is_open:
                self._ser.close()


# ── 多电机总线管理 ────────────────────────────────────────────────────────────

class MotorBus:
    """管理共享 RS485 总线上的多台电机, 提供统一的创建/销毁/轮询接口。"""

    def __init__(self, port: str, addresses: list, baudrate: int = BAUDRATE,
                 timeout: float = 0.01):
        self._ser = serial.Serial(
            port=port, baudrate=baudrate,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE, timeout=timeout,
        )
        Motor._set_serial_low_latency(self._ser)
        self.motors: list[Motor] = [
            Motor(port=port, address=addr, baudrate=baudrate,
                  timeout=timeout, _ser=self._ser)
            for addr in addresses
        ]

    def close(self) -> None:
        for m in self.motors:
            try:
                m.disable()
            except Exception:
                pass
        if self._ser.is_open:
            self._ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ── 主程序 ───────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="持续读取并显示多个电机状态信息 (RS485 协议 V3.03b3)",
    )
    parser.add_argument("--port", default=SERIAL_PORT,
                        help=f"串口设备路径 (默认: {SERIAL_PORT})")
    parser.add_argument("--ids", type=str, default="1,2,3,4",
                        help="电机地址列表, 逗号分隔 (默认: 1,2,3,4)")
    parser.add_argument("--baud", type=int, default=BAUDRATE,
                        help=f"波特率 (默认: {BAUDRATE})")
    parser.add_argument("--interval", type=float, default=READ_INTERVAL,
                        help=f"读取间隔/秒 (默认: {READ_INTERVAL})")
    args = parser.parse_args()

    addresses = [int(x.strip()) for x in args.ids.split(",")]

    CURSOR_UP = "\033[A"
    CLEAR_LINE = "\033[K"

    with MotorBus(port=args.port, addresses=addresses, baudrate=args.baud) as bus:
        print(f"连接 {args.port}, 电机地址: {addresses}")

        header = (
            f"{'ID':>3}  {'单圈角度':>8}  {'多圈角度':>10}  {'速度 rpm':>9}  "
            f"{'Q轴电流 A':>9}  {'母线 V':>6}  {'温度':>4}  "
            f"{'模式':>12}  {'使能':>6}  {'故障':>6}"
        )
        print(header)
        print("-" * len(header))

        error_counts = {addr: 0 for addr in addresses}
        lines_printed = 0

        try:
            while True:
                if lines_printed > 0:
                    print(CURSOR_UP * lines_printed, end="")

                lines_printed = 0

                for motor in bus.motors:
                    try:
                        s = motor.read_status()
                        error_counts[motor.address] = 0

                        mode_name = MODE_NAMES.get(s["run_mode"],
                                                   f"未知({s['run_mode']})")
                        fault_str = (f"0x{s['fault_code']:02X}"
                                     if s["fault_code"] else "无")

                        print(
                            CLEAR_LINE +
                            f"{motor.address:3d}  "
                            f"{s['single_turn_deg']:8.2f}  "
                            f"{s['multi_turn_deg']:10.2f}  "
                            f"{s['speed_rpm']:9.2f}  "
                            f"{s['q_current_a']:9.3f}  "
                            f"{s['bus_voltage_v']:6.2f}  "
                            f"{s['temperature']:4d}℃  "
                            f"{mode_name:>12}  "
                            f"{'使能' if s['motor_enabled'] else '失能':>6}  "
                            f"{fault_str:>6}",
                            flush=True,
                        )
                        lines_printed += 1

                    except (TimeoutError, serial.SerialException,
                            ValueError, struct.error) as e:
                        error_counts[motor.address] += 1
                        cnt = error_counts[motor.address]
                        err_type = "超时" if isinstance(e, TimeoutError) else "错误"
                        print(
                            CLEAR_LINE +
                            f"{motor.address:3d}  "
                            f"⚠ {err_type} (x{cnt}): {e}",
                            flush=True,
                        )
                        lines_printed += 1

                time.sleep(args.interval)

        except KeyboardInterrupt:
            pass
        finally:
            print("\n已断开")


if __name__ == "__main__":
    main()
