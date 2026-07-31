#!/usr/bin/env python3
"""四轴机械臂 — 坐标显示 + 键盘控制 + 控制台移动 (极简版)

坐标系: +X=左 +Y=前 +Z=上  (右手定则)
按键: W/S→臂伸缩  A/D→基座旋转  Shift/Ctrl→Z升降  F→继电器切换  Q/ESC→退出
控制台: 输入增量 <Δ角度°> [Δ前伸mm] [ΔZmm]  例如: 10 100 → 基座+10°, 臂+100mm
角度: 直接读取编码器多圈角度, 由物理限位保证零点
"""

import math
import os
import select
import sys
import time
import argparse
import keyboard
import serial
import serial.tools.list_ports

from GIM4310_driver import MotorBus, SERIAL_PORT, BAUDRATE

# ── 机械臂尺寸 (mm) ─────────────────────────────────────────────────────────
L1 = 150.0   # 大臂: 肩→肘
L2 = 150.0   # 小臂: 肘→腕
L3 = 0.0     # 手部: 腕→末端

# ── 执行器 (末端工具): 中心相对末端电机局部系偏移 ────────────────────────────
# 末端电机局部系: +x = 沿小臂方向向外, +y = 水平向右 (垂直臂平面), ID4 维持腕部水平
TOOL_OFFSET_X = 12.0   # 执行器中心: 局部 +x 偏移 (mm)
TOOL_OFFSET_Y = -36.0  # 执行器中心: 局部 +y 偏移 (mm, 实际为 -36)

# ── 上电偏置检测: 首次读数 > 18° 时, 该电机本会话减去 36° ───────────────────
BIAS_THRESHOLD  = 18.0    # 首次读取超过该值 (°) 触发修正
BIAS_CORRECTION = -36.0   # 修正量 (°), 叠加到编码器读数上

# ── 关节限位 (逻辑角度, °) ──────────────────────────────────────────────────
JOINT_LIMITS  = {1: (-90, 90), 2: (-5, 180), 3: (0, 160), 4: (-120, 120)}  # ID4 自动维持水平

# ── 电机映射: 逻辑角 ↔ 电机绝对角 ──────────────────────────────────────────
JOINT_SIGNS   = {1: +1, 2: +1, 3: -1, 4: +1}
JOINT_OFFSETS = {1: 0,  2: 0,  3: 160, 4: 24}


# ── 摄像头坐标标定: 机械臂水平面 (mm) → 摄像头像素 (px) ────────────────────
# 线性 (仿射) 变换, 4 组标定点最小二乘拟合 (手动标定, 最大残差 ~5.8 px ≈ 2.3 mm):
#   臂( 90,  -8.4) → 相机(123,  85)   臂(250, -11) → 相机(544,  95)
#   臂(100, -120)  → 相机(134, 361)   臂(262, -120) → 相机(538, 365)
#   u = 2.5634875·x + 0.23176316·y - 100.11222
#   v = 0.023520676·x - 2.4727018·y + 62.017411
#  CAMERA_A[0] = u 的 (x, y) 系数, CAMERA_A[1] = v 的 (x, y) 系数, CAMERA_A[2] = 常数项
CAMERA_A = (
    ( 2.563487500474e+00,  2.31763156852e-01),
    ( 2.3520676112e-02,  -2.472701774444e+00),
    (-1.00112215611229e+02, 6.2017411269669e+01),
)


def arm_to_camera(x, y):
    """机械臂水平面坐标 (x, y) [mm] → 摄像头像素坐标 (u, v) [px]."""
    u = CAMERA_A[0][0] * x + CAMERA_A[0][1] * y + CAMERA_A[2][0]
    v = CAMERA_A[1][0] * x + CAMERA_A[1][1] * y + CAMERA_A[2][1]
    return u, v


# ── 正运动学 (FK) ───────────────────────────────────────────────────────────
def forward_kinematics(joints):
    """关节角度 → (x, y, z).  joints: {1..4: deg}"""
    i1 = math.radians(joints[1])
    i2 = math.radians(joints[2])
    i3 = math.radians(joints[3])

    fa = i2 + i3                             # 小臂绝对角
    r_h = L1 * math.cos(i2) + L2 * math.cos(fa)
    z_h = L1 * math.sin(i2) + L2 * math.sin(fa)

    c1, s1 = math.cos(i1), math.sin(i1)
    return -c1 * r_h, s1 * r_h, z_h


# ── 执行器中心 (末端工具) ───────────────────────────────────────────────────
def tool_center(joints):
    """关节角度 → 执行器中心坐标 (x, y, z).

    执行器固定于腕部电机 (ID4) 输出端, 局部偏移 (TOOL_OFFSET_X, TOOL_OFFSET_Y):
    局部 +x = ID4=0° 时沿小臂方向 (臂平面内向外), 局部 +y = 水平向右 (沿 ID4 转轴).
    ID4 俯仰时 +x 部分在臂平面内随之旋转, +y 部分沿转轴不变.
    """
    i1 = math.radians(joints[1])
    i2 = math.radians(joints[2])
    i3 = math.radians(joints[3])
    i4 = math.radians(joints[4])

    fa = i2 + i3                                   # 小臂绝对角
    r_h = L1 * math.cos(i2) + L2 * math.cos(fa)
    z_h = L1 * math.sin(i2) + L2 * math.sin(fa)

    c1, s1 = math.cos(i1), math.sin(i1)

    # 末端电机局部系 (ID4=0 时): +x 沿小臂方向, +y 水平向右 (垂直臂平面, 即 ID4 转轴)
    ox, oy, oz = -c1 * math.cos(fa), s1 * math.cos(fa), math.sin(fa)   # 局部 x
    rx, ry = s1, c1                                                   # 局部 y (转轴)
    ux, uy, uz = c1 * math.sin(fa), -s1 * math.sin(fa), math.cos(fa)   # 臂平面内 ⊥小臂

    # ID4 绕局部 y (转轴) 旋转: 局部 x 在臂平面内转动
    cx, st = math.cos(i4), math.sin(i4)
    tx = ox * cx + ux * st
    ty = oy * cx + uy * st
    tz = oz * cx + uz * st

    x = -c1 * r_h + TOOL_OFFSET_X * tx + TOOL_OFFSET_Y * rx
    y =  s1 * r_h + TOOL_OFFSET_X * ty + TOOL_OFFSET_Y * ry
    z =  z_h      + TOOL_OFFSET_X * tz
    return x, y, z


# ── 逆运动学 (IK, 臂平面内) ────────────────────────────────────────────────
def inverse_kinematics_plane(r, z):
    """臂平面 (r, z) → (id2, id3).  固定臂朝前 (r_h > 0)，仅肘朝上构型。
    关节限位内无解则抛 ValueError."""
    d = math.hypot(r, z)
    if d > L1 + L2 + 0.001:
        raise ValueError(f"目标太远: {d:.1f} > {L1+L2:.1f} mm")
    if d < abs(L1 - L2) - 0.001:
        raise ValueError(f"目标太近: {d:.1f} < {abs(L1-L2):.1f} mm")

    cos_id3 = (d * d - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    cos_id3 = max(-1.0, min(1.0, cos_id3))
    id3_rad = math.acos(cos_id3)           # acos 返回 [0, π]，天然肘朝上

    gamma = math.atan2(z, r)
    psi = math.atan2(L2 * math.sin(id3_rad),
                     L1 + L2 * math.cos(id3_rad))
    id2 = math.degrees(gamma - psi)
    id3 = math.degrees(id3_rad)

    id2 = (id2 + 180.0) % 360.0 - 180.0
    id3 = (id3 + 180.0) % 360.0 - 180.0

    lo2, hi2 = JOINT_LIMITS[2]
    lo3, hi3 = JOINT_LIMITS[3]
    if lo2 <= id2 <= hi2 and lo3 <= id3 <= hi3:
        return id2, id3

    raise ValueError(f"IK 关节限位内无解: r={r:.1f}, z={z:.1f}")


# ── 机械臂控制 ──────────────────────────────────────────────────────────────
class Arm:
    def __init__(self, port=SERIAL_PORT, baudrate=BAUDRATE):
        self._bus = MotorBus(port=port, addresses=[1, 2, 3, 4], baudrate=baudrate)
        self._motors = {m.address: m for m in self._bus.motors}
        self._bias = {addr: 0.0 for addr in self._motors}
        self._wrist_level_ref = None  # 手腕水平参考角 (deg), 上电时设定

        for m in self._bus.motors:
            try:
                m.enable()
            except Exception as e:
                print(f"  ID={m.address} 使能失败: {e}")
            time.sleep(0.002)
        time.sleep(0.1)

        self._detect_bias()

    # ── 上电编码器偏置检测 ──

    def _detect_bias(self):
        """首次读取编码器: 读数 > BIAS_THRESHOLD 的电机, 本会话角度减去修正量.

        仅软件修正读数, 不移动电机; 补偿上电时多圈计数跳变.
        """
        for addr in self._motors:
            try:
                raw = self._motors[addr].read_status()["multi_turn_deg"]
            except Exception:
                raw = float("nan")
            if math.isnan(raw):
                print(f"  ID={addr}: 首次读取失败, 不修正")
            elif raw > BIAS_THRESHOLD:
                self._bias[addr] = BIAS_CORRECTION
                print(f"  ID={addr}: 首次读数 {raw:.2f}° > {BIAS_THRESHOLD:.0f}°, "
                      f"本会话减去 {abs(BIAS_CORRECTION):.0f}°")
            else:
                print(f"  ID={addr}: 首次读数 {raw:.2f}°, 无需修正")

    # ── 手腕水平维持 ──

    def set_wrist_level_ref(self):
        """设定手腕水平参考: 硬编码水平时 ID4 = 24° (逻辑角), 锚定当前肩/肘姿态。
        此后手腕将自动维持该姿态 (相对于水平面平行)。"""
        joints = self.get_joints()
        # 臂平面内手腕绝对角 = 肩 + 肘 + 腕;  24° 是 ID4 的水平角, 不是绝对角
        self._wrist_level_ref = joints[2] + joints[3] + 24.0
        print(f"手腕水平参考角: {self._wrist_level_ref:.2f}° (ID4 水平位 = 24°)")
        return self._wrist_level_ref

    def get_wrist_target(self, id2, id3):
        """给定目标肩/肘角度, 返回保持手腕水平的 ID4 角度."""
        if self._wrist_level_ref is None:
            return 0.0
        target = self._wrist_level_ref - (id2 + id3)
        lo, hi = JOINT_LIMITS[4]
        return max(lo, min(hi, target))

    # ── 角度转换 ──

    def _to_logical(self, addr, motor_deg):
        return JOINT_OFFSETS[addr] + JOINT_SIGNS[addr] * (motor_deg + self._bias[addr])

    def _to_motor(self, addr, logical_deg):
        return JOINT_SIGNS[addr] * (logical_deg - JOINT_OFFSETS[addr]) - self._bias[addr]

    # ── 关节读写 ──

    def get_joints(self):
        """读取所有关节逻辑角度."""
        result = {}
        for addr in self._motors:
            try:
                s = self._motors[addr].read_status()
                result[addr] = self._to_logical(addr, s["multi_turn_deg"])
            except Exception:
                result[addr] = float("nan")
        return result

    def move_joint(self, addr, logical_deg, speed_rpm=10):
        """单轴移动 (带限位保护)."""
        lo, hi = JOINT_LIMITS[addr]
        clamped = max(lo, min(hi, logical_deg))
        if clamped != logical_deg:
            print(f"  ID{addr}: {logical_deg:.1f}° 超出限位 [{lo}, {hi}], 截断为 {clamped:.1f}°")
        target = self._to_motor(addr, clamped)
        self._motors[addr].set_target_position_speed(target, speed_rpm=speed_rpm, wait=False)

    def move_all(self, angles, speed_rpm=10):
        """多轴同时移动."""
        for addr, deg in angles.items():
            self.move_joint(addr, deg, speed_rpm=speed_rpm)

    def move_to_polar(self, base_deg, r, speed_rpm=10):
        """移动到极坐标位置 (保持当前Z高度).

        Args:
            base_deg: 基座角度 (°) — 电机1旋转角
            r:        前伸距离 (mm) — 水平面内距原点距离
            speed_rpm: 电机转速

        Returns:
            (x, y, z): 目标笛卡尔坐标 (mm)
        """
        joints = self.get_joints()
        _, _, z = forward_kinematics(joints)

        lo1, hi1 = JOINT_LIMITS[1]
        base_deg = max(lo1, min(hi1, base_deg))

        id2, id3 = inverse_kinematics_plane(r, z)
        id4 = self.get_wrist_target(id2, id3)
        self.move_all({1: base_deg, 2: id2, 3: id3, 4: id4}, speed_rpm=speed_rpm)

        id1_rad = math.radians(base_deg)
        x = -r * math.cos(id1_rad)
        y =  r * math.sin(id1_rad)
        return x, y, z

    # ── 资源管理 ──

    def close(self):
        """失能全部电机后关闭总线."""
        for addr in sorted(self._motors.keys()):
            for attempt in range(3):
                try:
                    self._motors[addr].disable()
                    break
                except Exception:
                    time.sleep(0.02)
            else:
                print(f"  ⚠ ID={addr} 失能失败 (重试3次)")
        time.sleep(0.05)
        self._bus.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        try:
            self.close()
        except Exception as e:
            print(f"Arm.close() 失败: {e}")
        return False


# ── ESP32 继电器 (串口) ─────────────────────────────────────────────────────
RELAY_BAUDRATE = 115200
MAG_ON_CMD = "mag_high"    # 吸合
MAG_OFF_CMD = "mag_low"    # 断开
ESP32_VENDOR_HINTS = {
    0x10C4,  # Silicon Labs CP210x
    0x1A86,  # QinHeng CH340/CH341
    0x0403,  # FTDI
    0x303A,  # Espressif 原生 USB (ESP32-S2/S3/C3)
}

# 精确 PID 提示: FT232R (单通道 UART, 0403:6001).
# 仅按 VID=0x0403 匹配会误选 FT4232H 四通道 (0403:6011, 电机总线所在).
ESP32_PID_HINTS = {
    0x6001,  # FT232R — 本机 ESP32 用的转接芯片
}

# ESP32 的 by-id 固定路径 (唯一序列号, 插拔/重启后不变), 最高优先级
RELAY_SERIAL_BY_ID = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0"


def detect_esp32_port():
    """检测 ESP32 所在串口, 找不到返回 None.

    优先级: by-id 固定路径 → 精确 PID (FT232R) → 唯一串口.
    """
    if os.path.exists(RELAY_SERIAL_BY_ID):
        return RELAY_SERIAL_BY_ID
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if p.vid in ESP32_VENDOR_HINTS and p.pid in ESP32_PID_HINTS:
            return p.device
    if len(ports) == 1:
        return ports[0].device
    return None


class Relay:
    """ESP32 继电器 — 串口指令 mag_high/mag_low 切换通断."""

    def __init__(self, port, initial_off=True):
        self._on = False
        self._ser = serial.Serial(port, RELAY_BAUDRATE, timeout=2.0)   # ESP32 回复延迟 ~1s
        time.sleep(1.0)                   # 等 ESP32 启动
        self._ser.reset_input_buffer()    # 丢弃 ESP32 启动信息
        if initial_off:
            self.set(False)               # 上电默认断开, 保证安全

    @property
    def is_on(self):
        return self._on

    def set(self, on):
        cmd = MAG_ON_CMD if on else MAG_OFF_CMD
        self._ser.write((cmd + "\n").encode("utf-8"))
        self._ser.flush()
        reply = self._ser.readline().decode("utf-8", errors="replace").strip()
        if not reply:
            print(f"  ⚠ 继电器无回复: {cmd} — 可能串口不是 ESP32")
        self._on = on

    def toggle(self):
        """切换通断, 返回新状态."""
        self.set(not self._on)
        return self._on

    def close(self):
        try:
            self.set(False)               # 退出时确保断开
        except Exception:
            pass
        self._ser.close()


# ── 主循环 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="四轴机械臂坐标显示 + 键盘控制")
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=BAUDRATE)
    parser.add_argument("--relay-port", default=None,
                        help="ESP32 继电器串口 (默认自动检测)")
    args = parser.parse_args()

    STEP_MM = 2.0    # 平移步长 (mm)
    STEP_DEG = 2.0   # 旋转步长 (°)

    with Arm(port=args.port, baudrate=args.baud) as arm:
        # 手腕水平参考: ID4 水平角硬编码 24°, 锚定当前肩/肘姿态 (此后手腕自动维持水平)
        arm.set_wrist_level_ref()

        # 从当前实际位置初始化目标状态
        joints = arm.get_joints()
        id1_target = joints[1]

        # 由 FK 反推当前臂平面 (r, z)
        x0, y0, z0 = forward_kinematics(joints)
        r_target = math.hypot(x0, y0)       # 径向距离
        z_target = z0

        tx0, ty0, tz0 = tool_center(joints)
        cu0, cv0 = arm_to_camera(tx0, ty0)
        print(f"初始: X={x0:.1f} Y={y0:.1f} Z={z0:.1f}  "
              f"执行器中心 X={tx0:.1f} Y={ty0:.1f} Z={tz0:.1f}  "
              f"相机 U={cu0:6.1f} V={cv0:6.1f}  "
              f"ID1={joints[1]:.1f}° ID2={joints[2]:.1f}° "
              f"ID3={joints[3]:.1f}° ID4={joints[4]:.1f}°")

        # ── 继电器 (F 切换通断) ──
        relay = None
        esp_port = args.relay_port or detect_esp32_port()
        if esp_port:
            try:
                relay = Relay(esp_port)
                print(f"继电器已连接: {esp_port}  (按 F 切换通断)")
            except Exception as e:
                print(f"⚠ 继电器连接失败: {e} — F 键不可用")
        else:
            print("⚠ 未检测到 ESP32, F 键不可用 (可用 --relay-port 指定)")

        f_pending = [False]   # F 按下待处理 (回调线程置位, 主循环消费)

        def _on_key(e):
            if e.name == "f":
                f_pending[0] = True

        keyboard.on_press(_on_key)

        # ── 状态回读节流: 每圈读 4 个电机应答会阻塞总线/终端, 是"不丝滑"的根源 ──
        STATUS_INTERVAL = 0.1          # 状态显示刷新间隔 (10 Hz)
        last_status = time.monotonic()

        try:
            while True:
                if keyboard.is_pressed('esc') or keyboard.is_pressed('q'):
                    print("\n退出")
                    break

                # ── 继电器切换: F ──
                if f_pending[0]:
                    f_pending[0] = False
                    if relay:
                        on = relay.toggle()
                        print(f"\n继电器 {'吸合 ON' if on else '断开 OFF'} "
                              f"({MAG_ON_CMD if on else MAG_OFF_CMD})")
                    else:
                        print("\n⚠ 继电器未连接")

                dirty = False

                # ── 基座旋转: A/D ──
                if keyboard.is_pressed('a'):
                    id1_target -= STEP_DEG; dirty = True
                if keyboard.is_pressed('d'):
                    id1_target += STEP_DEG; dirty = True

                # ── 臂伸缩: W/S (W=收回/向+X, S=伸出) ──
                if keyboard.is_pressed('w'):
                    r_target -= STEP_MM; dirty = True
                if keyboard.is_pressed('s'):
                    r_target += STEP_MM; dirty = True

                # ── Z升降: Shift/Ctrl ──
                if keyboard.is_pressed('shift'):
                    z_target += STEP_MM; dirty = True
                if keyboard.is_pressed('ctrl'):
                    z_target -= STEP_MM; dirty = True

                # ── 控制台输入: <基座角度°> <前伸量mm> [Zmm] ──
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 1:
                            try:
                                if len(parts) >= 1:
                                    id1_target = float(parts[0]); dirty = True
                                if len(parts) >= 2:
                                    r_target = -float(parts[1]); dirty = True   # 前伸量取反
                                if len(parts) >= 3:
                                    z_target = float(parts[2]); dirty = True
                            except ValueError:
                                print(f"\n⚠ 格式错误，请输入数字，例如: 10 100")

                if dirty:
                    r_old, z_old = r_target, z_target
                    id1_old = id1_target

                    lo1, hi1 = JOINT_LIMITS[1]
                    id1_target = max(lo1, min(hi1, id1_target))

                    try:
                        id2, id3 = inverse_kinematics_plane(r_target, z_target)
                        id4 = arm.get_wrist_target(id2, id3)
                        arm.move_all({1: id1_target, 2: id2, 3: id3, 4: id4})
                    except ValueError:
                        # 关节限位内不可达: 撤销按键，停在边界
                        r_target, z_target = r_old, z_old
                        id1_target = id1_old

                # ── 状态回读与显示: 节流到 10 Hz, 不阻塞控制指令 ──
                now = time.monotonic()
                if now - last_status >= STATUS_INTERVAL:
                    last_status = now
                    joints = arm.get_joints()
                    ax, ay, az = forward_kinematics(joints)
                    tx, ty, tz = tool_center(joints)
                    cu, cv = arm_to_camera(tx, ty)

                    mag = "ON" if (relay and relay.is_on) else "OFF"
                    print(f"\r实际 X={ax:+8.1f} Y={ay:+8.1f} Z={az:+8.1f}  |  "
                          f"ID1={joints[1]:+7.1f}° ID2={joints[2]:+7.1f}° "
                          f"ID3={joints[3]:+7.1f}° ID4={joints[4]:+7.1f}°  |  "
                          f"目标 r={r_target:+7.1f}mm Z={z_target:+7.1f}mm  |  "
                          f"MAG={mag}  |  "
                          f"执行器 X={tx:+8.1f} Y={ty:+8.1f} Z={tz:+8.1f}  |  "
                          f"相机 U={cu:6.1f} V={cv:6.1f}",
                          end="", flush=True)
                elif not dirty:
                    time.sleep(0.005)   # 空闲时让出 CPU, 按键轮询仍保持 ~200 Hz

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            if relay:
                relay.close()
                print("继电器已断开 (mag_low)")


if __name__ == "__main__":
    main()
