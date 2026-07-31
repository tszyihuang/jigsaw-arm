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


# ── 笛卡尔 ↔ 极坐标 ─────────────────────────────────────────────────────────
def cartesian_to_polar(x, y):
    """笛卡尔 (x, y) → 极坐标 (基座角 θ1 [°], 前伸量 r [mm]).

    由 FK 反解: x = -r·cos(θ1), y = r·sin(θ1):
        r  = √(x² + y²)
        θ1 = atan2(y, -x)
    """
    r = math.hypot(x, y)
    base_deg = math.degrees(math.atan2(y, -x))
    return base_deg, r


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

    @property
    def wrist_level_ref(self):
        """手腕水平参考角 (°), 未锚定时为 None."""
        return self._wrist_level_ref

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


# ── 机械臂笛卡尔控制层 ──────────────────────────────────────────────────────
class CartesianArm:
    """机械臂笛卡尔控制层 — 目标状态跟踪 + 绝对笛卡尔移动 (执行器中心).

    在 Arm (电机级 API) 之上封装坐标级控制:
      - 待机 / 回到待机: go_standby / go_home
      - 腕部 / 执行器中心绝对坐标移动: move_to_cartesian / move_tool_to
      - 目标状态跟踪与等待到达: target / wait_for_arrival

    move_to_cartesian 采用 r<0 约定: r = -√(x²+y²), θ1 = atan2(y, x),
    使正 X 半平面 θ1 ∈ [-90°, 90°], 避开基座 ±90° 限位.
    move_tool_to 求解执行器中心 (腕部 + TOOL_OFFSET_X/Y 偏移, 见 tool_center):
    基座角由横向约束 x·sinθ1 + y·cosθ1 = TOOL_OFFSET_Y 解析确定,
    臂平面内腕部目标 = 工具目标 - TOOL_OFFSET_X·(cosREF, sinREF),
    REF = 手腕水平参考角 (ID4 维持水平 → 工具俯仰角恒定), 再迭代精修;
    tool_to_wrist 转换后喂入 move_to_cartesian (腕部) 移动.
    """

    def __init__(self, port=SERIAL_PORT, baudrate=BAUDRATE):
        self.arm = Arm(port=port, baudrate=baudrate)
        self.arm.set_wrist_level_ref()    # 锚定手腕水平参考 (ID4 自动维持水平)
        self._target = None               # 当前目标极坐标状态 (待机或首次移动时初始化)

    # ── 待机 ──

    def go_standby(self, x, y, z, speed_rpm=10.0):
        """移动到待机位置 (笛卡尔 X, Y, Z) 并更新目标状态.

        Returns:
            target: 目标关节角 {1..4: deg}, 供 wait_for_arrival 使用
        """
        base_deg, r = cartesian_to_polar(x, y)
        id2, id3 = inverse_kinematics_plane(r, z)
        id4 = self.arm.get_wrist_target(id2, id3)
        self._target = {"base": base_deg, "r": r, "z": z}

        target = {1: base_deg, 2: id2, 3: id3, 4: id4}
        print(f"待机目标 → 极坐标: 基座 {base_deg:.1f}°  r={r:.1f}mm  Z={z:.1f}mm")
        print(f"  关节: ID1={base_deg:.1f}°  ID2={id2:.1f}°  ID3={id3:.1f}°  ID4={id4:.1f}°")
        self.arm.move_all(target, speed_rpm=speed_rpm)
        return target

    def wait_for_arrival(self, target, tolerance=2.0, timeout=30.0):
        """等待所有关节到达目标角度."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            joints = self.arm.get_joints()
            if all(abs(joints[a] - target[a]) <= tolerance for a in target):
                return True
            time.sleep(0.1)
        return False

    def go_home(self, x, y, z, wait=True, speed_rpm=10.0):
        """回到待机位置 (x, y, z) 并等待到达."""
        target = self.go_standby(x, y, z, speed_rpm=speed_rpm)
        if wait:
            if self.wait_for_arrival(target, timeout=15.0):
                print("  ✓ 已到达待机位置")
            else:
                print("  ⚠ 等待超时未到达")
        return target

    # ── 绝对坐标移动 ──

    def move_to_cartesian(self, x, y, z=None, wait=True, speed_rpm=10.0):
        """移动到腕部绝对笛卡尔坐标 (X, Y, Z).

        内部转换为极坐标 (基座角 θ1, 前伸 r) 后 IK 求解移动 (r<0 约定):
            r = -√(x²+y²),  θ1 = atan2(y, x)

        Args:
            x, y: 腕部笛卡尔水平坐标 (mm)
            z:    Z 高度 (mm), None 时保持当前 Z
            wait: 是否等待到达 (超时 15s)
            speed_rpm: 电机转速

        Returns:
            target: 目标关节角 {1..4: deg}; 超出限位 / IK 无解时返回 None (未移动)
        """
        if self._target is None:
            self._init_target_from_actual()
        if z is None:
            z = self._target["z"]

        base_deg, r = cartesian_to_polar(-x, y)
        r = -r
        lo1, hi1 = JOINT_LIMITS[1]
        if not (lo1 <= base_deg <= hi1):
            print(f"  ⚠ 基座角度 {base_deg:+.1f}° 超出限位 [{lo1}, {hi1}] — 保持原位置")
            return None
        try:
            id2, id3 = inverse_kinematics_plane(r, z)
        except ValueError as e:
            print(f"  ⚠ {e} — 保持原位置")
            return None
        id4 = self.arm.get_wrist_target(id2, id3)

        self._target["base"], self._target["r"], self._target["z"] = base_deg, r, z
        target = {1: base_deg, 2: id2, 3: id3, 4: id4}
        self.arm.move_all(target, speed_rpm=speed_rpm)

        print(f"  目标 → 笛卡尔 X={x:+.1f}  Y={y:+.1f}  Z={z:+.1f} mm  "
              f"(极坐标 基座 {base_deg:+.1f}°  r={r:+.1f}mm)")
        if wait:
            if self.wait_for_arrival(target, timeout=15.0):
                print("  ✓ 已到达")
            else:
                print("  ⚠ 等待超时未到达")
        return target

    def tool_to_wrist(self, x, y, z=None):
        """执行器中心坐标 (x, y, z) → 腕部坐标 (真实几何腕部).

        解析求解, 偏移定义参考 tool_center:
          ① 基座角: 工具横向分量恒等于 TOOL_OFFSET_Y (与 ID4/臂型无关)
             → x·sinθ1 + y·cosθ1 = TOOL_OFFSET_Y, 两分支取基座限位内者
          ② 臂平面内: 腕部目标 = 工具目标 - TOOL_OFFSET_X·(cosREF, sinREF),
             REF = 手腕水平参考角 (ID4 维持水平 → 工具俯仰角恒定)
          ③ 用 tool_center 校验残差并迭代精修 (兜底 ID4 限位夹紧)

        Args:
            x, y: 执行器中心水平坐标 (mm)
            z:    执行器中心 Z 高度 (mm), None 时保持当前 Z

        Returns:
            (wx, wy, wz): 腕部坐标 (mm); 不可达时返回 None.
            注意: 直接喂入 move_to_cartesian 时 y 需取反 (见 move_tool_to).
        """
        if z is None:
            _, _, z = tool_center(self.arm.get_joints())   # 保持当前执行器 Z

        # ① 基座角: 由横向约束 x·sinθ1 + y·cosθ1 = TOOL_OFFSET_Y 解析求解
        rho = math.hypot(x, y)
        if rho < abs(TOOL_OFFSET_Y):
            print(f"  ⚠ 目标距轴心 {rho:.1f}mm < 执行器横向偏移 "
                  f"{abs(TOOL_OFFSET_Y):.0f}mm, 执行器中心不可达 — 保持原位置")
            return None
        alpha = math.atan2(y, x)
        t = math.asin(TOOL_OFFSET_Y / rho)
        lo1, hi1 = JOINT_LIMITS[1]
        base_deg = None
        for cand in (-alpha + t, -alpha + math.pi - t):    # 两分支 (左右镜像臂)
            d = math.degrees(cand)
            d = (d + 180.0) % 360.0 - 180.0
            if lo1 <= d <= hi1:
                base_deg = d
                break
        if base_deg is None:
            print("  ⚠ 目标方位对应基座角超出限位 — 保持原位置")
            return None

        # ② 臂平面内: 腕部目标 = 工具目标 - 工具偏移 (方向恒为手腕水平参考角)
        ref = self.arm.wrist_level_ref
        if ref is None:
            ref = 24.0
        ref_rad = math.radians(ref)
        base_rad = math.radians(base_deg)
        r_tool = -math.cos(base_rad) * x + math.sin(base_rad) * y   # 工具径向分量
        r_h = r_tool - TOOL_OFFSET_X * math.cos(ref_rad)
        z_h = z - TOOL_OFFSET_X * math.sin(ref_rad)

        # ③ 迭代精修 (解析解在 ID4 限位内已精确, 兜底夹紧/极端姿态)
        err = float("inf")
        for _ in range(10):
            try:
                id2, id3 = inverse_kinematics_plane(r_h, z_h)
            except ValueError as e:
                print(f"  ⚠ {e} — 保持原位置")
                return None
            id4 = self.arm.get_wrist_target(id2, id3)
            joints = {1: base_deg, 2: id2, 3: id3, 4: id4}

            tx, ty, tz = tool_center(joints)
            err = math.sqrt((tx - x) ** 2 + (ty - y) ** 2 + (tz - z) ** 2)
            if err < 0.5:
                break
            # 残差回代 (阻尼 0.5): 横向由 θ1 严格保证, 仅修正臂平面内分量
            r_h += 0.5 * (r_tool - (-math.cos(base_rad) * tx + math.sin(base_rad) * ty))
            z_h += 0.5 * (z - tz)
        if err > 1.0:
            print(f"  ⚠ 执行器中心未完全收敛 (残差 {err:.1f}mm) — 目标可能不可达")

        # 腕部真实坐标 (r_h<0 折叠构型): x = -cosθ1·r_h, y = sinθ1·r_h
        return -math.cos(base_rad) * r_h, math.sin(base_rad) * r_h, z_h

    def move_tool_to(self, x, y, z=None, wait=True, speed_rpm=10.0):
        """移动执行器中心到绝对笛卡尔坐标 (X, Y, Z).

        执行器中心 = 腕部 + 局部偏移 (TOOL_OFFSET_X=12, TOOL_OFFSET_Y=-36,
        见 tool_center). 流程: tool_to_wrist 转换 → move_to_cartesian.

        Returns:
            target: 目标关节角 {1..4: deg}; 不可达时返回 None (未移动)
        """
        res = self.tool_to_wrist(x, y, z)
        if res is None:
            return None
        wx, wy, wz = res
        # y 镜像补偿: move_to_cartesian 的 θ1 = atan2(y, x) 约定使腕部落点为 (wx, -wy),
        # 故传入 (wx, -wy), 实际腕部落在 (wx, wy), 执行器中心正好在 (x, y, z)
        target = self.move_to_cartesian(wx, -wy, wz, wait=wait, speed_rpm=speed_rpm)
        if target is None:
            return None
        tx, ty, tz = tool_center(target)
        print(f"  ✓ 执行器中心预计到达 X={tx:+.1f}  Y={ty:+.1f}  Z={tz:+.1f} mm")
        return target

    def _init_target_from_actual(self):
        """从实际关节角初始化目标状态 (未启动时调用的兜底)."""
        joints = self.arm.get_joints()
        x, y, z = forward_kinematics(joints)
        base_deg, r = cartesian_to_polar(x, y)
        self._target = {"base": base_deg, "r": r, "z": z}

    @property
    def target(self):
        """当前目标状态 (基座角°, r mm, z mm) — 供外部读取, 返回副本."""
        return None if self._target is None else dict(self._target)

    def status(self):
        """当前实际关节角 + 腕部/执行器中心坐标."""
        joints = self.arm.get_joints()
        return joints, forward_kinematics(joints), tool_center(joints)

    # ── 资源管理 ──

    def close(self):
        self.arm.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


# ── ESP32 继电器 (串口) ─────────────────────────────────────────────────────
RELAY_BAUDRATE = 115200
MAG_ON_CMD = "mag_high"    # 吸合
MAG_OFF_CMD = "mag_low"    # 断开
LED_RED_ON  = "red_on"     # 红灯亮
LED_RED_OFF = "red_off"    # 红灯灭
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


class Esp32Cmd:
    """ESP32 指令客户端 — 串口发送命令并等待回复 (LED 控制)."""

    def __init__(self, port):
        self._ser = serial.Serial(port, RELAY_BAUDRATE, timeout=2.0)
        time.sleep(1.0)                   # 等 ESP32 启动
        self._ser.reset_input_buffer()    # 丢弃 ESP32 启动信息

    def send(self, cmd):
        self._ser.write((cmd + "\n").encode("utf-8"))
        self._ser.flush()
        reply = self._ser.readline().decode("utf-8", errors="replace").strip()
        if not reply:
            print(f"  ⚠ ESP32 无回复: {cmd} — 可能串口不是 ESP32")
        elif "OK" not in reply:
            print(f"  ⚠ ESP32 回复异常: {cmd} → {reply}")
        else:
            print(f"  ✓ {cmd} → {reply}")
        return reply

    def red_on(self):
        """红灯亮."""
        return self.send(LED_RED_ON)

    def red_off(self):
        """红灯灭."""
        return self.send(LED_RED_OFF)

    def close(self):
        try:
            self.red_off()                # 退出时确保红灯熄灭
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
