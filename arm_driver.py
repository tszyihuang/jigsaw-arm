#!/usr/bin/env python3
"""四轴机械臂 — 坐标显示 + 键盘控制 (极简版)

坐标系: +X=左 +Y=前 +Z=上  (右手定则)
按键: W/S→臂伸缩  A/D→基座旋转  Shift/Ctrl→Z升降  空格→继电器开关  Q/ESC→退出
"""

import math
import os
import signal
import time
import argparse
import subprocess
import keyboard

from GIM4310_driver import MotorBus, SERIAL_PORT, BAUDRATE

# ── 继电器控制 ────────────────────────────────────────────────────────────────
RELAY_CHIP = "gpiochip0"
RELAY_LINE = 112       # Pin 11 → PR.04


class Relay:
    """通过 gpioset --mode=background 控制继电器.

    gpiod v1 默认 mode=exit 会在退出时释放 GPIO，电平不保持，
    因此使用 background 模式让进程常驻持住电平。
    低电平触发模块: 0=吸合(开), 1=断开(关).
    """

    def __init__(self, chip=RELAY_CHIP, line=RELAY_LINE, active_low=True):
        self._chip = chip
        self._line = line
        self._active_low = active_low
        self._state = False     # 逻辑状态: False=关, True=开
        self._proc = None       # 后台 gpioset 进程
        self.off()              # 初始化为关闭

    def _set(self, value):
        """启动后台 gpioset 进程持住电平，同时杀掉旧进程."""
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    os.kill(self._proc.pid, signal.SIGTERM)
                    self._proc.wait(timeout=2)
            except Exception:
                pass
        self._proc = subprocess.Popen(
            ["gpioset", "--mode=background", self._chip, f"{self._line}={value}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    def on(self):
        val = 0 if self._active_low else 1
        self._set(val)
        self._state = True

    def off(self):
        val = 1 if self._active_low else 0
        self._set(val)
        self._state = False

    def toggle(self):
        if self._state:
            self.off()
        else:
            self.on()
        return self._state

    @property
    def is_on(self):
        return self._state

    def close(self):
        self.off()
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    os.kill(self._proc.pid, signal.SIGTERM)
                    self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None

# ── 机械臂尺寸 (mm) ─────────────────────────────────────────────────────────
L1 = 150.0   # 大臂: 肩→肘
L2 = 150.0   # 小臂: 肘→腕
L3 = 0.0     # 手部: 腕→末端

# ── 关节限位 (逻辑角度, °) ──────────────────────────────────────────────────
JOINT_LIMITS  = {1: (-90, 90), 2: (-5, 180), 3: (0, 160), 4: (-90, 90)}  # ID4 自动维持水平

# ── 电机映射: 逻辑角 ↔ 电机绝对角 ──────────────────────────────────────────
JOINT_SIGNS   = {1: +1, 2: +1, 3: -1, 4: +1}
JOINT_OFFSETS = {1: 0,  2: 0,  3: 160, 4: 0}


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
        self._origins = {}
        self._wrist_level_ref = None  # 手腕水平参考角 (deg), 校准后设定

        for m in self._bus.motors:
            try:
                m.enable()
            except Exception as e:
                print(f"  ID={m.address} 使能失败: {e}")
            time.sleep(0.002)
        time.sleep(0.1)

    # ── 原点标定 ──

    def calibrate(self):
        """标定原点 — 请先将机械臂摆成蜷缩姿态:
        ID1=0 (基座居中), ID2=0 (大臂水平), ID3=160° (肘完全弯折), ID4=0"""
        print("原点标定中... 请保持蜷缩姿态")
        samples = {addr: [] for addr in self._motors}

        for _ in range(10):
            for addr, m in self._motors.items():
                try:
                    samples[addr].append(m.read_status()["multi_turn_deg"])
                except Exception:
                    pass
            time.sleep(0.03)

        for addr, vals in samples.items():
            if not vals:
                raise RuntimeError(f"电机 ID={addr} 标定失败: 无有效读数")
            self._origins[addr] = sum(vals) / len(vals)
            print(f"  ID={addr}: origin = {self._origins[addr]:.2f}°")
        print("标定完成\n")

    # ── 手腕水平维持 ──

    def set_wrist_level_ref(self):
        """记录当前手腕绝对角度为水平参考。
        此后手腕将自动维持该角度 (相对于水平面平行)。"""
        joints = self.get_joints()
        # 臂平面内手腕绝对角 = 肩 + 肘 + 腕
        self._wrist_level_ref = joints[2] + joints[3] + joints[4]
        print(f"手腕水平参考角: {self._wrist_level_ref:.2f}°")
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
        return JOINT_OFFSETS[addr] + JOINT_SIGNS[addr] * (motor_deg - self._origins[addr])

    def _to_motor(self, addr, logical_deg):
        return self._origins[addr] + JOINT_SIGNS[addr] * (logical_deg - JOINT_OFFSETS[addr])

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

    def move_joint(self, addr, logical_deg, speed_rpm=30):
        """单轴移动 (带限位保护)."""
        lo, hi = JOINT_LIMITS[addr]
        clamped = max(lo, min(hi, logical_deg))
        if clamped != logical_deg:
            print(f"  ID{addr}: {logical_deg:.1f}° 超出限位 [{lo}, {hi}], 截断为 {clamped:.1f}°")
        target = self._to_motor(addr, clamped)
        self._motors[addr].set_target_position_speed(target, speed_rpm=speed_rpm, wait=False)

    def move_all(self, angles, speed_rpm=30):
        """多轴同时移动."""
        for addr, deg in angles.items():
            self.move_joint(addr, deg, speed_rpm=speed_rpm)

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


# ── 主循环 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="四轴机械臂坐标显示 + 键盘控制")
    parser.add_argument("--port", default=SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=BAUDRATE)
    args = parser.parse_args()

    STEP_MM = 2.0    # 平移步长 (mm)
    STEP_DEG = 2.0   # 旋转步长 (°)

    with Arm(port=args.port, baudrate=args.baud) as arm:
        arm.calibrate()

        # 记录当前手腕角度为水平参考 (此后手腕自动维持水平)
        arm.set_wrist_level_ref()

        # 从当前实际位置初始化目标状态
        joints = arm.get_joints()
        id1_target = joints[1]

        # 由 FK 反推当前臂平面 (r, z)
        x0, y0, z0 = forward_kinematics(joints)
        r_target = math.hypot(x0, y0)       # 径向距离
        z_target = z0

        print(f"初始: X={x0:.1f} Y={y0:.1f} Z={z0:.1f}  "
              f"ID1={id1_target:.1f}°")

        # ── 继电器 ──
        relay = Relay()
        space_was_pressed = False
        print(f"继电器: {'开 🔴' if relay.is_on else '关 ⚫'} (空格切换)")

        try:
            while True:
                if keyboard.is_pressed('esc') or keyboard.is_pressed('q'):
                    print("\n退出")
                    break

                dirty = False

                # ── 继电器开关: 空格 (上升沿触发, 防抖) ──
                space_now = keyboard.is_pressed('space')
                if space_now and not space_was_pressed:
                    relay.toggle()
                space_was_pressed = space_now

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

                # ── 读一次当前状态 ──
                joints = arm.get_joints()
                ax, ay, az = forward_kinematics(joints)

                tx, ty, tz = ax, ay, az  # 默认: 目标=实际

                if dirty:
                    r_old, z_old = r_target, z_target
                    id1_old = id1_target

                    lo1, hi1 = JOINT_LIMITS[1]
                    id1_target = max(lo1, min(hi1, id1_target))

                    try:
                        id2, id3 = inverse_kinematics_plane(r_target, z_target)
                        id4 = arm.get_wrist_target(id2, id3)
                        arm.move_all({1: id1_target, 2: id2, 3: id3, 4: id4})
                        id1_rad = math.radians(id1_target)
                        tx = -r_target * math.cos(id1_rad)
                        ty =  r_target * math.sin(id1_rad)
                        tz = z_target
                    except ValueError:
                        # 关节限位内不可达: 撤销按键，停在边界
                        r_target, z_target = r_old, z_old
                        id1_target = id1_old

                print(f"\r实际 X={ax:+8.1f} Y={ay:+8.1f} Z={az:+8.1f}  |  "
                      f"目标 X={tx:+8.1f} Y={ty:+8.1f} Z={tz:+8.1f}  |  "
                      f"J1={joints[1]:+7.1f} J2={joints[2]:+7.1f} J3={joints[3]:+7.1f} J4={joints[4]:+7.1f}  |  "
                      f"{'🔴 开' if relay.is_on else '⚫ 关'}",
                      end="", flush=True)

                # time.sleep(0.05)  # 已移除，最大化控制频率

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            relay.close()
            print("继电器已关闭")


if __name__ == "__main__":
    main()
