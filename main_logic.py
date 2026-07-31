#!/usr/bin/env python3
"""机械臂主逻辑 — 第一步: 启动流程 + 控制台控制 (增量 / 绝对笛卡尔)

启动后:
  1. 机械臂移动到待机位置 (实际笛卡尔坐标 X=-9, Y=0, Z=+80 mm)
  2. 向 ESP32 发送 red_on, 红灯亮起, 并保持待机
  3. 控制台输入指令移动机械臂:
       增量 (相对当前位置): <Δ角度°> [Δ前伸mm] [ΔZmm]
         例: 10 100   → 基座 +10°, 前伸 +100mm (Z 不变)
             -5 20 10 → 基座 -5°, 前伸 +20mm, Z +10mm
       绝对 (笛卡尔坐标): xyz <Xmm> <Ymm> [Zmm]
         例: xyz 100 50 80 → 移动到 (X=100, Y=50, Z=80)

极坐标换算 (由 arm_driver.forward_kinematics 的 FK 反解):
    FK:  x = -r·cos(θ1),  y = r·sin(θ1)
    反解: r = √(x²+y²),   θ1 = atan2(y, -x)
    待机: r = √((-9)²+0²) = 9.0 mm,  θ1 = atan2(0, 9) = 0°
    Z 不变 (80 mm), 臂平面内用 inverse_kinematics_plane(r, z) 解 ID2/ID3,
    ID4 由手腕水平参考自动维持.

用法:
    python3 main_logic.py                  # 自动检测 ESP32 串口
    python3 main_logic.py --esp-port /dev/ttyUSB0
"""

import argparse
import math
import select
import sys
import time

import serial

from arm_driver import (
    Arm,
    JOINT_LIMITS,
    detect_esp32_port,
    forward_kinematics,
    inverse_kinematics_plane,
)

# ── 待机位置 (实际笛卡尔坐标, mm) ───────────────────────────────────────────
STANDBY_X = -9.0
STANDBY_Y = 0.0
STANDBY_Z = 80.0

STANDBY_SPEED_RPM = 10.0   # 待机移动转速 (rpm)

# ── ESP32 (LED) ─────────────────────────────────────────────────────────────
ESP32_BAUDRATE = 115200
LED_RED_ON  = "red_on"
LED_RED_OFF = "red_off"


def cartesian_to_polar(x, y):
    """笛卡尔 (x, y) → 极坐标 (基座角 θ1 [°], 前伸量 r [mm]).

    由 FK 反解: x = -r·cos(θ1), y = r·sin(θ1):
        r  = √(x² + y²)
        θ1 = atan2(y, -x)
    """
    r = math.hypot(x, y)
    base_deg = math.degrees(math.atan2(y, -x))
    return base_deg, r


class Esp32Cmd:
    """ESP32 指令客户端 — 串口发送命令并等待回复."""

    def __init__(self, port):
        self._ser = serial.Serial(port, ESP32_BAUDRATE, timeout=2.0)
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


class MainLogic:
    """机械臂主逻辑 — 启动流程 + 控制台控制 (增量 / 绝对笛卡尔)."""

    def __init__(self, esp_port=None):
        self._target = None               # 当前目标极坐标状态 (startup 或首次移动时初始化)
        self.arm = Arm()
        self.arm.set_wrist_level_ref()    # 锚定手腕水平参考 (ID4 自动维持水平)

        esp_port = esp_port or detect_esp32_port()
        self.esp = None
        if esp_port:
            try:
                self.esp = Esp32Cmd(esp_port)
                print(f"ESP32 已连接: {esp_port}")
            except Exception as e:
                print(f"⚠ ESP32 连接失败: {e} — 绿灯指令不可用")
        else:
            print("⚠ 未检测到 ESP32 — 绿灯指令不可用 (可用 --esp-port 指定)")

    # ── 待机 ──

    def go_standby(self, speed_rpm=STANDBY_SPEED_RPM):
        """移动到待机位置: 笛卡尔 (X, Y, Z) → 极坐标 → 臂平面 IK.

        Returns:
            target: 目标关节角 {1..4: deg}, 供 wait_for_arrival 使用
        """
        base_deg, r = cartesian_to_polar(STANDBY_X, STANDBY_Y)
        id2, id3 = inverse_kinematics_plane(r, STANDBY_Z)
        id4 = self.arm.get_wrist_target(id2, id3)

        target = {1: base_deg, 2: id2, 3: id3, 4: id4}
        print(f"待机目标 → 极坐标: 基座 {base_deg:.1f}°  r={r:.1f}mm  Z={STANDBY_Z:.1f}mm")
        print(f"  关节: ID1={base_deg:.1f}°  ID2={id2:.1f}°  ID3={id3:.1f}°  ID4={id4:.1f}°")
        self.arm.move_all(target, speed_rpm=speed_rpm)
        return target

    def wait_for_arrival(self, target, tolerance=2.0, timeout=30.0):
        """等待所有关节到达目标角度 (供后续步骤使用)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            joints = self.arm.get_joints()
            if all(abs(joints[a] - target[a]) <= tolerance for a in target):
                return True
            time.sleep(0.1)
        return False

    # ── 移动接口 (增量 / 绝对笛卡尔) ──

    def move_by_delta(self, d_base_deg, d_r_mm, d_z_mm=0.0, wait=True,
                      speed_rpm=STANDBY_SPEED_RPM):
        """增量移动机械臂 (相对当前目标位置, 移动成功后更新目标状态).

        Args:
            d_base_deg: 基座角度增量 (°)
            d_r_mm:     前伸量增量 (mm) — 正值向 +X 方向伸出, 与极坐标 r 方向相反 (取反)
            d_z_mm:     Z 高度增量 (mm)
            wait:       是否等待到达 (超时 15s)
            speed_rpm:  电机转速

        Returns:
            target: 目标关节角 {1..4: deg}; 超限位截断或 IK 无解时返回 None (未移动)
        """
        if self._target is None:
            self._init_target_from_actual()

        base = self._target["base"] + d_base_deg
        lo1, hi1 = JOINT_LIMITS[1]
        clamped = max(lo1, min(hi1, base))
        if clamped != base:
            print(f"  ⚠ 基座角度 {base:+.1f}° 超出限位 [{lo1}, {hi1}], "
                  f"截断为 {clamped:+.1f}°")
        base = clamped

        r = self._target["r"] - d_r_mm          # Δ前伸取反: 正输入 → r 减小 → +X
        z = self._target["z"] + d_z_mm
        try:
            id2, id3 = inverse_kinematics_plane(r, z)
        except ValueError as e:
            print(f"  ⚠ {e} — 保持原位置")
            return None
        id4 = self.arm.get_wrist_target(id2, id3)

        self._target["base"], self._target["r"], self._target["z"] = base, r, z
        target = {1: base, 2: id2, 3: id3, 4: id4}
        self.arm.move_all(target, speed_rpm=speed_rpm)

        x, y, zc = forward_kinematics(target)
        print(f"  目标 → 基座 {base:+.1f}°  r={r:+.1f}mm  Z={z:+.1f}mm  "
              f"(笛卡尔 X={x:+.1f}  Y={y:+.1f}  Z={zc:+.1f})")
        if wait:
            if self.wait_for_arrival(target, timeout=15.0):
                print("  ✓ 已到达")
            else:
                print("  ⚠ 等待超时未到达")
        return target

    def move_to_cartesian(self, x, y, z=None, wait=True,
                          speed_rpm=STANDBY_SPEED_RPM):
        """移动到绝对笛卡尔坐标 (X, Y, Z).

        内部转换为极坐标 (基座角 θ1, 前伸 r) 后 IK 求解移动:
            r  = √(x²+y²),  θ1 = atan2(y, -x)

        Args:
            x, y: 笛卡尔水平坐标 (mm)
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

    def _init_target_from_actual(self):
        """从实际关节角初始化目标状态 (未启动时调用 move_by_delta 的兜底)."""
        joints = self.arm.get_joints()
        x, y, z = forward_kinematics(joints)
        base_deg, r = cartesian_to_polar(x, y)
        self._target = {"base": base_deg, "r": r, "z": z}

    @property
    def target(self):
        """当前目标状态 (基座角°, r mm, z mm) — 供外部读取, 返回副本."""
        return None if self._target is None else dict(self._target)

    # ── 控制台解析 ──

    def _print_usage(self):
        """打印控制台指令说明."""
        print("控制台指令 (增量):  <Δ角度°> [Δ前伸mm] [ΔZmm]")
        print("控制台指令 (绝对):  xyz <Xmm> <Ymm> [Zmm]")
        print("  例: 10 100        → 基座 +10°, 前伸 +100mm (Z 不变)")
        print("      -5 20 10      → 基座 -5°, 前伸 +20mm, Z +10mm")
        print("      xyz 100 50 80 → 移动到 (X=100, Y=50, Z=80)")
        print("      status        → 显示当前位置     ? / help → 本帮助")

    def _show_status(self):
        """显示当前实际关节角与末端坐标."""
        joints = self.arm.get_joints()
        x, y, z = forward_kinematics(joints)
        print(f"  实际 ID1={joints[1]:+.1f}°  ID2={joints[2]:+.1f}°  "
              f"ID3={joints[3]:+.1f}°  ID4={joints[4]:+.1f}°")
        print(f"        末端 X={x:+.1f}  Y={y:+.1f}  Z={z:+.1f} mm")

    def _apply_command(self, line):
        """解析控制台指令并转调对应接口: 增量 / 绝对笛卡尔."""
        parts = line.split()
        if parts[0] in ("?", "h", "help"):
            self._print_usage()
            return
        if parts[0] in ("xyz", "goto"):
            self._apply_cartesian(parts[1:])
            return
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            print("  ⚠ 格式错误, 请输入: <Δ角度°> [Δ前伸mm] [ΔZmm]  例如: 10 100")
            print("      或绝对坐标: xyz <Xmm> <Ymm> [Zmm]  例如: xyz 100 50 80")
            return
        if len(vals) > 3:
            print("  ⚠ 参数过多, 最多 3 个: <Δ角度°> [Δ前伸mm] [ΔZmm]")
            return
        d_base, d_r, d_z = (vals + [0.0, 0.0, 0.0])[:3]
        self.move_by_delta(d_base, d_r, d_z)

    def _apply_cartesian(self, parts):
        """解析绝对笛卡尔指令: xyz <Xmm> <Ymm> [Zmm], 转调 move_to_cartesian."""
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            print("  ⚠ 格式错误, 请输入: xyz <Xmm> <Ymm> [Zmm]  例如: xyz 100 50 80")
            return
        if len(vals) < 2:
            print("  ⚠ 至少需要 X 和 Y: xyz <Xmm> <Ymm> [Zmm]")
            return
        if len(vals) > 3:
            print("  ⚠ 参数过多: xyz <Xmm> <Ymm> [Zmm]")
            return
        x, y = vals[0], vals[1]
        z = vals[2] if len(vals) > 2 else None
        self.move_to_cartesian(x, y, z)

    # ── 第一步: 启动流程 ──

    def startup(self):
        """① 移动到待机位置  ② 红灯亮起  ③ 初始化目标状态."""
        print("── 第一步: 启动流程 ──")
        self.go_standby()
        base_deg, r = cartesian_to_polar(STANDBY_X, STANDBY_Y)
        self._target = {"base": base_deg, "r": r, "z": STANDBY_Z}
        if self.esp:
            self.esp.red_on()
        else:
            print("⚠ ESP32 未连接, 跳过 red_on")

    def run(self):
        """主逻辑入口 — 启动完成后进入控制台增量控制循环."""
        self.startup()                        # 内部初始化 _target 为待机极坐标

        print(f"\n✓ 启动完成: 机械臂待机 (X={STANDBY_X:.0f}, Y={STANDBY_Y:.0f}, "
              f"Z={STANDBY_Z:.0f}), 红灯亮")
        self._print_usage()
        print("按 Ctrl+C 退出")

        try:
            while True:
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline()
                    if line == "":            # EOF (Ctrl+D)
                        print("\n输入结束, 退出")
                        break
                    line = line.strip()
                    if not line:
                        continue
                    if line in ("status", "s"):
                        self._show_status()
                    else:
                        self._apply_command(line)
                else:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n退出")

    # ── 资源管理 ──

    def close(self):
        if self.esp:
            try:
                self.esp.close()          # 熄灭绿灯
            except Exception as e:
                print(f"ESP32 close 失败: {e}")
        self.arm.close()                  # 失能电机并关闭总线

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


def main():
    parser = argparse.ArgumentParser(description="机械臂主逻辑 — 启动 + 控制台控制 (增量 / 绝对笛卡尔)")
    parser.add_argument("--esp-port", default=None,
                        help="ESP32 串口 (默认自动检测)")
    args = parser.parse_args()

    with MainLogic(esp_port=args.esp_port) as logic:
        logic.run()


if __name__ == "__main__":
    main()
