#!/usr/bin/env python3
"""机械臂主逻辑 — 第一步: 启动流程

启动后:
  1. 机械臂移动到待机位置 (实际笛卡尔坐标 X=-9, Y=0, Z=+80 mm)
  2. 向 ESP32 发送 red_on, 红灯亮起, 并保持待机

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
import time

import serial

from arm_driver import (
    Arm,
    detect_esp32_port,
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
    """机械臂主逻辑 — 当前实现第一步 (启动流程), 后续步骤在此扩展."""

    def __init__(self, esp_port=None):
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

    # ── 第一步: 启动流程 ──

    def startup(self):
        """① 移动到待机位置  ② 红灯亮起."""
        print("── 第一步: 启动流程 ──")
        self.go_standby()
        if self.esp:
            self.esp.red_on()
        else:
            print("⚠ ESP32 未连接, 跳过 red_on")

    def run(self):
        """主逻辑入口 — 启动后保持待机, 后续步骤在此扩展."""
        self.startup()
        print(f"\n✓ 启动完成: 机械臂待机 (X={STANDBY_X:.0f}, Y={STANDBY_Y:.0f}, "
              f"Z={STANDBY_Z:.0f}), 红灯亮")
        print("按 Ctrl+C 退出")
        try:
            while True:
                time.sleep(0.2)
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
    parser = argparse.ArgumentParser(description="机械臂主逻辑 — 第一步: 启动流程")
    parser.add_argument("--esp-port", default=None,
                        help="ESP32 串口 (默认自动检测)")
    args = parser.parse_args()

    with MainLogic(esp_port=args.esp_port) as logic:
        logic.run()


if __name__ == "__main__":
    main()
