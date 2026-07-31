#!/usr/bin/env python3
"""机械臂主逻辑 — 第一步: 启动流程 + 控制台控制 (执行器绝对坐标)

启动后:
  1. 机械臂移动到待机位置 (实际笛卡尔坐标 X=-9, Y=0, Z=+80 mm)
  2. 向 ESP32 发送 red_on, 红灯亮起, 并保持待机
  3. 控制台输入指令移动机械臂 (坐标对应执行器中心, 含 TOOL_OFFSET 偏移):
       <Xmm> <Ymm> [Zmm]   绝对笛卡尔坐标, Z 缺省保持当前
       例: 90 0 → 执行器中心移动到 (X=90, Y=0)
       home / standby      回到待机位置 (X=-9, Y=0, Z=80)

极坐标换算 (由 arm_driver.forward_kinematics 的 FK 反解):
    FK:  x = -r·cos(θ1),  y = r·sin(θ1)
    反解: r = √(x²+y²),   θ1 = atan2(y, -x)
    待机: r = √((-9)²+0²) = 9.0 mm,  θ1 = atan2(0, 9) = 0°
    Z 不变 (80 mm), 臂平面内用 inverse_kinematics_plane(r, z) 解 ID2/ID3,
    ID4 由手腕水平参考自动维持.
    move_to_cartesian 采用 r<0 约定: r = -√(x²+y²), θ1 = atan2(y, x),
    使正 X 半平面 θ1 ∈ [-90°, 90°], 避开基座 ±90° 限位.
    move_tool_to 求解执行器中心 (腕部 + TOOL_OFFSET_X/Y 偏移, 见 tool_center):
    基座角由横向约束 x·sinθ1 + y·cosθ1 = TOOL_OFFSET_Y 解析确定,
    臂平面内腕部目标 = 工具目标 - TOOL_OFFSET_X·(cosREF, sinREF),
    REF = 手腕水平参考角 (ID4 维持水平 → 工具俯仰角恒定), 再迭代精修;
    tool_to_wrist 转换后喂入 move_to_cartesian (腕部) 移动.

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
    TOOL_OFFSET_X,
    TOOL_OFFSET_Y,
    detect_esp32_port,
    forward_kinematics,
    inverse_kinematics_plane,
    tool_center,
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
    """机械臂主逻辑 — 启动流程 + 控制台绝对笛卡尔控制."""

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

    def go_home(self, wait=True, speed_rpm=STANDBY_SPEED_RPM):
        """回到待机位置 (X=-9, Y=0, Z=80) 并更新目标状态.

        Returns:
            target: 目标关节角 {1..4: deg}
        """
        target = self.go_standby(speed_rpm=speed_rpm)
        base_deg, r = cartesian_to_polar(STANDBY_X, STANDBY_Y)
        self._target = {"base": base_deg, "r": r, "z": STANDBY_Z}
        if wait:
            if self.wait_for_arrival(target, timeout=15.0):
                print("  ✓ 已到达待机位置")
            else:
                print("  ⚠ 等待超时未到达")
        return target

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

        内部转换为极坐标 (基座角 θ1, 前伸 r) 后 IK 求解移动 (r<0 约定):
            r = -√(x²+y²),  θ1 = atan2(y, x)

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

    def tool_to_wrist(self, x, y, z=None):
        """执行器中心坐标 (x, y, z) → 腕部坐标 (真实几何腕部).

        解析求解, 偏移定义参考 arm_driver.tool_center:
          ① 基座角: 工具横向分量恒等于 TOOL_OFFSET_Y (与 ID4/臂型无关)
             → x·sinθ1 + y·cosθ1 = TOOL_OFFSET_Y, 两分支取基座限位内者
          ② 臂平面内: 腕部目标 = 工具目标 - TOOL_OFFSET_X·(cosREF, sinREF),
             REF = 手腕水平参考角 (ID4 维持水平 → 工具俯仰角恒定)
          ③ 用 arm_driver.tool_center 校验残差并迭代精修 (兜底 ID4 限位夹紧)

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

    def move_tool_to(self, x, y, z=None, wait=True,
                     speed_rpm=STANDBY_SPEED_RPM):
        """移动执行器中心到绝对笛卡尔坐标 (X, Y, Z) — 控制台入口.

        执行器中心 = 腕部 + 局部偏移 (TOOL_OFFSET_X=12, TOOL_OFFSET_Y=-36,
        见 arm_driver.tool_center). 流程: tool_to_wrist 转换 → move_to_cartesian.

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
        print("控制台指令: <Xmm> <Ymm> [Zmm]   (执行器中心绝对坐标, Z 缺省保持当前)")
        print("           home               → 回到待机位置 (X=-9, Y=0, Z=80)")
        print("  例: 90 0      → 移动到 (X=90,  Y=0)")
        print("      100 50 80 → 移动到 (X=100, Y=50, Z=80)")
        print("      status    → 显示当前位置     ? / help → 本帮助")

    def _show_status(self):
        """显示当前实际关节角与执行器中心坐标."""
        joints = self.arm.get_joints()
        x, y, z = forward_kinematics(joints)
        tx, ty, tz = tool_center(joints)
        print(f"  实际 ID1={joints[1]:+.1f}°  ID2={joints[2]:+.1f}°  "
              f"ID3={joints[3]:+.1f}°  ID4={joints[4]:+.1f}°")
        print(f"        腕部   X={x:+.1f}  Y={y:+.1f}  Z={z:+.1f} mm")
        print(f"        执行器 X={tx:+.1f}  Y={ty:+.1f}  Z={tz:+.1f} mm")

    def _apply_command(self, line):
        """解析控制台指令: 绝对笛卡尔 <Xmm> <Ymm> [Zmm] → move_to_cartesian."""
        parts = line.split()
        if parts[0] in ("?", "h", "help"):
            self._print_usage()
            return
        if parts[0] in ("home", "standby"):
            self.go_home()
            return
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            print("  ⚠ 格式错误, 请输入: <Xmm> <Ymm> [Zmm]  例如: 90 0")
            return
        if len(vals) < 2:
            print("  ⚠ 至少需要 X 和 Y: <Xmm> <Ymm> [Zmm]  例如: 90 0")
            return
        if len(vals) > 3:
            print("  ⚠ 参数过多, 最多 3 个: <Xmm> <Ymm> [Zmm]")
            return
        x, y = vals[0], vals[1]
        z = vals[2] if len(vals) > 2 else None
        self.move_tool_to(x, y, z)

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
    parser = argparse.ArgumentParser(description="机械臂主逻辑 — 启动 + 控制台绝对笛卡尔控制")
    parser.add_argument("--esp-port", default=None,
                        help="ESP32 串口 (默认自动检测)")
    args = parser.parse_args()

    with MainLogic(esp_port=args.esp_port) as logic:
        logic.run()


if __name__ == "__main__":
    main()
