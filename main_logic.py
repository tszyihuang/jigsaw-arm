#!/usr/bin/python3
"""机械臂主逻辑 — 启动流程 + 控制台绝对笛卡尔控制 (执行器中心坐标)

启动后:
  1. 机械臂移动到待机位置 (实际笛卡尔坐标 X=-9, Y=0, Z=+80 mm)
  2. 向 ESP32 发送 red_on, 红灯亮起, 并保持待机
  3. 开启摄像头目标识别 (YOLO, 见 infer.py): 检测到目标后打印中心点
     像素坐标一次, 并将 ESP32 灯光切换为绿色常亮
  4. 控制台输入指令移动机械臂 (坐标对应执行器中心, 含 TOOL_OFFSET 偏移):
       <Xmm> <Ymm> [Zmm]   绝对笛卡尔坐标, Z 缺省保持当前
       n <u> <v> [Zmm]     摄像头像素坐标, 线性变换为执行器坐标后移动
       action             下降到 Z_DOWN → 继电器吸合 → Z 回升到 0 (抓取)
       put                下降到 Z_DOWN → 继电器释放 → Z 回升到 0 (放下)
       例: 90 0 → 执行器中心移动到 (X=90, Y=0)
       home / standby      回到待机位置 (X=-9, Y=0, Z=80)
       status              显示当前位置     ? / help → 本帮助

所有机械臂运动学/控制 (极坐标换算、IK、执行器偏移、目标状态跟踪) 均在
arm_driver.CartesianArm 中实现, 本文件只负责启动流程与指令解析.

用法:
    python3 main_logic.py                  # 自动检测 ESP32 串口
    python3 main_logic.py --esp-port /dev/ttyUSB0
    python3 main_logic.py --no             # 调试模式: 不运行任何视觉功能
"""

import argparse
import select
import sys
import time

from arm_driver import CartesianArm, Esp32Cmd, detect_esp32_port

# ── 待机位置 (实际笛卡尔坐标, mm) ───────────────────────────────────────────
STANDBY_X = -9.0
STANDBY_Y = 0.0
STANDBY_Z = 80.0

STANDBY_SPEED_RPM = 10.0   # 待机移动转速 (rpm)

# ── action 动作序列参数 (执行器中心 Z, mm) ────────────────────────────────
ACTION_Z_DOWN = -38.0   # ① 下降到该高度
ACTION_Z_UP   =   0.0   # ④ 动作结束后 Z 回升到该高度
ACTION_WAIT_DOWN = 1.0  # ② 到位后等待 (s)
ACTION_WAIT_MAG  = 0.5  # ③ 继电器吸合后等待 (s)
ACTION_SPEED_RPM = 3.0   # ⑤ 动作移动转速 (rpm, 临时限速: 缓慢下降测试用)
ACTION_ACCEL_RPM_S = 50.0   # 梯形曲线加速度 (rpm/s, 0x26 平滑加减速)
ACTION_DECEL_RPM_S = 50.0   # 梯形曲线减速度 (rpm/s)


class MainLogic:
    """机械臂主逻辑 — 启动流程 + 控制台绝对笛卡尔控制 (执行器中心)."""

    def __init__(self, esp_port=None, no_vision=False):
        self.arm = CartesianArm()          # 机械臂 + 笛卡尔控制层 (含手腕水平参考)
        self.no_vision = no_vision         # 调试模式: 跳过一切视觉功能
        esp_port = esp_port or detect_esp32_port()
        self.esp = None
        if esp_port:
            try:
                self.esp = Esp32Cmd(esp_port)
                print(f"ESP32 已连接: {esp_port}")
            except Exception as e:
                print(f"⚠ ESP32 连接失败: {e} — 红灯指令不可用")
        else:
            print("⚠ 未检测到 ESP32 — 红灯指令不可用 (可用 --esp-port 指定)")

    # ── 启动流程 ──

    def startup(self):
        """① 移动到待机位置  ② 红灯亮起."""
        print("── 第一步: 启动流程 ──")
        self.arm.go_standby(STANDBY_X, STANDBY_Y, STANDBY_Z,
                            speed_rpm=STANDBY_SPEED_RPM)
        if self.esp:
            self.esp.red_on()
        else:
            print("⚠ ESP32 未连接, 跳过 red_on")

    def detect_target(self):
        """目标识别: 开启摄像头等待检测, 打印中心点坐标一次, 灯光切换绿色常亮."""
        print("── 目标识别 ──")
        try:
            import infer
        except Exception as e:
            print(f"⚠ 无法加载目标识别模块: {e}")
            return
        try:
            model = infer.load_model()
            cap = infer.open_camera()
            if cap is None:
                print("⚠ 无法打开摄像头, 跳过目标识别")
                return
            try:
                res = infer.detect_first_target(model, cap)
            finally:
                cap.release()
            if res is None:
                print("⚠ 未检测到目标")
                return
            cx, cy, cls = res
            print(f"  ✓ 检测到目标 [{cls}], 中心点像素坐标 (u={cx}, v={cy})")
            if self.esp:
                self.esp.red_off()
                self.esp.green_on()
                print("  ✓ 灯光已切换为绿色常亮")
            else:
                print("⚠ ESP32 未连接, 跳过绿灯")
        except KeyboardInterrupt:
            print("⚠ 目标识别被中断, 跳过")
        except Exception as e:
            print(f"⚠ 目标识别失败: {e}")

    def run(self):
        """主逻辑入口 — 启动 → 目标识别 → 控制台控制循环."""
        self.startup()

        print(f"\n✓ 启动完成: 机械臂待机 (X={STANDBY_X:.0f}, Y={STANDBY_Y:.0f}, "
              f"Z={STANDBY_Z:.0f}), 红灯亮")
        if self.no_vision:
            print("⚠ 调试模式 (--no): 跳过目标识别, 视觉功能关闭")
        else:
            self.detect_target()

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

    # ── 控制台解析 ──

    def _print_usage(self):
        """打印控制台指令说明."""
        print("控制台指令: <Xmm> <Ymm> [Zmm]   (执行器中心绝对坐标, Z 缺省保持当前)")
        print("           n <u> <v> [Zmm]     (摄像头像素坐标, Z 缺省保持当前)")
        print("           home / standby     → 回到待机位置 (X=-9, Y=0, Z=80)")
        print(f"           action / act       → 下降到 Z={ACTION_Z_DOWN:.0f} → 继电器吸合 → Z 回升到 {ACTION_Z_UP:.0f} (抓取)")
        print(f"           put                → 下降到 Z={ACTION_Z_DOWN:.0f} → 继电器释放 → Z 回升到 {ACTION_Z_UP:.0f} (放下)")
        print("           status             → 显示当前位置     ? / help → 本帮助")
        print("  例: 90 0      → 移动到 (X=90,  Y=0)")
        print("      100 50 80 → 移动到 (X=100, Y=50, Z=80)")
        print("      n 123 85  → 相机像素 → 执行器坐标移动 (≈ X=90, Y=-8.4)")

    def _show_status(self):
        """显示当前实际关节角与执行器中心坐标."""
        joints, (x, y, z), (tx, ty, tz) = self.arm.status()
        print(f"  实际 ID1={joints[1]:+.1f}°  ID2={joints[2]:+.1f}°  "
              f"ID3={joints[3]:+.1f}°  ID4={joints[4]:+.1f}°")
        print(f"        腕部   X={x:+.1f}  Y={y:+.1f}  Z={z:+.1f} mm")
        print(f"        执行器 X={tx:+.1f}  Y={ty:+.1f}  Z={tz:+.1f} mm")

    def _run_action(self, mag_on=True):
        """抓取/放下序列: 当前位置 → 下降到 Z_DOWN → 等 1s → 继电器吸合/释放 → 等 0.5s → Z 回升到 0.

        mag_on=True  → 继电器吸合 (action 抓取); mag_on=False → 继电器释放 (put 放下).
        保持当前 X/Y 不变, 只改变执行器中心 Z 高度.
        """
        if not self.esp:
            print("  ⚠ ESP32 未连接, 无法控制继电器 — 动作中止")
            return
        *_, (tx, ty, tz) = self.arm.status()
        print(f"  [动作] 当前位置: 执行器 X={tx:+.1f}  Y={ty:+.1f}  Z={tz:+.1f} mm")
        print(f"  [动作] ① 下降到 Z={ACTION_Z_DOWN:.0f} mm "
              f"(转速 {ACTION_SPEED_RPM:.0f} rpm, 梯形曲线)")
        self.arm.move_tool_to(tx, ty, ACTION_Z_DOWN, speed_rpm=ACTION_SPEED_RPM,
                              trapezoid=True, max_accel_rpm_s=ACTION_ACCEL_RPM_S,
                              max_decel_rpm_s=ACTION_DECEL_RPM_S)
        print(f"  [动作] ② 等待 {ACTION_WAIT_DOWN:.0f}s")
        time.sleep(ACTION_WAIT_DOWN)
        name = "吸合" if mag_on else "释放"
        cmd = "mag_high" if mag_on else "mag_low"
        print(f"  [动作] ③ 继电器{name} ({cmd})")
        if mag_on:
            self.esp.relay_on()
        else:
            self.esp.relay_off()
        time.sleep(ACTION_WAIT_MAG)
        print(f"  [动作] ④ Z 回升到 {ACTION_Z_UP:.0f} mm "
              f"(转速 {ACTION_SPEED_RPM:.0f} rpm, 梯形曲线)")
        self.arm.move_tool_to(tx, ty, ACTION_Z_UP, speed_rpm=ACTION_SPEED_RPM,
                              trapezoid=True, max_accel_rpm_s=ACTION_ACCEL_RPM_S,
                              max_decel_rpm_s=ACTION_DECEL_RPM_S)
        print(f"  ✓ {'抓取' if mag_on else '放下'}完成")

    def _apply_command(self, line):
        """解析控制台指令: 绝对笛卡尔 <Xmm> <Ymm> [Zmm] → move_tool_to."""
        parts = line.split()
        if parts[0] in ("?", "h", "help"):
            self._print_usage()
            return
        if parts[0] in ("home", "standby"):
            self.arm.go_home(STANDBY_X, STANDBY_Y, STANDBY_Z,
                             speed_rpm=STANDBY_SPEED_RPM)
            return
        if parts[0] in ("action", "act"):
            self._run_action(mag_on=True)
            return
        if parts[0] in ("put", "place"):
            self._run_action(mag_on=False)
            return
        if parts[0] in ("n", "cam"):
            try:
                vals = [float(p) for p in parts[1:]]
            except ValueError:
                print("  ⚠ 格式错误, 请输入: n <u> <v> [Zmm]  例如: n 123 85")
                return
            if len(vals) < 2:
                print("  ⚠ 至少需要 U 和 V: n <u> <v>  例如: n 123 85")
                return
            if len(vals) > 3:
                print("  ⚠ 参数过多, 最多 3 个: n <u> <v> [Zmm]")
                return
            u, v = vals[0], vals[1]
            z = vals[2] if len(vals) > 2 else None
            self.arm.move_camera_to(u, v, z)
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
        self.arm.move_tool_to(x, y, z)

    # ── 资源管理 ──

    def close(self):
        if self.esp:
            try:
                self.esp.close()          # 熄灭红灯
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
    parser.add_argument("--no", "--no-vision", dest="no_vision",
                        action="store_true",
                        help="调试模式: 不运行任何视觉功能 (跳过摄像头/YOLO)")
    args = parser.parse_args()

    with MainLogic(esp_port=args.esp_port, no_vision=args.no_vision) as logic:
        logic.run()


if __name__ == "__main__":
    main()
