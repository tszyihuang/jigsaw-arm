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
       trans <u> <v> [角度]  搬运: 抓取点 → 抓取 → 放置点 → [舵机旋转角度] → 放下
                       (角度可选, 放置点放下前旋转, 正=逆时针, 如: trans 123 85 )
       transport <u1> <v1> <u2> <v2> [角度]  同 trans, 但放置点显式指定
                       (V2 仍自动 -330, 如: transport 200 200 300 300 +60 → 放置点 300, -30)
       r <角度>           舵机相对转动, 正=逆时针 (如: r 50, r -30)
       read               跑一遍装配流程 (等价于 infer.py 按 R 键),
                          输出 #N(原始坐标)-(爆炸图坐标, 旋转角) 碎片数据
       t1 ~ t4            read 后把第 N 个碎片数据传入 transport 执行:
                          抓取点=原始坐标, 放置点=爆炸图坐标 (V2 仍自动 -330),
                          角度=旋转角
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
from servo_driver import (FeetechSTSServo, SERVO_PORT,
                          SERVO_STEP_PER_DEG, SERVO_SPEED)

# ── 待机位置 (实际笛卡尔坐标, mm) ───────────────────────────────────────────
STANDBY_X = -9.0
STANDBY_Y = 0.0
STANDBY_Z = 60.0

STANDBY_SPEED_RPM = 10.0   # 待机移动转速 (rpm)

# ── 基座旋转 → 舵机反向补偿 ────────────────────────────────────────────────
# 电机1 (基座) 旋转 Δ° 后, 舵机自动反向旋转补偿, 抵消基座转动对工具姿态的
# 影响 (保持抓取物方向不变). 方向推导: 基座角增大 = 从上方 (摄像头视角) 看
# 顺时针 (FK 镜像约定 x=-r·cosθ1, y=r·sinθ1, 相机 +x 右 / +y 上), 而舵机
# 正 = 逆时针, 故补偿量 = +Δ 即满足 "基座顺时针 Δ° → 舵机逆时针 Δ°".
# 若实测补偿方向相反, 将该值改为 -1.
SERVO_BASE_COMP_GAIN = -1.0

# ── action 动作序列参数 (执行器中心 Z, mm) ────────────────────────────────
ACTION_Z_DOWN = -39.0   # ① 下降到该高度
ACTION_Z_UP   =   -20.0   # ④ 动作结束后 Z 回升到该高度
ACTION_WAIT_DOWN = 1.0  # ② 到位后等待 (s)
ACTION_WAIT_MAG  = 0.5  # ③ 继电器吸合后等待 (s)
ACTION_SPEED_RPM = 5.0   # ⑤ 动作移动转速 (rpm, 临时限速: 缓慢下降测试用)
ACTION_ID3_DELAY = 0.6   # ④ 回升段: 电机3 (ID3) 延时旋转 (s), 在 ID2 之后动
ACTION_ID4_DELAY = 0.6   # ④ 回升段: 电机4 (ID4) 延时旋转 (s), 在 ID3 之后动

# ── trans 搬运序列参数 (相机像素坐标, px) ─────────────────────────────────
TRANS_V_OFFSET = -400.0  # 放置点 V = 抓取点 V - 400 
TRANS_WAIT     = 1.5     # 各步骤之间的等待时间 (s)
TRANS_STEP_NUMS = ("①", "②", "③", "④", "⑤", "⑥")   # 序列步骤圈号 (最多 6 步)


class MainLogic:
    """机械臂主逻辑 — 启动流程 + 控制台绝对笛卡尔控制 (执行器中心)."""

    def __init__(self, esp_port=None, no_vision=False):
        self.arm = CartesianArm()          # 机械臂 + 笛卡尔控制层 (含手腕水平参考)
        # 基座旋转 → 舵机反向补偿 (保持工具姿态; 舵机未连接时内部自动跳过)
        self.arm.on_base_rotate = self._compensate_base_rotation
        self.no_vision = no_vision         # 调试模式: 跳过一切视觉功能
        self._model = None                 # YOLO 模型缓存 (detect_target 与 read 共用)
        self._fragments = None             # 最近一次 read 的碎片数据 (t1~tN 指令用)
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

        # ── 舵机 (Feetech STS/SCS) ──
        self.servo = None
        try:
            self.servo = FeetechSTSServo(SERVO_PORT)
            if not (self.servo.is_open and self.servo.is_online()):
                print("⚠ 舵机未应答 — r 指令不可用")
                self.servo.close()
                self.servo = None
            else:
                pos, _ = self.servo.read_position()
                print(f"舵机已连接: {SERVO_PORT} "
                      f"(位置 {pos} ≈ {pos / SERVO_STEP_PER_DEG:.1f}°)")
        except Exception as e:
            print(f"⚠ 舵机连接失败: {e} — r 指令不可用")
            self.servo = None

    # ── 基座旋转补偿 ──

    def _compensate_base_rotation(self, delta_deg):
        """基座 (ID1) 旋转后的舵机反向补偿 (arm.on_base_rotate 回调).

        电机1 旋转 Δ° 时, 舵机反向旋转 Δ × SERVO_BASE_COMP_GAIN
        (舵机正 = 逆时针), 保持工具/碎片绝对姿态不变.
        Δ 的符号约定: 基座角增大 = 从上方看顺时针.
        舵机未连接时警告并跳过, 不影响机械臂移动.
        """
        if not self.servo:
            print(f"  ⚠ 基座旋转 {delta_deg:+.1f}°, 舵机未连接 — 跳过旋转补偿")
            return
        comp_deg = delta_deg * SERVO_BASE_COMP_GAIN
        print(f"  [补偿] 基座 {delta_deg:+.1f}° → 舵机反向 {comp_deg:+.1f}°")
        new_pos = self.servo.move_relative_deg(comp_deg,
                                               target_speed=SERVO_SPEED)
        if new_pos is None:
            print("  ⚠ 舵机补偿失败 (读取当前位置失败)")
        else:
            print(f"  ✓ 舵机补偿 → 位置 {new_pos} "
                  f"({new_pos / SERVO_STEP_PER_DEG:.1f}°)")

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
            model = self._get_model()
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

    def _get_model(self):
        """惰性加载并缓存 YOLO 模型 (detect_target 与 read 共用)."""
        if self._model is None:
            import infer
            self._model = infer.load_model()
        return self._model

    def _run_read(self):
        """read: 跑一遍完整装配流程 (等价于 infer.py 按 R 键) 并输出碎片数据.

        输出格式: #N(原始坐标)-(爆炸图坐标, 旋转角)
        原始坐标 = 摄像头画面中碎片多边形几何中心 (像素);
        爆炸图坐标 = 爆炸图画布中该碎片位置几何中心 (像素);
        旋转角 = 拼接对齐相对原始位姿的旋转角 (°, 顺时针为正、逆时针为负,
                 已在 reassemble 计算源头取反, 为舵机方向约定,
                 可直接用于 trans 角度).
        """
        try:
            import infer
        except Exception as e:
            print(f"⚠ 无法加载目标识别模块: {e}")
            return
        try:
            model = self._get_model()
        except Exception as e:
            print(f"⚠ 模型加载失败: {e}")
            return
        cap = infer.open_camera()
        if cap is None:
            print("⚠ 无法打开摄像头, 跳过 read")
            return
        try:
            data = infer.run_read_pipeline(model, cap)
        finally:
            cap.release()
        if data is None:
            return
        self._fragments = data
        print("── 碎片数据: #N(原始坐标)-(爆炸图坐标, 旋转角) ──")
        for d in data:
            ox, oy = d["orig"]
            ex, ey = d["exploded"]
            print(f"  #{d['idx'] + 1}({ox}, {oy})-({ex}, {ey}, {d['rot_deg']:+.1f}°)")
        print(f"  (输入 t1 ~ t{len(data)} 将对应碎片数据传入 transport 执行搬运)")

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
        print(f"           trans <u> <v> [角度] → 搬运: 抓取点 n <u> <v> 0 → 抓取 → 放置点 n <u> <v{TRANS_V_OFFSET:+.0f}> 0 → [舵机旋转角度] → 放下")
        print("                       角度可选 (放置点放下前旋转, 放下后反向回旋防缠绕, 正=逆时针), 例如: trans 300 200 +60")
        print(f"           transport <u1> <v1> <u2> <v2> [角度] → 同 trans, 但放置点显式指定 (V2 仍 {TRANS_V_OFFSET:+.0f})")
        print("                       例如: transport 200 200 300 300 +60 → 放置点 (300, -30)")
        print("           r <角度>            → 舵机相对转动, 正=逆时针 (如: r 50, r -30)")
        print("           read               → 跑一遍装配流程 (等价于 infer.py 按 R), 输出 #N(原始坐标)-(爆炸图坐标, 旋转角)")
        print("           t1 ~ t4            → 将 read 结果第 N 个碎片数据传入 transport 执行")
        print("                        (抓取点=原始坐标, 放置点=爆炸图坐标, V2 仍自动 -330, 角度=旋转角)")
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
              f"(转速 {ACTION_SPEED_RPM:.0f} rpm)")
        self.arm.move_tool_to(tx, ty, ACTION_Z_DOWN, speed_rpm=ACTION_SPEED_RPM)
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
              f"(转速 {ACTION_SPEED_RPM:.0f} rpm, "
              f"电机3/4 分步延时 {ACTION_ID3_DELAY:.1f}s/{ACTION_ID4_DELAY:.1f}s)")
        self.arm.move_tool_to(tx, ty, ACTION_Z_UP, speed_rpm=ACTION_SPEED_RPM,
                              joint_delays={3: ACTION_ID3_DELAY,
                                            4: ACTION_ID4_DELAY})
        print(f"  ✓ {'抓取' if mag_on else '放下'}完成")

    def _run_trans(self, u, v, angle_deg=None, u_place=None, v_place=None):
        """trans / transport 搬运序列: 移动到抓取点 → 抓取 → 移动到放置点 → [舵机旋转] → 放下.

        u/v 为抓取点相机像素坐标; 放置点缺省取 (u, v + TRANS_V_OFFSET)
        (trans 行为). transport 变体显式指定放置点 (u_place, v_place),
        其 V 仍应用 TRANS_V_OFFSET 偏移 (实际位置 = v_place + TRANS_V_OFFSET),
        例如 transport 200 200 300 300 +60 → 放置点 (300, -30).
        各步骤之间等待 TRANS_WAIT 秒.

        angle_deg 非 None 时, 在移动到放置点后、放下前让舵机相对转动
        该角度 (在放置点旋转碎片, 正=逆时针), 例如 trans 300 200 +60.
        放下 (put) 完成后, 舵机再反向回旋相同度数, 防止线材缠绕.
        舵机未连接时警告并跳过旋转/回旋, 搬运流程继续.
        """
        u_place = u if u_place is None else u_place
        v_place = (v if v_place is None else v_place) + TRANS_V_OFFSET
        print(f"  [trans] ① 移动到抓取点: n {u:.0f} {v:.0f} 0")
        self.arm.move_camera_to(u, v, 0.0)
        time.sleep(TRANS_WAIT)
        print("  [trans] ② 抓取 (action)")
        self._run_action(mag_on=True)
        time.sleep(TRANS_WAIT)
        step = 2
        step += 1
        print(f"  [trans] {TRANS_STEP_NUMS[step - 1]} 移动到放置点: "
              f"n {u_place:.0f} {v_place:.0f} 0  (V{TRANS_V_OFFSET:+.0f})")
        self.arm.move_camera_to(u_place, v_place, 0.0)
        time.sleep(TRANS_WAIT)
        if angle_deg is not None:
            step += 1
            if not self.servo:
                print(f"  ⚠ 舵机未连接, 跳过旋转 {angle_deg:+.0f}°")
            else:
                print(f"  [trans] {TRANS_STEP_NUMS[step - 1]} 舵机旋转 "
                      f"{angle_deg:+.0f}° (放置点, 放下前旋转碎片)")
                new_pos = self.servo.move_relative_deg(
                    angle_deg, target_speed=SERVO_SPEED)
                if new_pos is None:
                    print("  ⚠ 读取舵机当前位置失败, 未旋转")
                else:
                    print(f"  ✓ 舵机 → 新位置 {new_pos} "
                          f"({new_pos / SERVO_STEP_PER_DEG:.1f}°)")
                    # 阻塞等待舵机到位 (替代固定延时, 到位才继续放下)
                    if self.servo.wait_for_arrival(new_pos):
                        print("  ✓ 舵机到位")
                    else:
                        print("  ⚠ 舵机旋转等待超时 — 继续放下")
        step += 1
        print(f"  [trans] {TRANS_STEP_NUMS[step - 1]} 放下 (put)")
        self._run_action(mag_on=False)
        if angle_deg is not None:
            step += 1
            if not self.servo:
                print(f"  ⚠ 舵机未连接, 跳过反向回旋 {angle_deg:+.0f}°")
            else:
                # 放下后反向回旋相同度数, 防止线材缠绕
                print(f"  [trans] {TRANS_STEP_NUMS[step - 1]} 舵机反向回旋 "
                      f"{angle_deg:+.0f}° (防止线材缠绕)")
                new_pos = self.servo.move_relative_deg(-angle_deg,
                                                       target_speed=SERVO_SPEED)
                if new_pos is None:
                    print("  ⚠ 读取舵机当前位置失败, 未回旋")
                else:
                    print(f"  ✓ 舵机 → 新位置 {new_pos} "
                          f"({new_pos / SERVO_STEP_PER_DEG:.1f}°)")
                    if self.servo.wait_for_arrival(new_pos):
                        print("  ✓ 舵机回旋到位")
                    else:
                        print("  ⚠ 舵机回旋等待超时")
        print("  ✓ trans 搬运完成")

    def _run_trans_by_fragment(self, n):
        """tN: 将最近一次 read 结果的第 N 个碎片数据传入 transport 执行.

        抓取点 = 原始坐标 (碎片当前在摄像头画面中的位置);
        放置点 = 爆炸图坐标 (V2 仍自动 -330, 实际放置 (ex, ey-330));
        角度   = 拼接对齐旋转角 (可直接作舵机旋转角).
        """
        if not self._fragments:
            print(f"  ⚠ 请先运行 read 获取碎片数据, 再使用 t{n}")
            return
        if n < 1 or n > len(self._fragments):
            print(f"  ⚠ 只有 {len(self._fragments)} 个碎片 (#1 ~ "
                  f"#{len(self._fragments)}), 没有 #{n}")
            return
        d = self._fragments[n - 1]
        ox, oy = d["orig"]
        ex, ey = d["exploded"]
        rot = d["rot_deg"]
        print(f"  [t{n}] 碎片 #{d['idx'] + 1}: 抓取 ({ox:.0f}, {oy:.0f}) → "
              f"放置 ({ex:.0f}, {ey:.0f}) 旋转 {rot:+.1f}°")
        self._run_trans(ox, oy, rot, ex, ey)

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
        if parts[0] == "trans":
            try:
                vals = [float(p) for p in parts[1:]]
            except ValueError:
                print("  ⚠ 格式错误, 请输入: trans <u> <v> [角度]  例如: trans 123 85 或 trans 123 85 +60")
                return
            if len(vals) < 2:
                print("  ⚠ 至少需要 U 和 V: trans <u> <v> [角度]  例如: trans 123 85")
                return
            if len(vals) > 3:
                print("  ⚠ 参数过多, 最多 3 个: trans <u> <v> [角度]  例如: trans 123 85 +60")
                return
            angle = vals[2] if len(vals) > 2 else None
            self._run_trans(vals[0], vals[1], angle)
            return
        if parts[0] in ("transport", "transp"):
            try:
                vals = [float(p) for p in parts[1:]]
            except ValueError:
                print("  ⚠ 格式错误, 请输入: transport <u1> <v1> <u2> <v2> [角度]  例如: transport 200 200 300 300 +60")
                return
            if len(vals) < 4:
                print("  ⚠ 至少需要 4 个参数: transport <u1> <v1> <u2> <v2> [角度]  例如: transport 200 200 300 300 +60")
                return
            if len(vals) > 5:
                print("  ⚠ 参数过多, 最多 5 个: transport <u1> <v1> <u2> <v2> [角度]")
                return
            angle = vals[4] if len(vals) > 4 else None
            # 放置点 (u2, v2): V 仍自动 -330 → 实际移动到 (u2, v2-330)
            self._run_trans(vals[0], vals[1], angle, vals[2], vals[3])
            return
        if parts[0] == "read":
            self._run_read()
            return
        # tN (t1~t4): read 后把第 N 个碎片数据传入 transport 执行
        if (len(parts[0]) > 1 and parts[0][0] == "t"
                and parts[0][1:].isdigit()):
            self._run_trans_by_fragment(int(parts[0][1:]))
            return
        if parts[0] in ("r", "servo"):
            if not self.servo:
                print("  ⚠ 舵机未连接, r 指令不可用")
                return
            if len(parts) < 2:
                print("  ⚠ 请输入偏转角度, 例如: r 50 (逆时针 50°) 或 r -30")
                return
            try:
                delta_deg = float(parts[1])
            except ValueError:
                print("  ⚠ 角度格式错误, 例如: r 50 或 r -30")
                return
            new_pos = self.servo.move_relative_deg(delta_deg,
                                                   target_speed=SERVO_SPEED)
            if new_pos is None:
                print("  ⚠ 读取舵机当前位置失败, 未移动")
            else:
                print(f"  ✓ 舵机 {delta_deg:+.0f}° → "
                      f"新位置 {new_pos} ({new_pos / SERVO_STEP_PER_DEG:.1f}°)")
                # 阻塞等待舵机到位
                if self.servo.wait_for_arrival(new_pos):
                    print("  ✓ 舵机到位")
                else:
                    print("  ⚠ 舵机旋转等待超时")
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
        if self.servo:
            try:
                self.servo.close()        # 关闭时自动失能扭矩
            except Exception as e:
                print(f"舵机 close 失败: {e}")
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
