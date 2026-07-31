import serial
import time

# ── 配置 ─────────────────────────────────────────────────────────────────────
SERVO_PORT = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0"  # 稳定 by-id 路径, 防 USB 枚举顺序变化
SERVO_STEP_PER_DEG = 4095 / 360   # 每度对应的步数
SERVO_SPEED = 1000                 # 常用转动速度

# ================= Feetech STS/SCS 串口舵机驱动 (ID=1) =================
class FeetechSTSServo:
    """Feetech STS/SCS 系列串口舵机底层驱动（半双工 TTL, 1Mbps, ID=1）"""

    SERVO_ID = 1

    ERROR_CODES = {
        0:   "正常",
        1:   "电压错误",
        2:   "角度错误",
        4:   "过热",
        8:   "超出范围",
        16:  "校验和错误",
        32:  "过载",
        64:  "指令错误",
        128: "格式错误",
    }

    def __init__(self, port, baudrate=1000000, timeout=0.1, debug=False):
        self.debug = debug
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            print(f"[INFO] 串口 {port} 打开成功, 波特率 {baudrate}")
        except Exception as e:
            print(f"[ERROR] 打开串口失败: {e}")
            self.ser = None

    @property
    def is_open(self):
        return self.ser is not None and self.ser.is_open

    @staticmethod
    def decode_error(code):
        if code == 0:
            return "正常"
        parts = []
        for bit, desc in FeetechSTSServo.ERROR_CODES.items():
            if bit > 0 and (code & bit):
                parts.append(desc)
        return " + ".join(parts) if parts else f"未知({code})"

    # ---------- 写指令 ----------

    def move_to(self, target_position, target_speed=2000, target_time=0):
        """位置控制, 0-4095 对应 0-360°"""
        if not self.is_open:
            return

        target_position = max(0, min(4095, int(target_position)))
        target_speed    = max(0, min(3400, int(target_speed)))
        target_time     = max(0, int(target_time))

        pos_l  = target_position & 0xFF
        pos_h  = (target_position >> 8) & 0xFF
        time_l = target_time & 0xFF
        time_h = (target_time >> 8) & 0xFF
        speed_l = target_speed & 0xFF
        speed_h = (target_speed >> 8) & 0xFF

        length      = 0x09
        instruction = 0x03  # WRITE
        address     = 0x2A  # 目标位置寄存器

        checksum_sum = (self.SERVO_ID + length + instruction + address +
                        pos_l + pos_h + time_l + time_h + speed_l + speed_h)
        checksum = (~checksum_sum) & 0xFF

        packet = [0xFF, 0xFF, self.SERVO_ID, length, instruction, address,
                  pos_l, pos_h, time_l, time_h, speed_l, speed_h, checksum]

        if self.debug:
            print(f"[DEBUG] WRITE pos:{target_position} speed:{target_speed} "
                  f"packet: {[hex(b) for b in packet]}")

        self.ser.write(bytearray(packet))

    def move_relative_deg(self, delta_deg, target_speed=2000):
        """相对当前角度偏转 (度), 正=逆时针, 负=顺时针 — 与当前位置无关.

        读取当前角度, 叠加偏转量后取模 4096 (可连续多圈), 再发绝对位置指令.
        返回新位置 (0-4095), 读取失败时返回 None.
        """
        pos, _ = self.read_position()
        if pos is None:
            return None
        new_pos = (pos - int(delta_deg * SERVO_STEP_PER_DEG)) % 4096
        self.move_to(new_pos, target_speed=target_speed)
        return new_pos

    def wait_for_arrival(self, target_position, tolerance=6, timeout=8.0):
        """阻塞等待舵机到达目标位置 (0-4095), 轮询读取位置直到到位.

        move_relative_deg 只是发送指令不等待, 本方法用于把旋转变成阻塞式:
        Args:
            target_position: 目标位置 (move_relative_deg 的返回值)
            tolerance: 位置容差 (步, 4095 步 = 360°, 6 步 ≈ 0.53°)
            timeout: 超时 (s), 超时返回 False (不抛异常, 由调用方决定后续)
        Returns:
            True 已到位; False 超时 / 持续读取失败.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pos, _ = self.read_position()
            if pos is not None:
                delta = (pos - target_position + 2048) % 4096 - 2048
                if abs(delta) <= tolerance:
                    return True
            time.sleep(0.02)
        return False

    def set_torque_limit(self, torque_value):
        """扭矩限制, 0-1000"""
        if not self.is_open:
            return

        torque_value = max(0, min(1000, int(torque_value)))
        torque_l = torque_value & 0xFF
        torque_h = (torque_value >> 8) & 0xFF

        length      = 0x05
        instruction = 0x03
        address     = 0x30

        checksum_sum = (self.SERVO_ID + length + instruction + address + torque_l + torque_h)
        checksum = (~checksum_sum) & 0xFF

        packet = [0xFF, 0xFF, self.SERVO_ID, length, instruction, address, torque_l, torque_h, checksum]
        self.ser.write(bytearray(packet))
        time.sleep(0.005)

    def disable_torque(self):
        """关闭扭矩 (失能) — 写扭矩开关寄存器 (0x31) = 0.

        舵机失去保持力, 可自由转动. 程序退出前应调用, 防止舵机持续通电保持.
        """
        if not self.is_open:
            return

        length      = 0x04
        instruction = 0x03  # WRITE
        address     = 0x31  # 扭矩开关
        data        = 0x00  # 0 = 关闭扭矩

        checksum_sum = (self.SERVO_ID + length + instruction + address + data)
        checksum = (~checksum_sum) & 0xFF

        packet = [0xFF, 0xFF, self.SERVO_ID, length, instruction, address, data, checksum]
        self.ser.write(bytearray(packet))
        time.sleep(0.005)

    # ---------- 读指令 ----------

    def read_position(self):
        """读取当前位置, 返回 (position, error_code)"""
        if not self.is_open:
            return None, None

        self.ser.reset_input_buffer()

        length      = 0x04
        instruction = 0x02  # READ
        address     = 0x38  # 当前位置寄存器
        data_len    = 0x02

        checksum_sum = (self.SERVO_ID + length + instruction + address + data_len)
        checksum = (~checksum_sum) & 0xFF

        packet = [0xFF, 0xFF, self.SERVO_ID, length, instruction, address, data_len, checksum]

        if self.debug:
            print(f"[DEBUG] READ REQ packet: {[hex(b) for b in packet]}")

        self.ser.write(bytearray(packet))
        response = self.ser.read(8)

        if self.debug:
            print(f"[DEBUG] READ RESP <- {len(response)} bytes: "
                  f"{[hex(b) for b in response] if response else 'EMPTY'}")

        if len(response) == 8 and response[0] == 0xFF and response[1] == 0xFF:
            error_code = response[4]
            pos_l = response[5]
            pos_h = response[6]
            position = pos_l | (pos_h << 8)
            if position > 32767:
                position -= 65536
            return position, error_code

        return None, None

    def read_register(self, address, data_len=2):
        """通用寄存器读取"""
        if not self.is_open:
            return None, None

        self.ser.reset_input_buffer()

        length      = 0x04
        instruction = 0x02
        data_len    = max(1, min(8, int(data_len)))

        checksum_sum = (self.SERVO_ID + length + instruction + address + data_len)
        checksum = (~checksum_sum) & 0xFF

        packet = [0xFF, 0xFF, self.SERVO_ID, length, instruction, address, data_len, checksum]
        self.ser.write(bytearray(packet))

        resp_len = 6 + data_len
        response = self.ser.read(resp_len)

        if len(response) >= 8 and response[0] == 0xFF and response[1] == 0xFF:
            error_code = response[4]
            data = list(response[5:5 + data_len])
            return data, error_code
        return None, None

    # ---------- 状态检查 ----------

    def is_online(self):
        """检测舵机是否在线"""
        pos, _ = self.read_position()
        return pos is not None

    def status(self):
        """打印舵机状态"""
        pos, err = self.read_position()
        if pos is not None:
            err_str = self.decode_error(err)
            print(f"[STATUS] 位置={pos} ({pos*360/4095:.1f}°), 状态={err_str}")
            return pos, err
        else:
            print("[STATUS] 舵机无应答")
            return None, None

    def close(self):
        """关闭串口前先关闭舵机扭矩 (失能), 保证程序退出后舵机自由不发热."""
        if self.ser and self.ser.is_open:
            try:
                self.disable_torque()
            except Exception:
                pass  # 串口异常时忽略, 仍尝试关闭串口
            self.ser.close()
            print("[INFO] 串口已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# ================= 调用示例 =================
if __name__ == "__main__":
    servo = FeetechSTSServo(port=SERVO_PORT, debug=False)

    if not servo.is_open:
        print("=== 串口连接失败 ===")
        exit(1)

    if not servo.is_online():
        print("[ERROR] 舵机 ID=1 无应答")
        servo.close()
        exit(1)

    # 读取当前位置
    pos, _ = servo.read_position()
    if pos is None:
        print("[ERROR] 无法读取当前位置")
        servo.close()
        exit(1)

    print(f"舵机在线, 当前位置 = {pos} ({pos/SERVO_STEP_PER_DEG:.1f}°)")
    print("输入相对偏转角度: +50 → 逆时针 50°, -30 → 顺时针 30° "
          "(与当前位置无关), q 退出\n")

    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                print()
                break
            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                break
            try:
                delta_deg = float(line)
            except ValueError:
                print("  ⚠ 请输入数字角度, 例如 +50 或 -30")
                continue

            new_pos = servo.move_relative_deg(delta_deg, target_speed=500)
            if new_pos is None:
                print("  ⚠ 读取当前位置失败, 未移动")
            else:
                print(f"  ✓ {delta_deg:+.0f}° → 新位置 {new_pos} "
                      f"({new_pos/SERVO_STEP_PER_DEG:.1f}°)")
    except KeyboardInterrupt:
        print()
    finally:
        servo.close()   # close 时自动关闭扭矩 (失能)
