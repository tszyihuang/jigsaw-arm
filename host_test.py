"""
ESP32 串口控制 - 主机端测试脚本 (适配 Linux / Jetson)

用法:
    python host_test.py                     # 自动检测 ESP32 端口（交互模式）
    python host_test.py /dev/ttyUSB0        # 指定端口
    python host_test.py --auto              # 自动测试，自动检测端口
    python host_test.py --auto /dev/ttyUSB0 # 指定端口自动测试
    python host_test.py --list              # 仅列出可用串口
"""

import os
import sys
import time
import serial
import serial.tools.list_ports

# 有效命令列表（按顺序：开→关 配对）
COMMANDS = [
    "mag_low", "mag_high",
    "red_on",  "red_off",
    "green_on","green_off",
]

# Linux 上串口通常以 /dev/tty 开头；Windows 则是 COMxx
DEFAULT_PORT = None   # None = 自动检测
BAUDRATE = 115200

# ESP32 常见 USB 芯片的 Vendor ID
ESP32_VENDOR_HINTS = {
    0x10C4,  # Silicon Labs CP210x
    0x1A86,  # QinHeng CH340/CH341
    0x0403,  # FTDI (部分 ESP32 开发板用 FTDI)
    0x303A,  # Espressif 原生 USB (ESP32-S2/S3/C3)
}

# 精确 PID 提示: FT232R (0403:6001). 仅按 VID=0x0403 会误选 FT4232H 四通道 (0403:6011)
ESP32_PID_HINTS = {
    0x6001,  # FT232R — 本机 ESP32 用的转接芯片
}

# ESP32 的 by-id 固定路径 (唯一序列号, 插拔/重启后不变), 最高优先级
ESP32_SERIAL_BY_ID = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0"


def list_ports():
    """列出所有可用串口（含详细信息）"""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("未检测到串口设备")
        return ports
    print("可用串口:")
    for p in ports:
        vid_pid = f"VID:0x{p.vid:04X} PID:0x{p.pid:04X}" if p.vid else "无 VID/PID"
        print(f"  {p.device} — {p.description}  [{vid_pid}]")
    return ports


def auto_detect_port():
    """
    自动检测 ESP32 所在串口。
    优先级: by-id 固定路径 → 精确 PID (FT232R) → 唯一串口.
    不再仅按 VID 匹配 — 否则多口 FTDI 会误选到电机总线等其它设备.
    """
    if os.path.exists(ESP32_SERIAL_BY_ID):
        print(f"按 by-id 固定路径检测到 ESP32: {ESP32_SERIAL_BY_ID}")
        return ESP32_SERIAL_BY_ID

    ports = serial.tools.list_ports.comports()
    if not ports:
        print("未检测到任何串口设备。请确认 ESP32 已连接。")
        sys.exit(1)

    # 精确 PID 匹配 (ESP32 的 FT232R), 避免误选 FT4232H 四通道
    for p in ports:
        if p.vid in ESP32_VENDOR_HINTS and p.pid in ESP32_PID_HINTS:
            print(f"自动检测到 ESP32 (PID 匹配): {p.device} — {p.description}")
            return p.device

    # 如果只有一个串口，直接用它
    if len(ports) == 1:
        p = ports[0]
        print(f"仅有一个串口，使用: {p.device} — {p.description}")
        return p.device

    # 多个串口但无法识别 — 让用户选
    print("检测到多个串口，无法自动判断哪个是 ESP32：")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} — {p.description}")
    print()
    while True:
        try:
            choice = input(f"请选择端口编号 (0-{len(ports)-1}): ").strip()
            idx = int(choice)
            if 0 <= idx < len(ports):
                return ports[idx].device
        except (ValueError, KeyboardInterrupt):
            pass
        print(f"输入无效，请输入 0-{len(ports)-1}")


def open_serial(port: str) -> serial.Serial:
    """打开串口并打印欢迎信息"""
    try:
        ser = serial.Serial(port, BAUDRATE, timeout=2)
        print(f"已连接 {port} @ {BAUDRATE} baud\n")
    except serial.SerialException as e:
        print(f"无法打开串口 {port}: {e}")
        list_ports()
        sys.exit(1)

    time.sleep(1.0)
    # 读取 ESP32 启动时可能输出的信息
    while ser.in_waiting:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line:
            print(f"  [ESP32] {line}")
    return ser


def send_command(ser: serial.Serial, cmd: str) -> str:
    """发送指令，返回 ESP32 回复"""
    ser.write((cmd + "\n").encode("utf-8"))
    reply = ser.readline().decode("utf-8", errors="replace").strip()
    return reply


# ==================== 交互模式 ====================
def interactive_mode(ser: serial.Serial):
    print(f"可用命令: {', '.join(COMMANDS)}")
    print("输入 'quit' 退出, 'auto' 自动测试\n")

    try:
        while True:
            user_input = input("> ").strip().lower()

            if user_input in ("quit", "exit", "q"):
                print("退出")
                break

            if not user_input:
                continue

            if user_input == "auto":
                serial.close()
                auto_test(ser.port)
                return

            if user_input in COMMANDS:
                print(f"  发送: {user_input}")
                reply = send_command(ser, user_input)
                if reply:
                    print(f"  ← {reply}")
            else:
                print(f"  未知命令: {user_input}")

    except KeyboardInterrupt:
        print("\n中断退出")
    finally:
        ser.close()
        print("串口已关闭")


# ==================== 自动测试模式 ====================
def auto_test(ser_or_port):
    """依次发送全部指令，每条约 2 秒间隔（含 LED 闪烁 1 秒）"""
    if isinstance(ser_or_port, serial.Serial):
        ser = ser_or_port
    else:
        ser = open_serial(ser_or_port)

    print("=" * 40)
    print("自动测试：依次发送全部指令")
    print("=" * 40)

    passed, failed = 0, 0

    try:
        for cmd in COMMANDS:
            print(f"\n>>> 发送: {cmd}")
            reply = send_command(ser, cmd)
            if reply:
                print(f"    ← {reply}")
                if "OK" in reply:
                    passed += 1
                else:
                    failed += 1
            else:
                print("    ← (无回复)")
                failed += 1
            time.sleep(1.5)  # 等 LED 闪烁完毕 + 留余量

    except KeyboardInterrupt:
        print("\n中断退出")
    finally:
        ser.close()

    print(f"\n{'=' * 40}")
    print(f"测试完成: 成功 {passed}, 失败 {failed}")
    print("串口已关闭")


# ==================== 入口 ====================
if __name__ == "__main__":
    mode = "interactive"
    port = DEFAULT_PORT

    for arg in sys.argv[1:]:
        if arg == "--auto":
            mode = "auto"
        elif arg == "--list":
            list_ports()
            sys.exit(0)
        elif arg.startswith("COM") or arg.startswith("/dev/") or arg.startswith("tty"):
            port = arg
        else:
            print(f"未知参数: {arg}")
            print(__doc__)
            sys.exit(1)

    # 未指定端口时自动检测
    if port is None:
        port = auto_detect_port()

    if mode == "auto":
        auto_test(port)
    else:
        ser = open_serial(port)
        interactive_mode(ser)
