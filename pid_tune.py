#!/usr/bin/python3
"""GIM4310 电机 PID 在线调参工具 (协议 0x14 读 / 0x15 在线改 / 0x16 保存).

用法:
    python3 pid_tune.py                          # 读取全部电机 (ID1-4) 当前 PID
    python3 pid_tune.py --id 2                   # 只读取 ID2
    python3 pid_tune.py --id 2 --pos-kp 60 --pos-ki 1      # 在线调整 (0x15, 掉电恢复)
    python3 pid_tune.py --id 2 --save --pos-kp 60          # 调整并保存 (0x16, 掉电保留)

参数说明 (对应协议 0x14 运动控制参数):
    --pos-kp 位置环 Kp     --pos-ki 位置环 Ki     --max-speed 位置模式最大速度 (rpm)
    --vel-kp 速度环 Kp     --vel-ki 速度环 Ki     --max-current 最大 Q 轴电流 (A)

调整建议:
    1. 位置环 Kp 决定到位刚度: 太小 → 到位慢/最终不到位; 太大 → 振荡/过冲/啸叫
    2. 位置环 Ki 消除稳态误差 (重力下垂/静摩擦): 先设 0 或很小, 有恒定偏差再加
    3. 速度环一般保持默认, 位置环调到极限仍振荡时才动速度环
    4. 每次只改一个参数, 小步 (±20~50%) 调整; 调整时确保电机静止、附近无障碍
    5. 用 main_logic.py 的 status / 坐标移动实测效果, 满意后再 --save 保存,
       否则掉电自动恢复旧参数

⚠ V3.03b3 固件实测: 电机失能时 0x15 (在线调整) 应答但不生效 (回读仍是旧值),
   故本工具默认走 0x16 保存; 如固件更新后 0x15 生效, 可加 --no-save 用回在线模式.
"""

import argparse

from GIM4310_driver import MotorBus, SERIAL_PORT, BAUDRATE


def fmt(p: dict) -> str:
    return (f"位置环 Kp={p['pos_kp']:.2f}  Ki={p['pos_ki']:.2f}  "
            f"最大速度={p['pos_max_speed_rpm']:.1f}rpm  |  "
            f"速度环 Kp={p['vel_kp']:.2f}  Ki={p['vel_ki']:.2f}  "
            f"最大电流={p['vel_max_current_a']:.3f}A")


def main():
    parser = argparse.ArgumentParser(description="GIM4310 电机 PID 在线调参")
    parser.add_argument("--port", default=SERIAL_PORT,
                        help=f"串口设备路径 (默认: {SERIAL_PORT})")
    parser.add_argument("--baud", type=int, default=BAUDRATE,
                        help=f"波特率 (默认: {BAUDRATE})")
    parser.add_argument("--id", type=int, default=None,
                        help="电机地址 (默认全部 1-4)")
    parser.add_argument("--pos-kp", type=float, default=None)
    parser.add_argument("--pos-ki", type=float, default=None)
    parser.add_argument("--max-speed", type=float, default=None,
                        help="位置模式最大速度 (rpm)")
    parser.add_argument("--vel-kp", type=float, default=None)
    parser.add_argument("--vel-ki", type=float, default=None)
    parser.add_argument("--max-current", type=float, default=None,
                        help="速度/位置模式最大 Q 轴电流 (A)")
    parser.add_argument("--no-save", action="store_true",
                        help="用 0x15 在线调整不保存 (⚠ V3.03b3 固件实测: "
                             "电机失能时 0x15 应答但不生效, 默认走 0x16 保存)")
    args = parser.parse_args()

    ids = [args.id] if args.id else [1, 2, 3, 4]
    write_any = any(v is not None for v in
                    (args.pos_kp, args.pos_ki, args.max_speed,
                     args.vel_kp, args.vel_ki, args.max_current))
    save = not args.no_save   # 0x15 实测无效, 默认 0x16 保存

    with MotorBus(port=args.port, addresses=ids, baudrate=args.baud) as bus:
        for m in bus.motors:
            if write_any:
                p = m.write_motion_params(
                    pos_kp=args.pos_kp, pos_ki=args.pos_ki,
                    pos_max_speed_rpm=args.max_speed,
                    vel_kp=args.vel_kp, vel_ki=args.vel_ki,
                    vel_max_current_a=args.max_current,
                    save=save)
                tag = "已保存 (0x16, 掉电保留)" if save \
                    else "在线生效 (0x15, 掉电恢复)"
                print(f"ID{m.address}: {tag}")
            else:
                p = m.read_motion_params()
                print(f"ID{m.address}: 当前参数  {fmt(p)}")


if __name__ == "__main__":
    main()
