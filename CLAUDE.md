# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

Jetson 上的四轴机械臂视觉拼装系统:摄像头实时 YOLO 分割识别碎片 → 拼接/爆炸图 → 机械臂按坐标抓取、搬运、旋转放置,完成装配。无独立测试框架,`reassemble.py <image>` 单图模式是最快的改动验证方式(不依赖硬件)。

## 运行命令

- 主程序(启动流程 + 控制台绝对笛卡尔控制):`/usr/bin/python3 main_logic.py [--esp-port /dev/ttyUSB0] [--no]`(`--no` = 调试模式跳过视觉)
- 实时分割窗口:`/usr/bin/python3 infer.py`(q 退出 / s 截图 / r 跑 read 流程 / t 平滑开关)
- 单图拼接+爆炸图测试:`/usr/bin/python3 reassemble.py <image_path>`(输出 reassembled.jpg / exploded.jpg)
- 辅助工具:camera_driver.py(相机查看)、host_test.py(ESP32 串口测试)、view_image.py(独立进程图片查看器,由 infer 弹窗调用)

⚠ **必须用 `/usr/bin/python3`**:该解释器带 GPU torch;miniforge 的 python3 只有 CPU torch。

## 硬件与通信(三路串口 + 摄像头)

| 设备 | 协议/端口 | 代码位置 |
|---|---|---|
| 机械臂 4 电机 (ID1 基座/ID2 肩/ID3 肘/ID4 腕) | GIM4310, RS485 115200 8N1 | `GIM4310_driver.py` (`MotorBus`, 地址 1-4) |
| 工具旋转舵机 (Feetech STS) | 半双工 TTL 1Mbps, ID=1 | `servo_driver.py` (正=逆时针) |
| ESP32 (继电器 + 红绿灯光) | 115200, 自动检测端口 | `arm_driver.py` `Esp32Cmd`, 指令 `mag_high/mag_low/red_on/green_on...` |
| 摄像头 | /dev/video0, GStreamer 优先 | `infer.build_gst_pipeline` |

⚠ 摄像头**不要两个进程同时打开**(实时窗口与 `read` 流程会互相抢占)。

## 架构分层

```
main_logic.py (业务/控制台) → arm_driver.CartesianArm (笛卡尔级) → arm_driver.Arm (电机级) → GIM4310_driver.MotorBus
```

- **main_logic.py**:启动流程 + 控制台指令(`x y z` 绝对坐标、`n u v` 相机像素、`action`/`put` 抓放、`trans`/`transport` 搬运、`r` 舵机、`read` 装配、`home`)与所有动作序列常量(`ACTION_*`/`TRANS_*`/`STANDBY_*`);注入基座旋转→舵机补偿回调。
- **arm_driver.py**:
  - 纯函数:FK/IK(`inverse_kinematics_plane`)、`cartesian_to_polar`、`tool_center`(执行器中心 = 腕部 + `TOOL_OFFSET_X/Y` 偏移)、相机标定仿射 `CAMERA_A`(`camera_to_arm`/`arm_to_camera`)。
  - `Arm`(电机级):关节限位 `JOINT_LIMITS`、上电编码器偏置检测、手腕水平参考(`ID4` 自动维持水平)。
  - `CartesianArm`(笛卡尔级):`move_tool_to`(执行器中心)、`move_camera_to`(像素→坐标)、`move_to_cartesian`(腕部)、`go_standby`/`go_home`、`wait_for_arrival`、分段移动与分关节延时。
- **infer.py**:YOLOv8 GPU 推理、`VertexSmoother` 跨帧 EMA 平滑、`run_read_pipeline`(read 全流程)、实时窗口。
- **reassemble.py**:`masks_from_yolo`、`reassemble`(DFS 回溯拼接:边长度匹配 + 面积比值 + 四边形角度校验)、爆炸图 `_exploded_layout`/`draw_exploded_view`、三合一图 `create_combined_view`。

## 关键约定(跨文件一致,修改需同步)

1. **坐标系**:FK 镜像约定 `x = -r·cos(θ1), y = r·sin(θ1)`;`move_to_cartesian` 用 r<0 约定;待机 (X=-9, Y=0),基座角 0°。基座角**增大 = 从上方(摄像头视角)看顺时针**。
2. **方向**:舵机**正 = 逆时针**;`read` 输出的旋转角约定为**顺时针为正、逆时针为负**,已在 `reassemble.py` 计算源头取反(可直接作 `trans` 角度用)。
3. **移动分段**:去程 `move_to_cartesian` 先臂平面(ID2/3/4)后基座(`base_after_plane=True`);回程 `go_standby` 先基座后平面(`base_first=True`);基座无需转动或读取失败时自动退化同步多轴。
4. **基座旋转→舵机反向补偿**:`CartesianArm.on_base_rotate` 回调由 `main_logic._compensate_base_rotation` 注入,补偿 = Δ × `SERVO_BASE_COMP_GAIN`(符号按实测调 ±1),保持工具绝对姿态并防线材缠绕。
5. **分关节延时**:`move_tool_to`/`move_to_cartesian` 的 `joint_delays={关节号: 秒}` — 无延时关节先动,到点后按序发指令;目标角与当前相差 ≤0.5° 自动跳过(如回升段电机3/4 分步 0.5s/1.0s)。
6. **read 全流程**(main_logic `read` 命令与 infer.py 按 R 完全共用 `run_read_pipeline`):连续采集 10 帧 → 碎片顶点跨帧时域平滑 → 拼接 → 爆炸图 → 独立进程三合一弹窗 → 输出 `#N(原始坐标)-(爆炸图坐标, 旋转角)`。
7. **舵机阻塞**:`move_relative_deg` 只发指令,需 `wait_for_arrival` 轮询到位(trans 旋转与 `r` 指令已接入)。
8. **爆炸图**:包围盒贴边算法 — 每个碎片沿"整体包围盒中心→碎片包围盒中心"方向推出,直到最外边缘贴到外扩后的目标包围盒(`expand` 默认 0.3,`edge_tolerance` 默认 5px);包围盒 = 对齐后碎片顶点的 [min,max] 闭区间(勿用 `cv2.boundingRect`,其含 +1 像素)。
9. **执行器中心与腕部**的换算关系、`move_tool_to` 的 y 镜像补偿在 `tool_to_wrist`/`tool_center` 中,改偏移参数(TOOL_OFFSET)时两者必须同步。

## 常见开发注意

- 动作参数都在文件顶部常量区,数值(待机 Z、下降高度、转速、延时)直接改常量即可,无需动逻辑。
- 拼接/爆炸图只改 `reassemble.py` 与 `infer.py`;机械臂运动只改 `arm_driver.py` 与 `main_logic.py`。
- 修改 `read` 输出角度约定时,`reassemble.py:413` 的取反、`infer.py` 与 `main_logic.py` 的 docstring 三处要一起改。
