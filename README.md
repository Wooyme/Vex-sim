# G1 MuJoCo simulator

这是从 `holosoma.run_sim` 提取出的独立、精简版本，仅保留 MuJoCo 与 Unitree G1 29DoF。
运行时不导入或访问原 `holosoma` 目录；机器人 MJCF 和其引用的网格均已放在本项目内。

## 安装与运行

```bash
uv sync
uv run g1-run-sim
```

默认加载平面地形、使用站立 PD 控制，并像原 `run_sim` 一样启用虚拟弹性吊架。
可用 `--no-gantry` 关闭吊架。程序化混合地形与外部地形示例：

```bash
uv run g1-run-sim --terrain mix --terrain-seed 7
uv run g1-run-sim --terrain-file /path/to/terrain.obj
uv run g1-run-sim --terrain-file /path/to/terrain.stl --terrain-scale 0.01 \
  --terrain-position 1.0,0.0,-0.05
```

无窗口验证模型和地形（运行 100 步后退出）：

```bash
uv run g1-run-sim --headless --no-realtime --steps 100 --terrain mix
```

旧入口的组合参数也可继续使用：

```bash
uv run g1-run-sim simulator:mujoco robot:g1_29dof terrain:terrain_locomotion_mix
```

运行测试：

```bash
uv run pytest
```

## Gantry 键盘控制

MuJoCo 窗口中的键位与 holosoma `run_sim` 一致：

| 按键 | 功能 |
| --- | --- |
| `7` | 吊索长度减小 0.1 m（升高） |
| `8` | 吊索长度增加 0.1 m（降低） |
| `9` | 启用/停用 gantry |
| `0` | 按当前方向调整附加力 10 N（范围 -100～100 N） |
| `-` | 切换附加力调整方向 |
| `Y` | 切换相机跟踪 |
| `G` | 显示/隐藏帮助菜单 |
| `Backspace` | 重置仿真和 gantry 吊点 |

窗口会显示 gantry 状态、当前长度和附加力，并绘制红色吊点与蓝色吊索。

## MQTT 状态发布

同时提供 broker、主题和发布频率即可启用 MQTT。发布频率使用墙钟时间调度，单位为 Hz：

```bash
uv run g1-run-sim \
  --mqtt-broker mqtt://localhost:1883 \
  --mqtt-topic robot/g1/mujoco/state \
  --mqtt-frequency 50
```

broker 支持 `host[:port]`、`mqtt://user:password@host:port` 和启用 TLS 的
`mqtts://user:password@host:port`。每条消息为 JSON：

```json
{
  "timestamp": 1750000000.123,
  "simulation_time": 1.25,
  "joint_names": ["left_hip_pitch_joint"],
  "joint_angles": [-0.312],
  "base_xyz": [0.0, 0.0, 0.82],
  "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0]
}
```

实际消息中的 `joint_names` 和 `joint_angles` 均包含全部 29 个关节；四元数保持
MuJoCo 的 `[w, x, y, z]` 顺序。三个 MQTT 参数必须同时提供，且发布频率不能高于仿真 FPS。

## Unitree bridge

安装 bridge extra 后，可用 `--bridge` 开启 G1 LowState/LowCmd DDS 桥：

```bash
uv sync --extra bridge
uv run --extra bridge g1-run-sim --bridge --bridge-interface lo
```

此时 DDS 指令直接驱动 29 个电机；未开启 bridge 时可通过 `--control stand|zero`
选择站立 PD 或零力矩控制。
