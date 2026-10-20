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
  "joint_values": [-0.312],
  "base_xyz": [0.0, 0.0, 0.82],
  "base_quat_wxyz": [1.0, 0.0, 0.0, 0.0]
}
```

实际消息中的 `joint_names` 和 `joint_values` 均包含全部 29 个关节；四元数保持
MuJoCo 的 `[w, x, y, z]` 顺序。状态发布需要同时提供上述三个 MQTT 参数，且发布频率不能高于仿真 FPS。

## MQTT 外力订阅

使用 `--mqtt-force-topic` 订阅施力指令，可以单独启用，也可以与状态发布共用同一 broker 和连接：

```bash
uv run g1-run-sim \
  --mqtt-broker mqtt://localhost:1883 \
  --mqtt-force-topic robot/g1/mujoco/force
```

需要同时发布状态时，追加 `--mqtt-topic robot/g1/mujoco/state --mqtt-frequency 50`。
两个主题必须不同，均不支持通配符。订阅使用 QoS 0，连接及重连后自动订阅；订阅启动失败会报错退出。

消息为 JSON 对象，`forces` 数组中可包含一个或多个力。单力示例（需要安装 `mosquitto_pub`）：

```bash
mosquitto_pub -h localhost -t robot/g1/mujoco/force -m \
  '{"forces":[{"body_name":"torso_link","direction":[1,0,0],"magnitude":50,"duration":0.5}]}'
```

多力示例：

```bash
mosquitto_pub -h localhost -t robot/g1/mujoco/force -m \
  '{"forces":[{"body_name":"torso_link","direction":[1,0,0],"magnitude":50,"duration":0.5},{"body_name":"pelvis","direction":[0,1,0],"magnitude":20,"duration":1.0}]}'
```

| 字段 | 含义 |
| --- | --- |
| `body_name` | 模型中的 body 名称，例如 `torso_link`、`pelvis`，不允许 world body |
| `direction` | 世界坐标系中的三维非零方向向量，自动归一化，不随 body 姿态旋转 |
| `magnitude` | 力的大小，单位 N，须为非负有限数值 |
| `duration` | 持续的仿真秒数，须为正的有限数值 |

力作用于对应 body 的质心，不附加力矩。同一消息的力在仿真主循环消费该消息时同时开始，
每个力独立计时；同一 body 上来自同批或不同消息的力，以及虚拟吊架力，均按向量相加。
计时使用仿真时间，不使用现实时间，因此 `--no-realtime` 不改变施力的仿真时长。
持续时间以仿真步长为精度，向上取整到完整步，短于一步的正持续时间仍施加一步。

非法消息（包括空数组、错误字段、非有限数值、未知 body）整批丢弃并记录原因，已有力不受影响。
Backspace 重置会清除待执行及有效力。断线后已生效的力继续至到期；重连不会重放历史命令。
请使用非 retained 发布（不要添加 `mosquitto_pub -r`）；订阅端忽略带 retained 标志的历史消息。
QoS 0 不保证送达，每条实际收到的有效消息都作为一批新的力执行。

### 点控制器（points）

同一主题的消息还可以包含 `points` 数组（可与 `forces` 同时出现，整条消息一起校验、一起生效）。
每个 point 在其持续时间内每个仿真步运行一次从 holosoma 移植的 `PointController`：
高度 PD 加水平速度 PID，按测得的 body 质心位置和速度计算世界坐标系下的力，并作用于该 body 质心：

```bash
mosquitto_pub -h localhost -t robot/g1/mujoco/force -m \
  '{"points":[{"body_name":"pelvis","height":0.8,"velocity":[0.3,0],"duration":2.0}]}'
```

| 字段 | 含义 |
| --- | --- |
| `body_name` | 同 `forces`，不允许 world body |
| `height` | 质心目标高度（世界坐标 z），单位 m |
| `velocity` | 质心目标水平速度 `[vx, vy]`（世界坐标），单位 m/s |
| `duration` | 持续的仿真秒数，须为正的有限数值 |

控制器在开始时用测得的高度和速度初始化参考值，再以限速（高度 0.2 m/s，速度 2 m/s²）趋近目标；
不做重力补偿，竖直力上限 300 N，水平力上限 100 N（默认增益见 `g1_mujoco_sim/points.py` 的
`PointControllerCfg`）。对同一 body 已在生效的 point 再次下发指令时，只更新目标和到期时间、
保留控制器状态，因此可以连续流式发送指令；point 力与 `forces`、吊架力按向量相加。

## Unitree bridge

安装 bridge extra 后，可用 `--bridge` 开启 G1 LowState/LowCmd DDS 桥：

```bash
uv sync --extra bridge
uv run --extra bridge g1-run-sim --bridge --bridge-interface lo
```

此时 DDS 指令直接驱动 29 个电机；未开启 bridge 时可通过 `--control stand|zero`
选择站立 PD 或零力矩控制。
