# DQN Flappy Bird (PyTorch)

Cài đặt thuật toán **Deep Q-Network (DQN)** — kèm hai phần mở rộng tuỳ chọn **Double DQN** và **Dueling DQN** — để huấn luyện một agent chơi game **Flappy Bird** thông qua môi trường [`flappy_bird_gymnasium`](chuẩn Gymnasium/OpenAI Gym).
Toàn bộ mạng neural và vòng lặp huấn luyện được viết bằng PyTorch từ đầu (không dùng thư viện RL có sẵn như Stable-Baselines3).

## 1. Cấu trúc project
```
dqn_pytorch/
├── agent.py                    # Vòng lặp train/test chính, class Agent
├── dqn.py                      # Kiến trúc mạng neural (DQN / Dueling DQN)
├── experience_replay.py        # Replay Memory (buffer kinh nghiệm)
├── hyperparameters.yml         # Các bộ tham số huấn luyện (theo từng "run")
├── dqn_flappybird_kaggle.ipynb # Notebook để chạy training trên Kaggle
├── .vscode/launch.json         # Cấu hình debug/run sẵn cho VS Code
└── runs/                       # Output: log, biểu đồ, checkpoint model
    ├── flappybird1.log/.png/.pt
    └── flappybird3.log/.png/.pt
```

## 2. Cài đặt môi trường

Yêu cầu: **Python 3.10** (venv đi kèm project được tạo với 3.10.11).

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirement.txt 
```

## 3. Cách sử dụng

Script `agent.py` nhận vào **tên bộ hyperparameter** (khớp với một key trong `hyperparameters.yml`) và cờ `--train`.

**Huấn luyện (training)** — chạy vô hạn vòng lặp episode cho tới khi bấm dừng (Ctrl+C):

```bash
python agent.py flappybird3 --train
```

**Kiểm thử / xem agent chơi (đã huấn luyện xong)** — load model đã lưu và render màn hình:
```bash
python agent.py flappybird3
```

Kết quả (log huấn luyện, biểu đồ reward/epsilon, checkpoint `.pt`) được ghi tự động vào `runs/<tên-set>.{log,png,pt}`.

## 4. Các bộ hyperparameter có sẵn (`hyperparameters.yml`)

> ⚠️ **Lưu ý quan trọng**: bản `hyperparameters.yml` hiện tại trên đĩa đã bị chỉnh sửa sau khi `runs/flappybird1.log/.pt` được tạo ra. Cấu hình **gốc** thực sự dùng để train ra kết quả `flappybird1` (best reward 106.4)

| Tham số | flappybird1 (cấu hình gốc, đã dùng để train) | flappybird3 |
|---|---|---|
| `env_id` | FlappyBird-v0 | FlappyBird-v0 |
| `replay_memory_size` | 100,000 | 200,000 |
| `mini_batch_size` | 32 | 64 |
| `epsilon_init` → `epsilon_min` | 1 → 0.05 | 1 → 0.01 |
| `epsilon_decay` | 0.99995 | 0.99995 |
| `network_sync_rate` | 10 bước | 100 bước |
| `learning_rate_a` | 0.0001 | 0.00005 |
| `discount_factor_g` (γ) | 0.99 | 0.995 |
| `fc1_nodes` | 512 | 512 |
| `enable_double_dqn` | **True** | False |
| `enable_dueling_dqn` | **True** | True |
| `env_make_params.use_lidar` | False | False |
| Loss function (tại thời điểm train) | `MSELoss` | `SmoothL1Loss` |

## 5. Thuật toán & kiến trúc mạng

- **Input**: vector trạng thái 12 chiều từ `FlappyBird-v0` (`use_lidar=False`) — vị trí/vận tốc ống và chim.
- **Output**: Q-value cho 2 hành động — `0 = không làm gì`, `1 = vỗ cánh (flap)`.
- **Mạng**: `state_dim → Linear(fc1_nodes) → ReLU`, sau đó rẽ nhánh:
  - Nếu **Dueling DQN**: nhánh Value `V(s)` và nhánh Advantage `A(s,a)`, kết hợp bằng `Q = V + A − mean(A)`.
  - Nếu không: một `Linear` output thẳng ra Q-values.
- **Loss**: `SmoothL1Loss` (Huber loss) giữa Q hiện tại và Q mục tiêu (Bellman target), có hỗ trợ **Double DQN** để giảm overestimation.
- **Optimizer**: Adam.
## 6. Kết quả huấn luyện đã ghi nhận
┌─────────────┬────────┬─────────┬─────────────┬───────┬───────────────┬─────────────┬──────────┬───────────┬────────────┐
│     Run     │ Double │ Dueling │ Replay Size │ Batch │ Learning Rate │ Best Reward │ Episodes │ Thời gian │ Reward/giờ │
├─────────────┼────────┼─────────┼─────────────┼───────┼───────────────┼─────────────┼──────────┼───────────┼────────────┤
│ flappybird1 │ ✓      │ ✗       │ 100,000     │ 32    │ 0.00010       │ 106.4       │ 941,863  │ 23.09h    │ 4.6        │
├─────────────┼────────┼─────────┼─────────────┼───────┼───────────────┼─────────────┼──────────┼───────────┼────────────┤
│ flappybird2 │ ✓      │ ✗       │ 200,000     │ 64    │ 0.00005       │ 278.9       │ 403,527  │ 5.25h     │ 53.2       │
├─────────────┼────────┼─────────┼─────────────┼───────┼───────────────┼─────────────┼──────────┼───────────┼────────────┤
│ flappybird3 │ ✗      │ ✓       │ 200,000     │ 64    │ 0.00005       │ 272.8       │ 548,802  │ 7.33h     │ 37.2       │
├─────────────┼────────┼─────────┼─────────────┼───────┼───────────────┼─────────────┼──────────┼───────────┼────────────┤
│ flappybird4 │ ✓      │ ✓       │ 200,000     │ 64    │ 0.00005       │ 204.4       │ 377,378  │ 6.17h     │ 33.1       │
└─────────────┴────────┴─────────┴─────────────┴───────┴───────────────┴─────────────┴──────────┴───────────┴────────────┘

Xem biểu đồ tại log_comparison.png
