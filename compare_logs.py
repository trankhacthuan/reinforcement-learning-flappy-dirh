"""So sanh cac lan train FlappyBird (flappybird1..4) tu file log trong runs/.

Chay: python compare_logs.py
Ket qua: bang so sanh in ra console + bieu do luu tai runs/log_comparison.png
"""
import os
import re
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

RUNS_DIR = "runs"
HYPERPARAMS_FILE = "hyperparameters.yml"
OUT_IMG = os.path.join(RUNS_DIR, "log_comparison.png")

LOG_LINE_RE = re.compile(
    r"^(?P<ts>\d{2}-\d{2} \d{2}:\d{2}:\d{2}): New best reward (?P<reward>-?\d+\.?\d*) .* at episode (?P<episode>\d+)"
)


def load_run_labels():
    """Doc hyperparameters.yml de gan nhan mo ta ngan gon cho tung run."""
    labels = {}
    if os.path.exists(HYPERPARAMS_FILE):
        with open(HYPERPARAMS_FILE, "r") as f:
            hp = yaml.safe_load(f) or {}
        for name, cfg in hp.items():
            double_dqn = cfg.get("enable_double_dqn", False)
            dueling_dqn = cfg.get("enable_dueling_dqn", False)
            tag = []
            if double_dqn:
                tag.append("Double")
            if dueling_dqn:
                tag.append("Dueling")
            labels[name] = "+".join(tag) if tag else "Vanilla DQN"
    return labels


def parse_log(path):
    """Tra ve list[(elapsed_seconds, episode, reward)] tu 1 file log."""
    points = []
    start_dt = None
    with open(path, "r") as f:
        for line in f:
            m = LOG_LINE_RE.match(line.strip())
            if not m:
                continue
            dt = datetime.strptime(m.group("ts"), "%m-%d %H:%M:%S")
            if start_dt is None:
                start_dt = dt
            elapsed = (dt - start_dt).total_seconds()
            # datetime.strptime gan nam mac dinh 1900, neu log bat qua nam
            # (vd 12-31 -> 01-01) elapsed se am; sua bang cach cong them 1 nam
            if elapsed < 0:
                elapsed += 365 * 24 * 3600
            points.append((elapsed, int(m.group("episode")), float(m.group("reward"))))
    return points


def summarize(name, points):
    if not points:
        return None
    last_elapsed, last_episode, last_reward = points[-1]
    total_hours = last_elapsed / 3600
    return {
        "name": name,
        "final_reward": last_reward,
        "final_episode": last_episode,
        "total_hours": total_hours,
        "episodes_per_sec": last_episode / last_elapsed if last_elapsed > 0 else 0,
        "reward_per_hour": last_reward / total_hours if total_hours > 0 else 0,
        "num_improvements": len(points),
    }


def print_summary_table(summaries):
    headers = ["Run", "Config", "Best reward", "Episodes", "Thoi gian (h)", "Reward/gio"]
    rows = []
    for s in summaries:
        rows.append([
            s["name"],
            s["label"],
            f"{s['final_reward']:.1f}",
            f"{s['final_episode']:,}",
            f"{s['total_hours']:.2f}",
            f"{s['reward_per_hour']:.1f}",
        ])

    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt_row(cols):
        return " | ".join(c.ljust(w) for c, w in zip(cols, col_widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt_row(r))


def pick_best(summaries):
    # Tieu chi chinh: best reward dat duoc cao nhat (chat luong policy cuoi cung).
    # Tieu chi phu: reward/gio de danh gia hieu qua thoi gian train khi diem so ngang nhau.
    return max(summaries, key=lambda s: (s["final_reward"], s["reward_per_hour"]))


def plot_comparison(all_points, summaries, best_name):
    colors = {"flappybird1": "#4C72B0", "flappybird2": "#DD8452",
              "flappybird3": "#55A868", "flappybird4": "#C44E52"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # 1) Reward vs Episode (hieu qua ve so mau/episode can de dat reward)
    ax = axes[0]
    for name, points in all_points.items():
        episodes = [p[1] for p in points]
        rewards = [p[2] for p in points]
        lw = 3 if name == best_name else 1.5
        ax.plot(episodes, rewards, marker="o", markersize=3, linewidth=lw,
                 label=name, color=colors.get(name))
    ax.set_xlabel("Episode")
    ax.set_ylabel("Best reward")
    ax.set_title("Reward vs Episode (hieu qua ve so mau)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2) Reward vs Thoi gian thuc (hieu qua ve wall-clock time)
    ax = axes[1]
    for name, points in all_points.items():
        hours = [p[0] / 3600 for p in points]
        rewards = [p[2] for p in points]
        lw = 3 if name == best_name else 1.5
        ax.plot(hours, rewards, marker="o", markersize=3, linewidth=lw,
                 label=name, color=colors.get(name))
    ax.set_xlabel("Thoi gian train (gio)")
    ax.set_ylabel("Best reward")
    ax.set_title("Reward vs Thoi gian (hieu qua wall-clock)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3) Bar chart tong ket best reward cuoi cung
    ax = axes[2]
    names = [s["name"] for s in summaries]
    finals = [s["final_reward"] for s in summaries]
    bar_colors = [colors.get(n) for n in names]
    bars = ax.bar(names, finals, color=bar_colors)
    for bar, name in zip(bars, names):
        if name == best_name:
            bar.set_edgecolor("black")
            bar.set_linewidth(2.5)
    for bar, val in zip(bars, finals):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}",
                 ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Best reward dat duoc")
    ax.set_title(f"Tong ket: {best_name} tot nhat")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT_IMG, dpi=150)
    print(f"\nDa luu bieu do so sanh tai: {OUT_IMG}")


def main():
    labels = load_run_labels()

    all_points = {}
    summaries = []
    for i in range(1, 5):
        name = f"flappybird{i}"
        log_path = os.path.join(RUNS_DIR, f"{name}.log")
        if not os.path.exists(log_path):
            print(f"Bo qua {name}: khong tim thay {log_path}")
            continue
        points = parse_log(log_path)
        if not points:
            print(f"Bo qua {name}: khong parse duoc dong log nao")
            continue
        all_points[name] = points
        s = summarize(name, points)
        s["label"] = labels.get(name, "?")
        summaries.append(s)

    if not summaries:
        print("Khong co du lieu log nao de so sanh.")
        return

    print_summary_table(summaries)

    best = pick_best(summaries)
    print(f"\n=> File tot nhat: {best['name']} ({best['label']}) "
          f"- best reward = {best['final_reward']:.1f} "
          f"sau {best['final_episode']:,} episodes / {best['total_hours']:.2f} gio")

    plot_comparison(all_points, summaries, best["name"])


if __name__ == "__main__":
    main()
