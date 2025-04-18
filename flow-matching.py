import argparse
import os
from model import MLP
from utils import get_checker_board_samples, visualize_points, generate_gif
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt


def train(model, start_sample_points, target_sampled_points, ckpt_path="model.pt"):
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    data = torch.Tensor(target_sampled_points)
    if start_sample_points is not None:
        start_data = torch.Tensor(start_sample_points)
    else:
        start_data = None
    training_steps = 100_000
    batch_size = 64
    pbar = tqdm.tqdm(range(training_steps))
    losses = []
    for i in pbar:
        x1 = data[torch.randint(data.size(0), (batch_size,))]
        if start_data is None:
            x0 = torch.randn_like(x1)
        else:
            x0 = start_data[torch.randint(start_data.size(0), (batch_size,))]
        target = x1 - x0
        t = torch.rand(x1.size(0))[:, None]
        xt = (1 - t) * x0 + t * x1
        pred = model(xt, t)  # also add t here
        target = x1 - x0
        loss = ((target - pred[..., 0]) ** 2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())

    torch.save(model.state_dict(), ckpt_path)
    print(f"Saving model to {ckpt_path}")
    return model, ckpt_path


def eval(model, start, end, n=1000, task="x0_to_x2", out_dir="output"):
    torch.manual_seed(42)
    model.eval().requires_grad_(False)
    steps = 1000
    plot_every = 100
    begin_name = task.split("_")[0]
    if begin_name == "x0":
        start = torch.randn(n, 2)
    elif begin_name == "x1" or begin_name == "x2":
        start = torch.Tensor(start)
    save_dir = os.path.join(out_dir, task)
    os.makedirs(save_dir, exist_ok=True)
    ext = ".png"
    sample_idx = 0
    for i, t in enumerate(torch.linspace(0, 1, steps), start=1):
        t_samples = t.expand(start.size(0))[..., None]
        # print(t_samples.shape)
        # approaching the first distribution.
        pred = model(start, t_samples)
        start = start + (1 / steps) * pred[:, :, 0]
        if i % plot_every == 0:
            plt.figure(figsize=(6, 6))
            plt.scatter(end[:, 0], end[:, 1], color="red", marker="o")
            plt.scatter(start[:, 0], start[:, 1], color="green", marker="o")
            # plt.show()
            plt.savefig(os.path.join(save_dir, f"frame_{sample_idx}{ext}"))
            plt.close()
            sample_idx += 1
    generate_gif(save_dir, out_dir, ext=ext)
    model.train().requires_grad_(True)
    print("Done Sampling")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and eval distributions.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/flow_matching",
        help="Output directory.",
    )
    parser.add_argument(
        "--input-size", type=int, default=1000, help="number of samples."
    )
    parser.add_argument(
        "--target-signal",
        type=int,
        default=1,
        help="Decides which set of checkers will be sampled from.",
    )
    parser.add_argument(
        "--start-signal",
        type=int,
        default=-1,
        help="Starting signal -1 is noise, 0 and 1 corresponding to two different checker signals.",
    )

    args = parser.parse_args()
    N = args.input_size
    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)
    model = MLP(layers=5, channels=512, t_dof=1)
    target_sample_points = get_checker_board_samples(
        N=N,
        checker_signal=args.target_signal,
        save_path=os.path.join(outdir, "target_sample_points.png"),
    )
    if args.start_signal < 0:
        start_sample_points = np.random.randn(N, 2)
    else:
        start_sample_points = get_checker_board_samples(
            N=N,
            checker_signal=args.start_signal,
            save_path=os.path.join(outdir, "start_sample_points.png"),
        )
    visualize_points(
        [start_sample_points, target_sample_points],
        save_path=os.path.join(outdir, "distributions.png"),
    )
    ckpt_path = os.path.join(outdir, "model.pt")
    if not os.path.exists(ckpt_path):
        model, ckpt_path = train(
            model=model,
            start_sample_points=start_sample_points,
            target_sampled_points=target_sample_points,
            ckpt_path=ckpt_path,
        )
    else:
        print(f"{ckpt_path} exists.")
    print(f"Loading model from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    if args.start_signal < 0:
        start = None
    else:
        start = start_sample_points
    task = f"x{args.start_signal+1}_to_x{args.target_signal+1}"
    eval(
        model=model,
        n=N,
        start=start,
        end=target_sample_points,
        task=task,
        out_dir=outdir,
    )
