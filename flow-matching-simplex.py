import tqdm

import torch
import numpy as np

import matplotlib.pyplot as plt
import os
import argparse

from model import MLP
from utils import get_checker_board_samples, visualize_points, generate_gif


def sample_simplex_batch(n, b, alpha_factor=1.0):
    alpha = torch.ones(n) * alpha_factor
    return torch.distributions.Dirichlet(alpha).sample((b,))


def interpolate_samples(t_samples, anchor_samples):
    B, N = t_samples.shape
    D = anchor_samples[0].shape[-1]
    interpolated = torch.zeros(B, D)
    for i, sample in enumerate(anchor_samples):
        if sample.dim() == 1:
            sample = sample.unsqueeze(0).expand(B, -1)
        interpolated += t_samples[:, i : i + 1] * sample
    return interpolated


def train(model, sampled_points_1, sampled_points_2, t_dof=2, ckpt_path="model.pt"):
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    alpha_factor = 0.3
    data = [
        torch.Tensor(sampled_points)
        for sampled_points in [sampled_points_1, sampled_points_2]
    ]
    training_steps = 100_000
    batch_size = 64
    pbar = tqdm.tqdm(range(training_steps))
    losses = []
    for i in pbar:
        x1 = data[0][torch.randint(data[0].size(0), (batch_size,))]
        x2 = data[1][torch.randint(data[1].size(0), (batch_size,))]
        x0 = torch.randn_like(x1)
        # target = x1 - x0
        # t = torch.rand(x1.size(0))
        t = sample_simplex_batch(t_dof + 1, batch_size, alpha_factor=alpha_factor)
        t[:, 2] = 1 - t[:, 0] - t[:, 1]  # making sure it adds to 1.
        # xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        xt = interpolate_samples(t, [x1, x2, x0])
        pred = model(xt, t[..., :t_dof])  # also add t here
        # print("pred", pred.shape)
        target = torch.stack([x1 - x0, x2 - x0], dim=1)
        # print("target", target.shape)
        loss = ((target - pred) ** 2).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        pbar.set_postfix(loss=loss.item())
        losses.append(loss.item())
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saving model to {ckpt_path}")


def eval(model, start, end, n=1000, task="x0_to_x2", out_dir="output"):
    torch.manual_seed(42)
    model.eval().requires_grad_(False)
    steps = 1000
    plot_every = 100
    target_name = task.split("_")[-1]
    begin_name = task.split("_")[0]
    if target_name == "x1":
        pred_ind = 0
    elif target_name == "x2":
        pred_ind = 1
    else:
        print(f"target_name is not defined {target_name}")
        return
    if begin_name == "x0":
        start = torch.randn(n, 2)
    elif begin_name == "x1" or begin_name == "x2":
        start = torch.Tensor(start)
    save_dir = os.path.join(out_dir, task)
    os.makedirs(save_dir, exist_ok=True)
    ext = ".png"
    sample_idx = 0
    for i, t in enumerate(torch.linspace(0, 1, steps), start=1):
        if task == "x0_to_x2":
            t_samples = torch.stack(
                [t.expand(start.size(0)), torch.zeros(start.size(0))], dim=-1
            )
        elif task == "x0_to_x1":
            t_samples = torch.stack(
                [torch.zeros(start.size(0)), t.expand(start.size(0))], dim=-1
            )
        elif task == "x1_to_x2":
            t_2 = torch.ones(n) * t
            t_samples = torch.stack([torch.ones(n) - t_2, t_2], dim=-1)
        else:
            print(f"task is unknown {task}")
            return
        # print(t_samples.shape)
        # approaching the first distribution.
        pred = model(start, t_samples)
        start = start + (1 / steps) * pred[:, :, pred_ind]
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
        default="output/flow_matching_simplex",
        help="Output directory.",
    )
    parser.add_argument(
        "--input-size", type=int, default=1000, help="number of samples."
    )
    args = parser.parse_args()
    N = args.input_size
    outdir = args.out_dir
    os.makedirs(outdir, exist_ok=True)
    model = MLP(layers=5, channels=512, t_dof=2)
    sampled_points_1 = get_checker_board_samples(
        N=N, checker_signal=1, save_path=os.path.join(outdir, "sampled_points_1.png")
    )
    sampled_points_2 = get_checker_board_samples(
        N=N, checker_signal=0, save_path=os.path.join(outdir, "sampled_points_2.png")
    )

    noise = np.random.randn(N, 2)
    visualize_points(
        [sampled_points_1, sampled_points_2, noise],
        save_path=os.path.join(outdir, "distributions.png"),
    )
    ckpt_path = os.path.join(outdir, "model.pt")
    if not os.path.exists(ckpt_path):
        train(
            model=model,
            sampled_points_1=sampled_points_1,
            sampled_points_2=sampled_points_2,
            ckpt_path=ckpt_path,
        )
    else:
        print(f"Foud {ckpt_path}")
    print(f"Loading model from {ckpt_path}.")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    eval(
        model=model,
        n=N,
        start=None,
        end=sampled_points_2,
        task="x0_to_x2",
        out_dir=outdir,
    )
    eval(
        model=model,
        n=N,
        start=None,
        end=sampled_points_1,
        task="x0_to_x1",
        out_dir=outdir,
    )
    eval(
        model=model,
        n=N,
        start=sampled_points_1,
        end=sampled_points_2,
        task="x1_to_x2",
        out_dir=outdir,
    )
