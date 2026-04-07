from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from medmnist import BloodMNIST
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 8
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TRAIN_FRACTIONS = [1.0, 0.5, 0.25, 0.10, 0.05]
DEFAULT_SEEDS = [42, 43, 44]

DEFAULT_CONFIG = {
    "model_key": "swin_tiny",
    "timm_name": "swin_tiny_patch4_window7_224",
    "input_size": 224,
    "batch_size": 32,
    "num_workers": 0,
    "weight_decay": 1e-4,
    "seed": 42,
}

INIT_CONFIGS = {
    "pretrained": {
        "pretrained": True,
        "lr": 1e-4,
        "num_epochs": 12,
        "patience": 3,
        "description": "ImageNet-pretrained Swin-Tiny with full fine-tuning.",
    },
    "scratch": {
        "pretrained": False,
        "lr": 5e-4,
        "num_epochs": 30,
        "patience": 5,
        "description": "Randomly initialized Swin-Tiny trained from scratch.",
    },
}


def set_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def build_datasets(image_size: int):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    train_dataset = BloodMNIST(split="train", transform=transform, download=True)
    val_dataset = BloodMNIST(split="val", transform=transform, download=True)
    test_dataset = BloodMNIST(split="test", transform=transform, download=True)
    return train_dataset, val_dataset, test_dataset


def create_subset(dataset, fraction: float, seed_value: int):
    if fraction >= 1.0:
        return dataset

    labels = np.asarray(dataset.labels).reshape(-1)
    indices = np.arange(len(dataset))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=seed_value)
    subset_indices, _ = next(splitter.split(indices, labels))
    indices = subset_indices.tolist()
    return Subset(dataset, indices)


def make_loaders(config: dict, train_fraction: float):
    train_dataset, val_dataset, test_dataset = build_datasets(config["input_size"])
    train_subset = create_subset(train_dataset, train_fraction, config["seed"])
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_subset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_model(config: dict, pretrained: bool) -> nn.Module:
    model = timm.create_model(
        config["timm_name"],
        pretrained=pretrained,
        num_classes=NUM_CLASSES,
    )
    return model.to(DEVICE)


def multiclass_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
            labels=list(range(NUM_CLASSES)),
        )
    except ValueError:
        return float("nan")


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    y_prob = torch.cat(all_probs).numpy()
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "auc": multiclass_auc(y_true, y_prob),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion: nn.Module) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.squeeze().long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def count_total_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def estimate_model_size_mb(model: nn.Module) -> float:
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)


def serialize_metrics(metrics: dict) -> dict:
    return {key: float(value) for key, value in metrics.items()}


def run_experiment(
    init_mode: str,
    train_fraction: float,
    config: dict,
    output_dir: Path,
) -> dict:
    init_cfg = INIT_CONFIGS[init_mode]
    set_seed(config["seed"])
    criterion = nn.CrossEntropyLoss()
    (
        train_subset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = make_loaders(config, train_fraction)

    model = build_model(config, pretrained=init_cfg["pretrained"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=init_cfg["lr"],
        weight_decay=config["weight_decay"],
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auc": [],
    }

    best_state = None
    best_val_auc = float("-inf")
    best_epoch = -1
    wait = 0

    ratio_label = f"{int(train_fraction * 100)}pct"
    checkpoint_path = output_dir / "checkpoints" / (
        f"bloodmnist_swin_tiny_{init_mode}_{ratio_label}_seed{config['seed']}_best.pth"
    )
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

    start_time = time.perf_counter()

    for epoch in range(init_cfg["num_epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, criterion)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(float(val_metrics["accuracy"]))
        history["val_f1"].append(float(val_metrics["macro_f1"]))
        history["val_auc"].append(float(val_metrics["auc"]))

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = float(val_metrics["auc"])
            best_epoch = epoch + 1
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
        else:
            wait += 1

        print(
            f"[{init_mode} | {ratio_label}] Epoch {epoch + 1:02d}/{init_cfg['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.3f} | "
            f"Val F1: {val_metrics['macro_f1']:.3f} | Val AUC: {val_metrics['auc']:.3f}"
        )

        if wait >= init_cfg["patience"]:
            print(f"[{init_mode} | {ratio_label}] Early stopping at epoch {epoch + 1}.")
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint was saved for {init_mode} at {ratio_label}.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion)
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "model_key": config["model_key"],
        "timm_name": config["timm_name"],
        "init_mode": init_mode,
        "description": init_cfg["description"],
        "train_fraction": train_fraction,
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "history": history,
        "test_metrics": serialize_metrics(test_metrics),
        "total_params": count_total_params(model),
        "model_size_mb": estimate_model_size_mb(model),
        "elapsed_seconds": elapsed_seconds,
        "dataset_sizes": {
            "train": len(train_subset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "config": {
            **config,
            "lr": init_cfg["lr"],
            "num_epochs": init_cfg["num_epochs"],
            "patience": init_cfg["patience"],
            "pretrained": init_cfg["pretrained"],
        },
    }


def save_individual_json(result: dict, output_dir: Path) -> Path:
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    ratio_label = f"{int(result['train_fraction'] * 100)}pct"
    json_path = results_dir / (
        f"swin_tiny_{result['init_mode']}_{ratio_label}_seed{result['config']['seed']}_metrics.json"
    )
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return json_path


def save_runs_csv(results: list[dict], output_dir: Path) -> Path:
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    csv_path = results_dir / "swin_tiny_pretrained_vs_scratch_runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "init_mode",
                "train_fraction",
                "seed",
                "best_epoch",
                "best_val_auc",
                "test_accuracy",
                "test_macro_f1",
                "test_auc",
                "elapsed_seconds",
                "total_params",
                "model_size_mb",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "init_mode": result["init_mode"],
                    "train_fraction": f"{result['train_fraction']:.2f}",
                    "seed": result["config"]["seed"],
                    "best_epoch": result["best_epoch"],
                    "best_val_auc": f"{result['best_val_auc']:.6f}",
                    "test_accuracy": f"{result['test_metrics']['accuracy']:.6f}",
                    "test_macro_f1": f"{result['test_metrics']['macro_f1']:.6f}",
                    "test_auc": f"{result['test_metrics']['auc']:.6f}",
                    "elapsed_seconds": f"{result['elapsed_seconds']:.2f}",
                    "total_params": result["total_params"],
                    "model_size_mb": f"{result['model_size_mb']:.2f}",
                    "checkpoint_path": result["checkpoint_path"],
                }
            )
    return csv_path


def aggregate_results(results: list[dict]) -> list[dict]:
    grouped = {}
    for result in results:
        key = (result["init_mode"], result["train_fraction"])
        grouped.setdefault(key, []).append(result)

    aggregated = []
    for (init_mode, train_fraction), group in sorted(grouped.items(), key=lambda item: (item[0][0], -item[0][1])):
        test_accuracy = np.array([item["test_metrics"]["accuracy"] for item in group], dtype=float)
        test_macro_f1 = np.array([item["test_metrics"]["macro_f1"] for item in group], dtype=float)
        test_auc = np.array([item["test_metrics"]["auc"] for item in group], dtype=float)
        best_val_auc = np.array([item["best_val_auc"] for item in group], dtype=float)
        best_epoch = np.array([item["best_epoch"] for item in group], dtype=float)
        elapsed_seconds = np.array([item["elapsed_seconds"] for item in group], dtype=float)
        aggregated.append(
            {
                "init_mode": init_mode,
                "train_fraction": train_fraction,
                "num_seeds": len(group),
                "seeds": [item["config"]["seed"] for item in group],
                "test_accuracy_mean": float(test_accuracy.mean()),
                "test_accuracy_std": float(test_accuracy.std(ddof=0)),
                "test_macro_f1_mean": float(test_macro_f1.mean()),
                "test_macro_f1_std": float(test_macro_f1.std(ddof=0)),
                "test_auc_mean": float(test_auc.mean()),
                "test_auc_std": float(test_auc.std(ddof=0)),
                "best_val_auc_mean": float(best_val_auc.mean()),
                "best_val_auc_std": float(best_val_auc.std(ddof=0)),
                "best_epoch_mean": float(best_epoch.mean()),
                "best_epoch_std": float(best_epoch.std(ddof=0)),
                "elapsed_seconds_mean": float(elapsed_seconds.mean()),
                "elapsed_seconds_std": float(elapsed_seconds.std(ddof=0)),
                "total_params": group[0]["total_params"],
                "model_size_mb": group[0]["model_size_mb"],
                "description": group[0]["description"],
            }
        )
    return aggregated


def save_aggregate_json(aggregated_results: list[dict], output_dir: Path) -> Path:
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    json_path = results_dir / "swin_tiny_pretrained_vs_scratch_aggregate.json"
    json_path.write_text(json.dumps(aggregated_results, indent=2), encoding="utf-8")
    return json_path


def save_aggregate_csv(aggregated_results: list[dict], output_dir: Path) -> Path:
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    csv_path = results_dir / "swin_tiny_pretrained_vs_scratch_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "init_mode",
                "train_fraction",
                "num_seeds",
                "seeds",
                "test_accuracy_mean",
                "test_accuracy_std",
                "test_macro_f1_mean",
                "test_macro_f1_std",
                "test_auc_mean",
                "test_auc_std",
                "best_val_auc_mean",
                "best_val_auc_std",
                "best_epoch_mean",
                "best_epoch_std",
                "elapsed_seconds_mean",
                "elapsed_seconds_std",
                "total_params",
                "model_size_mb",
            ],
        )
        writer.writeheader()
        for result in aggregated_results:
            writer.writerow(
                {
                    "init_mode": result["init_mode"],
                    "train_fraction": f"{result['train_fraction']:.2f}",
                    "num_seeds": result["num_seeds"],
                    "seeds": ",".join(str(seed) for seed in result["seeds"]),
                    "test_accuracy_mean": f"{result['test_accuracy_mean']:.6f}",
                    "test_accuracy_std": f"{result['test_accuracy_std']:.6f}",
                    "test_macro_f1_mean": f"{result['test_macro_f1_mean']:.6f}",
                    "test_macro_f1_std": f"{result['test_macro_f1_std']:.6f}",
                    "test_auc_mean": f"{result['test_auc_mean']:.6f}",
                    "test_auc_std": f"{result['test_auc_std']:.6f}",
                    "best_val_auc_mean": f"{result['best_val_auc_mean']:.6f}",
                    "best_val_auc_std": f"{result['best_val_auc_std']:.6f}",
                    "best_epoch_mean": f"{result['best_epoch_mean']:.6f}",
                    "best_epoch_std": f"{result['best_epoch_std']:.6f}",
                    "elapsed_seconds_mean": f"{result['elapsed_seconds_mean']:.2f}",
                    "elapsed_seconds_std": f"{result['elapsed_seconds_std']:.2f}",
                    "total_params": result["total_params"],
                    "model_size_mb": f"{result['model_size_mb']:.2f}",
                }
            )
    return csv_path


def plot_accuracy_auc(aggregated_results: list[dict], output_dir: Path, train_fractions: list[float]) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)

    grouped = {"pretrained": {}, "scratch": {}}
    for result in aggregated_results:
        grouped[result["init_mode"]][result["train_fraction"]] = result

    x_labels = [f"{int(fraction * 100)}%" for fraction in train_fractions]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for init_mode, marker in [("pretrained", "o"), ("scratch", "s")]:
        acc_values = [grouped[init_mode][fraction]["test_accuracy_mean"] for fraction in train_fractions]
        acc_errors = [grouped[init_mode][fraction]["test_accuracy_std"] for fraction in train_fractions]
        auc_values = [grouped[init_mode][fraction]["test_auc_mean"] for fraction in train_fractions]
        auc_errors = [grouped[init_mode][fraction]["test_auc_std"] for fraction in train_fractions]
        axes[0].errorbar(
            x_labels,
            acc_values,
            yerr=acc_errors,
            marker=marker,
            linewidth=2,
            capsize=4,
            label=f"{init_mode} mean±std",
        )
        axes[1].errorbar(
            x_labels,
            auc_values,
            yerr=auc_errors,
            marker=marker,
            linewidth=2,
            capsize=4,
            label=f"{init_mode} mean±std",
        )

    axes[0].set_title("Test Accuracy vs Training Fraction")
    axes[0].set_xlabel("Training Fraction")
    axes[0].set_ylabel("Accuracy (mean ± std)")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title("Test AUC vs Training Fraction")
    axes[1].set_xlabel("Training Fraction")
    axes[1].set_ylabel("AUC (mean ± std)")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    plot_path = figures_dir / "bloodmnist_swin_pretrained_vs_scratch_accuracy_auc.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_macro_f1(aggregated_results: list[dict], output_dir: Path, train_fractions: list[float]) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)

    grouped = {"pretrained": {}, "scratch": {}}
    for result in aggregated_results:
        grouped[result["init_mode"]][result["train_fraction"]] = result

    x = np.arange(len(train_fractions))
    width = 0.35

    pretrained_values = [
        grouped["pretrained"][fraction]["test_macro_f1_mean"] for fraction in train_fractions
    ]
    pretrained_errors = [
        grouped["pretrained"][fraction]["test_macro_f1_std"] for fraction in train_fractions
    ]
    scratch_values = [
        grouped["scratch"][fraction]["test_macro_f1_mean"] for fraction in train_fractions
    ]
    scratch_errors = [
        grouped["scratch"][fraction]["test_macro_f1_std"] for fraction in train_fractions
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, pretrained_values, width=width, yerr=pretrained_errors, capsize=4, label="pretrained")
    ax.bar(x + width / 2, scratch_values, width=width, yerr=scratch_errors, capsize=4, label="scratch")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(fraction * 100)}%" for fraction in train_fractions])
    ax.set_xlabel("Training Fraction")
    ax.set_ylabel("Macro F1 (mean ± std)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Swin-Tiny Macro F1: Pretrained vs From Scratch")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    plot_path = figures_dir / "bloodmnist_swin_pretrained_vs_scratch_macro_f1.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Swin-Tiny pretrained vs from-scratch experiments across data fractions."
    )
    parser.add_argument(
        "--train-fractions",
        nargs="+",
        type=float,
        default=TRAIN_FRACTIONS,
        help="Training fractions to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to evaluate for each configuration.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        }
    )

    train_fractions = list(args.train_fractions)
    seeds = list(args.seeds)

    print(f"Using device: {DEVICE}")
    print(f"Train fractions: {train_fractions}")
    print(f"Seeds: {seeds}")
    print(f"Batch size: {config['batch_size']}")

    results = []
    json_paths = []
    for init_mode in ["pretrained", "scratch"]:
        for train_fraction in train_fractions:
            for seed in seeds:
                run_config = copy.deepcopy(config)
                run_config["seed"] = seed
                print(f"\n=== Running {init_mode} with {train_fraction:.0%} data | seed={seed} ===")
                result = run_experiment(
                    init_mode=init_mode,
                    train_fraction=train_fraction,
                    config=run_config,
                    output_dir=args.output_dir,
                )
                results.append(result)
                json_paths.append(save_individual_json(result, args.output_dir))

    aggregated_results = aggregate_results(results)
    runs_csv_path = save_runs_csv(results, args.output_dir)
    summary_csv_path = save_aggregate_csv(aggregated_results, args.output_dir)
    summary_json_path = save_aggregate_json(aggregated_results, args.output_dir)
    acc_auc_path = plot_accuracy_auc(aggregated_results, args.output_dir, train_fractions)
    f1_path = plot_macro_f1(aggregated_results, args.output_dir, train_fractions)

    print("\nAggregated summary:")
    for result in aggregated_results:
        print(
            f"- {result['init_mode']} | {result['train_fraction']:.0%} | "
            f"Seeds={result['seeds']}: "
            f"Test Acc={result['test_accuracy_mean']:.3f}±{result['test_accuracy_std']:.3f}, "
            f"Macro F1={result['test_macro_f1_mean']:.3f}±{result['test_macro_f1_std']:.3f}, "
            f"AUC={result['test_auc_mean']:.3f}±{result['test_auc_std']:.3f}"
        )

    print("\nSaved artifacts:")
    for json_path in json_paths:
        print(f"- {json_path}")
    print(f"- {runs_csv_path}")
    print(f"- {summary_csv_path}")
    print(f"- {summary_json_path}")
    print(f"- {acc_auc_path}")
    print(f"- {f1_path}")


if __name__ == "__main__":
    main()
