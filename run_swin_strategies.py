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

DEFAULT_CONFIG = {
    "model_key": "swin_tiny",
    "timm_name": "swin_tiny_patch4_window7_224",
    "input_size": 224,
    "train_fraction": 1.0,
    "batch_size": 32,
    "num_workers": 0,
    "num_epochs": 12,
    "patience": 3,
    "weight_decay": 1e-4,
    "seed": 42,
}

STRATEGY_CONFIGS = {
    "feature_extraction": {
        "base_lr": 1e-4,
        "description": "Freeze the backbone and train only the classification head.",
    },
    "gradual_unfreezing": {
        "base_lr": 1e-4,
        "description": "Unfreeze the Swin backbone in stages during training.",
    },
    "discriminative_learning_rates": {
        "base_lr": 1e-4,
        "description": "Train all layers with lower learning rates for earlier stages.",
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


def make_loaders(config: dict):
    train_dataset, val_dataset, test_dataset = build_datasets(config["input_size"])
    train_subset = create_subset(train_dataset, config["train_fraction"], config["seed"])
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    return train_subset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_model(config: dict) -> nn.Module:
    model = timm.create_model(config["timm_name"], pretrained=True, num_classes=NUM_CLASSES)
    return model.to(DEVICE)


def freeze_all_but_head(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    classifier = model.get_classifier()
    if isinstance(classifier, nn.Module):
        for param in classifier.parameters():
            param.requires_grad = True


def set_requires_grad_by_prefix(model: nn.Module, prefixes: list[str]) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in prefixes)


def get_gradual_stage(epoch_index: int) -> tuple[str, list[str]]:
    if epoch_index < 2:
        return "head_only", ["head"]
    if epoch_index < 4:
        return "head_plus_last_stage", ["layers.3", "norm", "head"]
    return "full_model", [""]


def apply_gradual_unfreezing(model: nn.Module, epoch_index: int) -> str:
    stage_name, prefixes = get_gradual_stage(epoch_index)
    set_requires_grad_by_prefix(model, prefixes)
    return stage_name


def build_optimizer(model: nn.Module, strategy_name: str, base_lr: float, weight_decay: float):
    if strategy_name == "discriminative_learning_rates":
        param_groups = []
        group_specs = [
            ("patch_embed", base_lr * 0.25),
            ("layers.0", base_lr * 0.25),
            ("layers.1", base_lr * 0.5),
            ("layers.2", base_lr * 0.75),
            ("layers.3", base_lr),
            ("norm", base_lr),
            ("head", base_lr * 2.0),
        ]
        for prefix, group_lr in group_specs:
            params = [
                param
                for name, param in model.named_parameters()
                if name.startswith(prefix) and param.requires_grad
            ]
            if params:
                param_groups.append({"params": params, "lr": group_lr, "group_name": prefix})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        lr_summary = [{"group": group["group_name"], "lr": group["lr"]} for group in param_groups]
        return optimizer, lr_summary

    params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    return optimizer, [{"group": "trainable_params", "lr": base_lr}]


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


def count_trainable_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def count_total_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def estimate_model_size_mb(model: nn.Module) -> float:
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)


def to_serializable_metrics(metrics: dict) -> dict:
    return {key: float(value) for key, value in metrics.items()}


def run_strategy(strategy_name: str, config: dict, output_dir: Path) -> dict:
    set_seed(config["seed"])
    criterion = nn.CrossEntropyLoss()
    (
        train_subset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = make_loaders(config)

    model = build_model(config)
    if strategy_name == "feature_extraction":
        freeze_all_but_head(model)
    elif strategy_name == "gradual_unfreezing":
        apply_gradual_unfreezing(model, epoch_index=0)

    optimizer, lr_summary = build_optimizer(
        model,
        strategy_name=strategy_name,
        base_lr=STRATEGY_CONFIGS[strategy_name]["base_lr"],
        weight_decay=config["weight_decay"],
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auc": [],
        "trainable_params": [],
        "stage": [],
        "epoch_lr_groups": [],
    }

    best_state = None
    best_val_auc = float("-inf")
    best_epoch = -1
    wait = 0

    checkpoint_path = output_dir / "checkpoints" / (
        f"bloodmnist_swin_tiny_{strategy_name}_{int(config['train_fraction'] * 100)}pct_best.pth"
    )
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)

    start_time = time.perf_counter()

    for epoch in range(config["num_epochs"]):
        stage_name = "full_model"
        if strategy_name == "gradual_unfreezing":
            stage_name = apply_gradual_unfreezing(model, epoch)
            optimizer, lr_summary = build_optimizer(
                model,
                strategy_name="full_fine_tuning",
                base_lr=STRATEGY_CONFIGS[strategy_name]["base_lr"],
                weight_decay=config["weight_decay"],
            )
        current_lrs = [{"group": group["group_name"] if "group_name" in group else f"group_{idx}", "lr": group["lr"]} for idx, group in enumerate(optimizer.param_groups)]

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_metrics = evaluate(model, val_loader, criterion)
        current_trainable_params = count_trainable_params(model)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_acc"].append(float(val_metrics["accuracy"]))
        history["val_f1"].append(float(val_metrics["macro_f1"]))
        history["val_auc"].append(float(val_metrics["auc"]))
        history["trainable_params"].append(int(current_trainable_params))
        history["stage"].append(stage_name)
        history["epoch_lr_groups"].append(current_lrs)

        improved = val_metrics["auc"] > best_val_auc
        if improved:
            best_val_auc = float(val_metrics["auc"])
            best_epoch = epoch + 1
            wait = 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
        else:
            wait += 1

        print(
            f"[{strategy_name}] Epoch {epoch + 1:02d}/{config['num_epochs']} | "
            f"Stage: {stage_name} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.3f} | "
            f"Val F1: {val_metrics['macro_f1']:.3f} | Val AUC: {val_metrics['auc']:.3f} | "
            f"Trainable Params: {current_trainable_params:,}"
        )

        if wait >= config["patience"]:
            print(f"[{strategy_name}] Early stopping at epoch {epoch + 1}.")
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint was saved for strategy {strategy_name}.")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, criterion)
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "strategy_name": strategy_name,
        "description": STRATEGY_CONFIGS[strategy_name]["description"],
        "model_key": config["model_key"],
        "timm_name": config["timm_name"],
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "history": history,
        "test_metrics": to_serializable_metrics(test_metrics),
        "trainable_params_at_end": count_trainable_params(model),
        "total_params": count_total_params(model),
        "model_size_mb": estimate_model_size_mb(model),
        "lr_summary": lr_summary,
        "elapsed_seconds": elapsed_seconds,
        "dataset_sizes": {
            "train": len(train_subset),
            "val": len(val_dataset),
            "test": len(test_dataset),
        },
        "config": config,
    }


def save_strategy_json(result: dict, output_dir: Path) -> Path:
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    json_path = results_dir / f"swin_tiny_{result['strategy_name']}_metrics.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return json_path


def save_summary_csv(results: list[dict], output_dir: Path) -> Path:
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    csv_path = results_dir / "swin_tiny_strategy_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "strategy_name",
                "best_epoch",
                "best_val_auc",
                "test_accuracy",
                "test_macro_f1",
                "test_auc",
                "elapsed_seconds",
                "trainable_params_at_end",
                "total_params",
                "model_size_mb",
                "checkpoint_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "strategy_name": result["strategy_name"],
                    "best_epoch": result["best_epoch"],
                    "best_val_auc": f"{result['best_val_auc']:.6f}",
                    "test_accuracy": f"{result['test_metrics']['accuracy']:.6f}",
                    "test_macro_f1": f"{result['test_metrics']['macro_f1']:.6f}",
                    "test_auc": f"{result['test_metrics']['auc']:.6f}",
                    "elapsed_seconds": f"{result['elapsed_seconds']:.2f}",
                    "trainable_params_at_end": result["trainable_params_at_end"],
                    "total_params": result["total_params"],
                    "model_size_mb": f"{result['model_size_mb']:.2f}",
                    "checkpoint_path": result["checkpoint_path"],
                }
            )
    return csv_path


def plot_training_curves(results: list[dict], output_dir: Path) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for result in results:
        epochs = range(1, len(result["history"]["train_loss"]) + 1)
        label = result["strategy_name"]
        axes[0].plot(epochs, result["history"]["train_acc"], marker="o", label=f"{label} train")
        axes[0].plot(epochs, result["history"]["val_acc"], marker="s", linestyle="--", label=f"{label} val")
        axes[1].plot(epochs, result["history"]["val_auc"], marker="o", label=label)

    axes[0].set_title("Swin-Tiny Strategy Accuracy Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend()

    axes[1].set_title("Swin-Tiny Strategy Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    plot_path = figures_dir / "bloodmnist_swin_tiny_strategy_training_curves.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_test_metrics(results: list[dict], output_dir: Path) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)
    labels = [result["strategy_name"] for result in results]
    accuracy = [result["test_metrics"]["accuracy"] for result in results]
    macro_f1 = [result["test_metrics"]["macro_f1"] for result in results]
    auc = [result["test_metrics"]["auc"] for result in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, accuracy, width=width, label="Accuracy")
    ax.bar(x, macro_f1, width=width, label="Macro F1")
    ax.bar(x + width, auc, width=width, label="AUC")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Swin-Tiny Fine-Tuning Strategy Comparison on BloodMNIST")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    plot_path = figures_dir / "bloodmnist_swin_tiny_strategy_test_metrics.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Swin-Tiny fine-tuning strategies on BloodMNIST.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=list(STRATEGY_CONFIGS.keys()),
        choices=list(STRATEGY_CONFIGS.keys()),
        help="Strategies to run.",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--num-workers", type=int, default=DEFAULT_CONFIG["num_workers"])
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_CONFIG["train_fraction"])
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = copy.deepcopy(DEFAULT_CONFIG)
    config.update(
        {
            "num_epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "train_fraction": args.train_fraction,
            "seed": args.seed,
        }
    )

    print(f"Using device: {DEVICE}")
    print(f"Running strategies: {args.strategies}")
    print(f"Train fraction: {config['train_fraction']:.0%}")
    print(f"Epochs: {config['num_epochs']}")
    print(f"Patience: {config['patience']}")

    results = []
    json_paths = []
    for strategy_name in args.strategies:
        print(f"\n=== Running {strategy_name} ===")
        result = run_strategy(strategy_name=strategy_name, config=config, output_dir=args.output_dir)
        results.append(result)
        json_paths.append(save_strategy_json(result, args.output_dir))

    csv_path = save_summary_csv(results, args.output_dir)
    curves_path = plot_training_curves(results, args.output_dir)
    bars_path = plot_test_metrics(results, args.output_dir)

    print("\nSummary:")
    for result in results:
        metrics = result["test_metrics"]
        print(
            f"- {result['strategy_name']}: "
            f"Test Acc={metrics['accuracy']:.3f}, "
            f"Macro F1={metrics['macro_f1']:.3f}, "
            f"AUC={metrics['auc']:.3f}, "
            f"Best Epoch={result['best_epoch']}, "
            f"Checkpoint={result['checkpoint_path']}"
        )

    print("\nSaved artifacts:")
    for json_path in json_paths:
        print(f"- {json_path}")
    print(f"- {csv_path}")
    print(f"- {curves_path}")
    print(f"- {bars_path}")


if __name__ == "__main__":
    main()
