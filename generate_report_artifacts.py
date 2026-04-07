from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from medmnist import BloodMNIST, INFO
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 8
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224
BATCH_SIZE = 32

LABEL_MAP = {int(key): value for key, value in INFO["bloodmnist"]["label"].items()}

ARCHITECTURE_CONFIGS = [
    {
        "model_key": "resnet50",
        "timm_name": "resnet50",
        "checkpoint_path": CHECKPOINTS_DIR / "bloodmnist_resnet50_full_fine_tuning_100pct_best.pth",
    },
    {
        "model_key": "swin_tiny",
        "timm_name": "swin_tiny_patch4_window7_224",
        "checkpoint_path": CHECKPOINTS_DIR / "bloodmnist_swin_tiny_full_fine_tuning_100pct_best.pth",
    },
    {
        "model_key": "vit_b16",
        "timm_name": "vit_base_patch16_224",
        "checkpoint_path": CHECKPOINTS_DIR / "bloodmnist_vit_b16_full_fine_tuning_100pct_best.pth",
    },
]

BEST_MODEL_CONFIG = {
    "model_key": "swin_tiny",
    "timm_name": "swin_tiny_patch4_window7_224",
    "checkpoint_path": CHECKPOINTS_DIR / "bloodmnist_swin_tiny_gradual_unfreezing_100pct_best.pth",
    "strategy_name": "gradual_unfreezing",
}


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_test_datasets() -> tuple[BloodMNIST, BloodMNIST]:
    raw_dataset = BloodMNIST(split="test", download=True)
    eval_dataset = BloodMNIST(split="test", transform=build_eval_transform(), download=True)
    return raw_dataset, eval_dataset


def make_test_loader(dataset: BloodMNIST) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(timm_name: str) -> nn.Module:
    model = timm.create_model(timm_name, pretrained=False, num_classes=NUM_CLASSES)
    return model.to(DEVICE)


def load_checkpoint(model: nn.Module, checkpoint_path: Path) -> nn.Module:
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def multiclass_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return roc_auc_score(
        y_true,
        y_prob,
        multi_class="ovr",
        average="macro",
        labels=list(range(NUM_CLASSES)),
    )


def count_total_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def estimate_model_size_mb(model: nn.Module) -> float:
    total_bytes = 0
    for tensor in list(model.parameters()) + list(model.buffers()):
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 ** 2)


def evaluate_model(model: nn.Module, loader: DataLoader) -> dict:
    criterion = nn.CrossEntropyLoss()
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
    wrong_mask = y_pred != y_true
    wrong_confidence = y_prob[np.arange(len(y_pred)), y_pred]

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "auc": multiclass_auc(y_true, y_prob),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "wrong_mask": wrong_mask,
        "wrong_confidence": wrong_confidence,
    }


def benchmark_latency_ms_per_image(model: nn.Module, batch_size: int = 32, warmup_steps: int = 10, timed_steps: int = 30) -> float:
    dummy_input = torch.randn(batch_size, 3, INPUT_SIZE, INPUT_SIZE, device=DEVICE)

    with torch.no_grad():
        for _ in range(warmup_steps):
            _ = model(dummy_input)
        if DEVICE == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        for _ in range(timed_steps):
            _ = model(dummy_input)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - start_time

    total_images = batch_size * timed_steps
    return elapsed_seconds * 1000.0 / total_images


def save_architecture_benchmark(loader: DataLoader) -> Path:
    rows = []
    for config in ARCHITECTURE_CONFIGS:
        model = build_model(config["timm_name"])
        load_checkpoint(model, config["checkpoint_path"])
        metrics = evaluate_model(model, loader)
        latency_ms = benchmark_latency_ms_per_image(model)
        rows.append(
            {
                "model_key": config["model_key"],
                "timm_name": config["timm_name"],
                "checkpoint_path": str(config["checkpoint_path"].relative_to(PROJECT_ROOT)),
                "device_used_for_latency": DEVICE,
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "test_auc": metrics["auc"],
                "total_params": count_total_params(model),
                "model_size_mb": estimate_model_size_mb(model),
                "inference_latency_ms_per_image": latency_ms,
            }
        )

    csv_path = RESULTS_DIR / "bloodmnist_architecture_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_key",
                "timm_name",
                "checkpoint_path",
                "device_used_for_latency",
                "test_accuracy",
                "test_macro_f1",
                "test_auc",
                "total_params",
                "model_size_mb",
                "inference_latency_ms_per_image",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "test_accuracy": f"{row['test_accuracy']:.6f}",
                    "test_macro_f1": f"{row['test_macro_f1']:.6f}",
                    "test_auc": f"{row['test_auc']:.6f}",
                    "model_size_mb": f"{row['model_size_mb']:.2f}",
                    "inference_latency_ms_per_image": f"{row['inference_latency_ms_per_image']:.3f}",
                }
            )

    json_path = RESULTS_DIR / "bloodmnist_architecture_benchmark.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path


def save_per_class_metrics(report: dict) -> tuple[Path, Path]:
    rows = []
    for class_index in range(NUM_CLASSES):
        class_metrics = report[str(class_index)]
        rows.append(
            {
                "class_index": class_index,
                "class_name": LABEL_MAP[class_index],
                "precision": class_metrics["precision"],
                "recall": class_metrics["recall"],
                "f1_score": class_metrics["f1-score"],
                "support": int(class_metrics["support"]),
            }
        )

    csv_path = RESULTS_DIR / "bloodmnist_best_model_per_class_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_index", "class_name", "precision", "recall", "f1_score", "support"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "f1_score": f"{row['f1_score']:.6f}",
                }
            )

    ranked_rows = sorted(rows, key=lambda item: item["f1_score"], reverse=True)
    summary = {
        "ranking_metric": "per-class F1 score",
        "best_class": ranked_rows[0],
        "worst_class": ranked_rows[-1],
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "overall_accuracy": report["accuracy"],
    }
    json_path = RESULTS_DIR / "bloodmnist_best_model_class_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path


def save_confusion_matrix_figure(y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("BloodMNIST Best Model Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels([LABEL_MAP[idx] for idx in range(NUM_CLASSES)], rotation=45, ha="right")
    ax.set_yticklabels([LABEL_MAP[idx] for idx in range(NUM_CLASSES)])

    for row_index in range(NUM_CLASSES):
        for col_index in range(NUM_CLASSES):
            value = cm[row_index, col_index]
            text_color = "white" if value > cm.max() * 0.5 else "black"
            ax.text(col_index, row_index, str(value), ha="center", va="center", color=text_color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    figure_path = FIGURES_DIR / "bloodmnist_best_model_confusion_matrix.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def save_misclassified_examples(raw_dataset: BloodMNIST, evaluation: dict, top_k: int = 16) -> tuple[Path, Path]:
    wrong_indices = np.flatnonzero(evaluation["wrong_mask"])
    sorted_indices = wrong_indices[np.argsort(evaluation["wrong_confidence"][wrong_indices])[::-1]]
    selected_indices = sorted_indices[:top_k]

    records = []
    figure, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.flatten()

    for axis, sample_index in zip(axes, selected_indices):
        image = raw_dataset.imgs[sample_index]
        true_label = int(evaluation["y_true"][sample_index])
        pred_label = int(evaluation["y_pred"][sample_index])
        pred_confidence = float(evaluation["y_prob"][sample_index, pred_label])

        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"Idx {sample_index}\nTrue: {LABEL_MAP[true_label]}\nPred: {LABEL_MAP[pred_label]} ({pred_confidence:.2f})",
            fontsize=8,
        )
        records.append(
            {
                "sample_index": int(sample_index),
                "true_label_index": true_label,
                "true_label_name": LABEL_MAP[true_label],
                "predicted_label_index": pred_label,
                "predicted_label_name": LABEL_MAP[pred_label],
                "predicted_confidence": pred_confidence,
            }
        )

    for axis in axes[len(selected_indices):]:
        axis.axis("off")

    figure.suptitle("Representative Misclassified BloodMNIST Test Examples", fontsize=14)
    figure.tight_layout()
    figure_path = FIGURES_DIR / "bloodmnist_best_model_misclassified_examples.png"
    figure.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    csv_path = RESULTS_DIR / "bloodmnist_best_model_misclassified_examples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "true_label_index",
                "true_label_name",
                "predicted_label_index",
                "predicted_label_name",
                "predicted_confidence",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    **record,
                    "predicted_confidence": f"{record['predicted_confidence']:.6f}",
                }
            )

    return figure_path, csv_path


def save_best_model_analysis(raw_dataset: BloodMNIST, loader: DataLoader) -> Path:
    model = build_model(BEST_MODEL_CONFIG["timm_name"])
    load_checkpoint(model, BEST_MODEL_CONFIG["checkpoint_path"])
    evaluation = evaluate_model(model, loader)
    report = classification_report(
        evaluation["y_true"],
        evaluation["y_pred"],
        labels=list(range(NUM_CLASSES)),
        output_dict=True,
        zero_division=0,
    )

    save_per_class_metrics(report)
    save_confusion_matrix_figure(evaluation["y_true"], evaluation["y_pred"])
    save_misclassified_examples(raw_dataset, evaluation, top_k=16)

    summary = {
        "model_key": BEST_MODEL_CONFIG["model_key"],
        "strategy_name": BEST_MODEL_CONFIG["strategy_name"],
        "checkpoint_path": str(BEST_MODEL_CONFIG["checkpoint_path"].relative_to(PROJECT_ROOT)),
        "device": DEVICE,
        "test_loss": evaluation["loss"],
        "test_accuracy": evaluation["accuracy"],
        "test_macro_f1": evaluation["macro_f1"],
        "test_auc": evaluation["auc"],
        "num_misclassified": int(evaluation["wrong_mask"].sum()),
    }
    json_path = RESULTS_DIR / "bloodmnist_best_model_analysis_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return json_path


def save_hyperparameter_table() -> Path:
    rows = [
        {
            "experiment_group": "architecture_comparison",
            "experiment_name": "resnet50_full_fine_tuning",
            "model_key": "resnet50",
            "pretrained": "yes",
            "train_fraction": "100%",
            "seed_or_seeds": "42",
            "optimizer": "AdamW",
            "learning_rate": "1e-4",
            "lr_schedule": "none",
            "batch_size": "32",
            "num_epochs": "12",
            "early_stopping_patience": "3",
            "weight_decay": "1e-4",
            "image_size": "224",
            "fine_tuning_strategy": "full_fine_tuning",
            "augmentation_details": "Resize to 224x224; ImageNet normalization; no extra augmentation",
        },
        {
            "experiment_group": "architecture_comparison",
            "experiment_name": "swin_tiny_full_fine_tuning",
            "model_key": "swin_tiny",
            "pretrained": "yes",
            "train_fraction": "100%",
            "seed_or_seeds": "42",
            "optimizer": "AdamW",
            "learning_rate": "1e-4",
            "lr_schedule": "none",
            "batch_size": "32",
            "num_epochs": "12",
            "early_stopping_patience": "3",
            "weight_decay": "1e-4",
            "image_size": "224",
            "fine_tuning_strategy": "full_fine_tuning",
            "augmentation_details": "Resize to 224x224; ImageNet normalization; no extra augmentation",
        },
        {
            "experiment_group": "architecture_comparison",
            "experiment_name": "vit_b16_full_fine_tuning",
            "model_key": "vit_b16",
            "pretrained": "yes",
            "train_fraction": "100%",
            "seed_or_seeds": "42",
            "optimizer": "AdamW",
            "learning_rate": "1e-4",
            "lr_schedule": "none",
            "batch_size": "32",
            "num_epochs": "12",
            "early_stopping_patience": "3",
            "weight_decay": "1e-4",
            "image_size": "224",
            "fine_tuning_strategy": "full_fine_tuning",
            "augmentation_details": "Resize to 224x224; ImageNet normalization; no extra augmentation",
        },
        {
            "experiment_group": "strategy_comparison",
            "experiment_name": "swin_tiny_feature_extraction",
            "model_key": "swin_tiny",
            "pretrained": "yes",
            "train_fraction": "100%",
            "seed_or_seeds": "42",
            "optimizer": "AdamW",
            "learning_rate": "1e-4",
            "lr_schedule": "none",
            "batch_size": "32",
            "num_epochs": "12",
            "early_stopping_patience": "3",
            "weight_decay": "1e-4",
            "image_size": "224",
            "fine_tuning_strategy": "feature_extraction",
            "augmentation_details": "Resize to 224x224; ImageNet normalization; no extra augmentation",
        },
        {
            "experiment_group": "strategy_comparison",
            "experiment_name": "swin_tiny_gradual_unfreezing",
            "model_key": "swin_tiny",
            "pretrained": "yes",
            "train_fraction": "100%",
            "seed_or_seeds": "42",
            "optimizer": "AdamW",
            "learning_rate": "1e-4",
            "lr_schedule": "none",
            "batch_size": "32",
            "num_epochs": "12",
            "early_stopping_patience": "3",
            "weight_decay": "1e-4",
            "image_size": "224",
            "fine_tuning_strategy": "gradual_unfreezing",
            "augmentation_details": "Resize to 224x224; ImageNet normalization; no extra augmentation",
        },
        {
            "experiment_group": "strategy_comparison",
            "experiment_name": "swin_tiny_discriminative_learning_rates",
            "model_key": "swin_tiny",
            "pretrained": "yes",
            "train_fraction": "100%",
            "seed_or_seeds": "42",
            "optimizer": "AdamW",
            "learning_rate": "patch/layer groups = [2.5e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 1e-4, 2e-4]",
            "lr_schedule": "none",
            "batch_size": "32",
            "num_epochs": "12",
            "early_stopping_patience": "3",
            "weight_decay": "1e-4",
            "image_size": "224",
            "fine_tuning_strategy": "discriminative_learning_rates",
            "augmentation_details": "Resize to 224x224; ImageNet normalization; no extra augmentation",
        },
    ]

    for fraction in ["100%", "50%", "25%", "10%", "5%"]:
        rows.append(
            {
                "experiment_group": "data_efficiency",
                "experiment_name": f"swin_tiny_pretrained_{fraction}",
                "model_key": "swin_tiny",
                "pretrained": "yes",
                "train_fraction": fraction,
                "seed_or_seeds": "42,123,3185",
                "optimizer": "AdamW",
                "learning_rate": "1e-4",
                "lr_schedule": "none",
                "batch_size": "32",
                "num_epochs": "12",
                "early_stopping_patience": "3",
                "weight_decay": "1e-4",
                "image_size": "224",
                "fine_tuning_strategy": "full_fine_tuning",
                "augmentation_details": "Resize to 224x224; ImageNet normalization; no extra augmentation",
            }
        )
        rows.append(
            {
                "experiment_group": "data_efficiency",
                "experiment_name": f"swin_tiny_scratch_{fraction}",
                "model_key": "swin_tiny",
                "pretrained": "no",
                "train_fraction": fraction,
                "seed_or_seeds": "42,123,3185",
                "optimizer": "AdamW",
                "learning_rate": "5e-4",
                "lr_schedule": "none",
                "batch_size": "32",
                "num_epochs": "30",
                "early_stopping_patience": "5",
                "weight_decay": "1e-4",
                "image_size": "224",
                "fine_tuning_strategy": "full_fine_tuning",
                "augmentation_details": "Resize to 224x224; ImageNet normalization; no extra augmentation",
            }
        )

    csv_path = RESULTS_DIR / "hyperparameter_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_group",
                "experiment_name",
                "model_key",
                "pretrained",
                "train_fraction",
                "seed_or_seeds",
                "optimizer",
                "learning_rate",
                "lr_schedule",
                "batch_size",
                "num_epochs",
                "early_stopping_patience",
                "weight_decay",
                "image_size",
                "fine_tuning_strategy",
                "augmentation_details",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    FIGURES_DIR.mkdir(exist_ok=True, parents=True)

    raw_test_dataset, eval_test_dataset = build_test_datasets()
    test_loader = make_test_loader(eval_test_dataset)

    benchmark_csv = save_architecture_benchmark(test_loader)
    best_model_json = save_best_model_analysis(raw_test_dataset, test_loader)
    hyperparams_csv = save_hyperparameter_table()

    print("Saved artifacts:")
    print(f"- {benchmark_csv}")
    print(f"- {best_model_json}")
    print(f"- {hyperparams_csv}")
    print(f"- {RESULTS_DIR / 'bloodmnist_best_model_per_class_metrics.csv'}")
    print(f"- {RESULTS_DIR / 'bloodmnist_best_model_class_summary.json'}")
    print(f"- {RESULTS_DIR / 'bloodmnist_best_model_misclassified_examples.csv'}")
    print(f"- {FIGURES_DIR / 'bloodmnist_best_model_confusion_matrix.png'}")
    print(f"- {FIGURES_DIR / 'bloodmnist_best_model_misclassified_examples.png'}")


if __name__ == "__main__":
    main()
