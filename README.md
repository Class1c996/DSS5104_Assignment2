# Transfer Learning on BloodMNIST

This repository contains my DSS5104 Assignment 2 project on transfer learning for medical image classification using the `BloodMNIST` dataset.

The project includes three main experiment blocks:
- architecture comparison across `ResNet-50`, `Swin-Tiny`, and `ViT-B/16`
- fine-tuning strategy comparison on `Swin-Tiny`
- data-efficiency comparison between pretrained and from-scratch `Swin-Tiny` using `100%`, `50%`, `25%`, `10%`, and `5%` of the training set with 3 random seeds

## Repository Structure

- `BloodMNIST_resnet_swin_vit.ipynb`: notebook for pretrained architecture comparison among ResNet-50, Swin-Tiny, and ViT-B/16 on BloodMNIST
- `Swin_strategy_finetuning.ipynb`: notebook interface for launching the Swin-Tiny fine-tuning strategy comparison; internally invokes `run_swin_strategies.py`
- `Swin_pretrained_vs_scratch.ipynb`: notebook interface for launching the pretrained-versus-scratch data-efficiency experiment; internally invokes `run_swin_pretrained_vs_scratch.py`
- `run_swin_strategies.py`: underlying command-line script used by `Swin_strategy_finetuning.ipynb` to execute the Swin-Tiny fine-tuning strategy comparison
- `run_swin_pretrained_vs_scratch.py`: underlying command-line script used by `Swin_pretrained_vs_scratch.ipynb` to execute the pretrained-versus-scratch data-efficiency experiment
- `generate_report_artifacts.py`: generates report-ready tables and figures
- `results/`: CSV and JSON result summaries
- `figures/`: generated figures used in the report
- `checkpoints/`: saved model checkpoints

## Environment Setup

Create a fresh virtual environment locally.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you use Conda or another environment manager, install the packages listed in `requirements.txt`.

## How To Run

### 1. Architecture comparison

Open `BloodMNIST_resnet_swin_vit.ipynb` in Jupyter or VS Code and run all cells.

### 2. Fine-tuning strategy comparison

```powershell
python .\run_swin_strategies.py --strategies feature_extraction gradual_unfreezing discriminative_learning_rates
```

### 3. Data-efficiency experiment

```powershell
python .\run_swin_pretrained_vs_scratch.py --seeds 42 123 3185
```

### 4. Generate report-ready artifacts

```powershell
python .\generate_report_artifacts.py
```

## Main Outputs

- `results/swin_tiny_strategy_summary.csv`: summary table for the three fine-tuning strategies, including best epoch, validation AUC, test accuracy, macro F1, AUC, runtime, and model size
- `results/swin_tiny_pretrained_vs_scratch_summary.csv`: aggregated data-efficiency results for pretrained vs scratch Swin-Tiny across all training fractions, reported as mean and standard deviation over 3 seeds
- `results/bloodmnist_architecture_benchmark.csv`: architecture comparison table for ResNet-50, Swin-Tiny, and ViT-B/16, including test metrics, parameter count, model size, and inference latency
- `results/hyperparameter_table.csv`: consolidated experiment settings, including optimizer, learning rate, batch size, epochs, patience, weight decay, image size, and fine-tuning strategy
- `results/bloodmnist_best_model_per_class_metrics.csv`: per-class precision, recall, F1 score, and support for the selected best model on the test set
- `results/bloodmnist_best_model_class_summary.json`: summary of the best-performing and worst-performing classes for the best model, together with macro and weighted averages
- `results/bloodmnist_best_model_misclassified_examples.csv`: list of representative misclassified test samples, including true label, predicted label, and prediction confidence
- `figures/bloodmnist_architecture_comparison_training_curves.png`: training and validation curves for the three pretrained architectures
- `figures/bloodmnist_architecture_comparison_test_metrics.png`: bar chart comparing test accuracy, macro F1, and AUC across the three pretrained architectures
- `figures/bloodmnist_swin_tiny_strategy_training_curves.png`: training and validation curves for the three Swin-Tiny fine-tuning strategies
- `figures/bloodmnist_swin_tiny_strategy_test_metrics.png`: bar chart comparing test accuracy, macro F1, and AUC across the three fine-tuning strategies
- `figures/bloodmnist_swin_pretrained_vs_scratch_accuracy_auc.png`: plot of pretrained vs scratch performance across training fractions using accuracy and AUC
- `figures/bloodmnist_swin_pretrained_vs_scratch_macro_f1.png`: plot of pretrained vs scratch performance across training fractions using macro F1
- `figures/bloodmnist_best_model_confusion_matrix.png`: confusion matrix for the final best model on the held-out test set
- `figures/bloodmnist_best_model_misclassified_examples.png`: grid of representative misclassified BloodMNIST examples used for qualitative error analysis

## Notes

- Training-set subsampling in the Python scripts uses stratified sampling to preserve class proportions.
- Validation and test splits remain fixed; only the training split is subsampled.
- The latency benchmark is reported in `ms/image` on the device available during execution.
