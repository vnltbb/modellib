import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

def _iter_batches(dataloader, x_key: str = "pixel_values", y_key: str = "labels"):
    """
    dataloader에서 배치를 표준 (inputs, labels) 튜플로 통일.
    - dict 배치: batch[x_key], batch[y_key]
    - 튜플/리스트 배치: (inputs, labels)
    """
    for batch in dataloader:
        if isinstance(batch, dict):
            if x_key not in batch or y_key not in batch:
                raise KeyError(f"Dict batch must contain keys '{x_key}' and '{y_key}'. "
                               f"Got keys: {list(batch.keys())}")
            yield batch[x_key], batch[y_key]
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            yield batch[0], batch[1]
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}. "
                            "Expected dict with pixel/label keys or (inputs, labels).")

def _get_predictions(model, dataloader, device, x_key: str = "pixel_values", y_key: str = "labels"):
    """
    모델의 확률 예측을 반환.
    Returns:
        y_true: (N,) int
        y_pred_proba: (N, C) float
    """
    model.eval()
    y_true_list = []
    y_proba_list = []
    with torch.no_grad():
        for inputs, labels in _iter_batches(dataloader, x_key=x_key, y_key=y_key):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probas = torch.nn.functional.softmax(outputs, dim=1)
            y_true_list.append(labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels))
            y_proba_list.append(probas.cpu().numpy())
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred_proba = np.concatenate(y_proba_list, axis=0)
    return y_true, y_pred_proba

def save_classification_results(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: List[str],
    save_dir: str,
    model_name: str,
    device: str = 'cuda'
):
    """
    모델의 평가 결과를 종합하여 Confusion Matrix, Classification Report,
    ROC Curve, PR Curve를 생성하고 파일로 저장합니다.

    Args:
        model (torch.nn.Module): 평가할 PyTorch 모델.
        dataloader (torch.utils.data.DataLoader): 테스트 데이터로더.
        class_names (List[str]): 클래스 이름 목록.
        save_dir (str): 결과물을 저장할 디렉토리 경로.
        model_name (str): 저장할 파일의 이름.
        device (str): 모델을 실행할 디바이스 ('cuda' or 'cpu').
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    y_true, y_pred_proba = _get_predictions(model, dataloader, device)
    y_pred_class = np.argmax(y_pred_proba, axis=1)
    num_classes = len(class_names)

    # 1. Classification Report 저장
    report = classification_report(y_true, y_pred_class, target_names=class_names, zero_division=0)
    report_path = Path(save_dir) / f"{model_name}_classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to '{report_path}'")

    # 2. Confusion Matrix 시각화 및 저장
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    cm_path = Path(save_dir) / f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to '{cm_path}'")
    
    # One-vs-Rest를 위한 이진화
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    # 3. ROC Curve (One-vs-Rest)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of {class_names[i]} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - Multi-class ROC Curve')
    plt.legend(loc="lower right")
    roc_path = Path(save_dir) / f"{model_name}_roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to '{roc_path}'")

    # 4. Precision-Recall Curve (One-vs-Rest)
    precision, recall, avg_precision = dict(), dict(), dict()
    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        avg_precision[i] = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])

    plt.figure(figsize=(10, 8))
    for i, color in zip(range(num_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'PR curve of {class_names[i]} (AP = {avg_precision[i]:0.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Multi-class Precision-Recall Curve')
    plt.legend(loc="best")
    prc_path = Path(save_dir) / f"{model_name}_pr_curve.png"
    plt.savefig(prc_path, dpi=300)
    plt.close()
    print(f"Precision-Recall curve saved to '{prc_path}'")