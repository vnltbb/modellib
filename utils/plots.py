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
from typing import List

def _get_predictions(model, dataloader, device):
    """모델과 데이터로더를 받아 예측 확률과 실제 라벨을 반환하는 헬퍼 함수"""
    model.eval()
    all_labels = []
    all_probas = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probas = torch.nn.functional.softmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
    return np.array(all_labels), np.array(all_probas)

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