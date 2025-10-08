# 코드 디자인
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

# seaborn은 선택적 사용 (없어도 동작)
try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False


def _normalize_history_keys(
    history: Dict[str, List[float]]
) -> Dict[str, List[float]]:
    """
    다양한 키 스킴(train_* / val_* vs 무접두) 을 표준 키로 맞춘다.
    표준 키: loss, val_loss, f1, val_f1, acc, val_acc
    """
    # 원본 키 집합
    keys = set(history.keys())

    def pick(*cands):
        for k in cands:
            if k in history:
                return k
        return None

    # 표준 키 사전
    out = {}

    # 1) Loss
    k_tr_loss = pick("loss", "train_loss")
    k_va_loss = pick("val_loss")
    if k_tr_loss is None or k_va_loss is None:
        # 최소한 loss 두 축은 있어야 그림을 그릴 수 있음
        raise KeyError("History must contain loss keys. Expected one of "
                       "{'loss' or 'train_loss'} AND {'val_loss'}.")

    out["loss"] = history[k_tr_loss]
    out["val_loss"] = history[k_va_loss]

    # 2) f1
    k_tr_f1 = pick("f1", "train_f1")
    k_va_f1 = pick("val_f1")
    if k_tr_f1 is not None and k_va_f1 is not None:
        out["f1"] = history[k_tr_f1]
        out["val_f1"] = history[k_va_f1]

    # 3) acc
    k_tr_acc = pick("acc", "train_acc")
    k_va_acc = pick("val_acc")
    if k_tr_acc is not None and k_va_acc is not None:
        out["acc"] = history[k_tr_acc]
        out["val_acc"] = history[k_va_acc]

    return out


def save_history_plot(
    history: Dict[str, List[float]],
    save_dir: str,
    model_name: str,
    # --- New / Existing Arguments ---
    fixed_scales: bool = False,
    x_max: Optional[int] = None,
    y_loss_max: Optional[float] = None,
    y_metric_max: float = 1.0,
    metrics: Sequence[str] | None = None,   # 예: ("f1","acc") 여러 메트릭 동시 플롯
) -> None:
    """
    학습 history를 받아 Loss + Metric 그래프를 저장.
    다양한 키 스킴(train_*, val_*)을 자동 정규화하여 사용한다.

    Args:
        history: 학습 기록 딕셔너리. (키는 유연하게 허용)
        save_dir: 저장 폴더.
        model_name: 파일명 접두어.
        fixed_scales: 축 고정 여부.
        x_max: fixed_scales=True일 때 x축 최대(미지정 시 len-기반 자동).
        y_loss_max: fixed_scales=True일 때 손실 y축 최대(None=자동 110%).
        y_metric_max: fixed_scales=True일 때 메트릭 y축 최대(기본 1.0).
        metrics: 그릴 메트릭 목록. None이면 history에 존재하는 ['f1','acc'] 중 가능한 것 자동 선택.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(save_dir, f"{model_name}_history.png")
    print(f"[history] Saving plot to '{file_path}' ...")

    # 스타일
    if _HAS_SNS:
        sns.set_style("whitegrid")

    # 1) 키 정규화
    h = _normalize_history_keys(history)

    # 2) 사용할 메트릭 결정
    available_metrics = [m for m in ("f1", "acc") if (m in h and f"val_{m}" in h)]
    if metrics is None:
        metrics = available_metrics
    else:
        # 요청된 metrics 중 사용 가능한 것만
        metrics = [m for m in metrics if (m in h and f"val_{m}" in h)]
    # 최소 Loss + (0~N개의 metric)
    num_plots = 1 + len(metrics)

    # 3) figure 생성
    fig, axes = plt.subplots(1, num_plots, figsize=(7.5 * num_plots, 5.5))
    if num_plots == 1:
        axes = [axes]

    # 에폭 축
    epochs = np.arange(1, len(h["loss"]) + 1)

    # (1) Loss
    ax = axes[0]
    ax.plot(epochs, h["loss"], "o-", label="Train Loss")
    ax.plot(epochs, h["val_loss"], "o-", label="Validation Loss")
    ax.set_title(f"{model_name} - Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True)

    # (2~) Metrics
    for i, m in enumerate(metrics, start=1):
        axm = axes[i]
        axm.plot(epochs, h[m], "o-", label=f"Train {m.upper()}")
        axm.plot(epochs, h[f"val_{m}"], "o-", label=f"Val {m.upper()}")
        axm.set_title(f"{model_name} - {m.upper()}")
        axm.set_xlabel("Epoch"); axm.set_ylabel(m.upper()); axm.legend(); axm.grid(True)

    # 4) 축 고정
    if fixed_scales:
        # x축
        if x_max is None:
            x_max = int(epochs[-1])  # 마지막 에폭
        for ax in axes:
            ax.set_xlim(0, x_max)

        # y축: loss
        if y_loss_max is None:
            max_loss_val = float(max(max(h["loss"]), max(h["val_loss"])))
            y_loss_max = max_loss_val * 1.1
        axes[0].set_ylim(0, y_loss_max)

        # y축: metrics
        for i, m in enumerate(metrics, start=1):
            axes[i].set_ylim(0, y_metric_max)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close(fig)
    print("[history] Done.")
