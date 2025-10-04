import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

def save_history_plot(
    history: Dict[str, List[float]],
    save_dir: str,
    model_name: str,
    # --- New Arguments ---
    fixed_scales: bool = False,
    x_max: int = 50,
    y_loss_max: Optional[float] = None,
    y_metric_max: float = 1.0
):
    """
    훈련 과정의 history를 받아 Loss와 Metric 그래프를 파일로 저장합니다.

    Args:
        history (Dict[str, List[float]]): 학습 기록 딕셔너리.
        save_dir (str): 그래프 이미지를 저장할 디렉토리.
        model_name (str): 저장할 파일의 이름.
        fixed_scales (bool, optional): 축 스케일을 고정할지 여부. Defaults to False.
        x_max (int, optional): fixed_scales=True일 때 x축 최대값. Defaults to 50.
        y_loss_max (Optional[float], optional): fixed_scales=True일 때 Loss 축 최대값. 
                                                None이면 자동으로 계산. Defaults to None.
        y_metric_max (float, optional): fixed_scales=True일 때 Metric 축 최대값. Defaults to 1.0.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(save_dir, f"{model_name}_history.png")
    print(f"Saving training history plot to '{file_path}'...")

    sns.set_style("whitegrid")
    metric_keys = [k for k in history.keys() if 'loss' not in k and 'val' not in k]
    num_plots = 2 if metric_keys else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
    if num_plots == 1:
        axes = [axes] # 서브플롯이 하나일 때도 리스트로 만들어 일관성 유지

    ax1 = axes[0]
    epochs = range(1, len(history['train_loss']) + 1)

    # --- 1. Loss 그래프 ---
    ax1.plot(epochs, history['train_loss'], 'o-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'o-', label='Validation Loss')
    ax1.set_title(f'{model_name} - Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # --- 2. Metric 그래프 (있는 경우) ---
    if num_plots == 2:
        ax2 = axes[1]
        metric_name = metric_keys[0]
        ax2.plot(epochs, history[metric_name], 'o-', label=f'Train {metric_name.replace("_", " ").title()}')
        ax2.plot(epochs, history[f'val_{metric_name}'], 'o-', label=f'Validation {metric_name.replace("_", " ").title()}')
        ax2.set_title(f'{model_name} - Model {metric_name.replace("_", " ").title()}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel(metric_name.replace("_", " ").title())
        ax2.legend()
        ax2.grid(True)
    
    # --- 3. 축 스케일 고정 로직 ---
    if fixed_scales:
        # Loss 축 (ax1)
        ax1.set_xlim(0, x_max)
        if y_loss_max is None:
            # y_loss_max가 지정되지 않으면, 실제 데이터 최대값의 110%로 자동 설정
            max_loss_val = max(max(history['train_loss']), max(history['val_loss']))
            ax1.set_ylim(0, max_loss_val * 1.1)
        else:
            ax1.set_ylim(0, y_loss_max)
        
        # Metric 축 (ax2)
        if num_plots == 2:
            ax2.set_xlim(0, x_max)
            ax2.set_ylim(0, y_metric_max)

    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

    print("Successfully saved history plot.")