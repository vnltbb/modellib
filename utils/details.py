import torch
import torchinfo
import os
from pathlib import Path
from datetime import datetime

def save_model_details(
    model: torch.nn.Module,
    save_dir: str,
    model_name: str,
    input_size: tuple
):
    """
    모델의 상세 정보(요약, 파라미터 수, 용량)를 텍스트 파일로 저장합니다.

    Args:
        model (torch.nn.Module): 정보를 저장할 PyTorch 모델.
        save_dir (str): 상세 정보 파일을 저장할 디렉토리 경로.
        model_name (str): 저장할 파일의 이름 (e.g., 'EfficientNetB0_best').
        input_size (tuple): 모델의 요약을 위한 입력 데이터 크기 (e.g., (1, 3, 224, 224)).
    """
    # 저장 디렉토리 생성
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    file_path = os.path.join(save_dir, f"{model_name}_details.txt")
    
    print(f"Saving model details to '{file_path}'...")

    # 1. 모델 가중치의 파일 크기 (용량) 계산
    # 임시로 state_dict를 저장하여 파일 크기를 확인하고 삭제
    temp_path = Path(save_dir) / "temp_weights.pth"
    torch.save(model.state_dict(), temp_path)
    model_size_mb = temp_path.stat().st_size / (1024 * 1024)
    temp_path.unlink() # 임시 파일 삭제

    # 2. torchinfo를 사용하여 모델 요약 정보 생성
    # col_names로 표시할 정보 선택: 레이어 이름, 출력 형태, 파라미터 수, 연산량(MACs)
    model_summary = torchinfo.summary(
        model,
        input_size=input_size,
        col_names=["kernel_size", "output_size", "num_params", "mult_adds"],
        verbose=0  # 화면에 출력하지 않음
    )

    # 3. 파일에 정보 작성
    with open(file_path, 'w') as f:
        f.write(f"Model Details for: {model_name}\n")
        f.write(f"Saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total Parameters: {model_summary.total_params:,}\n")
        f.write(f"Trainable Parameters: {model_summary.trainable_params:,}\n")
        f.write(f"Non-trainable Parameters: {len(model.state_dict()) - model_summary.trainable_params:,}\n")
        f.write(f"Estimated Model Size (MB): {model_size_mb:.2f} MB\n")
        f.write(f"Total MACs (G): {model_summary.total_mult_adds / 1e9:.2f} G\n\n")
        
        f.write("="*80 + "\n")
        f.write("Model Architecture Summary:\n")
        f.write("="*80 + "\n")
        f.write(str(model_summary))

    print("Successfully saved model details.")