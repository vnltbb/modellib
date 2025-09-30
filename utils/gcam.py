import torch
import numpy as np
import cv2
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image
from pathlib import Path
from typing import List

def _find_target_layer(model):
    """timm 모델에서 Grad-CAM을 위한 마지막 conv layer를 찾는 헬퍼 함수"""
    # 일반적인 timm 모델 구조를 기반으로 마지막 conv layer를 탐색
    if hasattr(model, 'conv_head'):
        return model.conv_head
    if hasattr(model, 'features') and hasattr(model.features, 'conv'):
        return model.features.conv
    if hasattr(model, 'layer4'): # ResNet 계열
        return model.layer4[-1]
    if hasattr(model, 'features'): # DenseNet, MobileNet 계열
        # features의 마지막 모듈이 Conv2d, BatchNorm2d를 포함하는 블록인 경우가 많음
        for module in reversed(list(model.features)):
            if isinstance(module, torch.nn.Conv2d):
                return module
            if hasattr(module, 'conv'): # MobileNetV2 InvertedResidual
                 for sub_module in reversed(list(module.conv)):
                     if isinstance(sub_module, torch.nn.Conv2d):
                         return sub_module
    
    # 찾지 못한 경우 예외 발생
    raise ValueError("Could not find a suitable target layer for Grad-CAM. Please specify it manually.")

def generate_grad_cam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    save_dir: str,
    filename: str,
    target_layer: torch.nn.Module = None,
    true_label: str = None,
    pred_label: str = None
):
    """
    주어진 이미지에 대한 Grad-CAM을 생성하고 원본 이미지와 겹쳐 저장합니다.

    Args:
        model (torch.nn.Module): CAM을 생성할 PyTorch 모델.
        input_tensor (torch.Tensor): 모델에 입력될 텐서 (1, C, H, W).
        original_image (Image.Image): 겹쳐서 표시할 원본 PIL 이미지.
        save_dir (str): 결과 이미지를 저장할 디렉토리.
        filename (str): 저장될 파일의 이름 (확장자 제외).
        target_layer (torch.nn.Module, optional): CAM을 추출할 레이어. None이면 자동 탐색.
        true_label (str, optional): 이미지에 표시할 실제 라벨.
        pred_label (str, optional): 이미지에 표시할 예측 라벨.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if target_layer is None:
        target_layer = _find_target_layer(model)
        print(f"Automatically selected target layer: {target_layer.__class__.__name__}")
        
    # 1. Grad-CAM 객체 생성
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # 2. CAM 생성
    # 예측 확률이 가장 높은 클래스를 타겟으로 설정
    targets = [ClassifierOutputTarget(torch.argmax(model(input_tensor)))]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] # 배치 차원 제거

    # 3. 원본 이미지와 CAM 오버레이
    # PIL 이미지를 0-1 범위의 float32 numpy 배열로 변환
    rgb_img = np.array(original_image.convert('RGB')).astype(np.float32) / 255
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 4. 이미지 저장
    # 라벨 정보를 포함한 최종 이미지 생성
    final_image = Image.fromarray(visualization)
    if true_label is not None and pred_label is not None:
        title = f"True: {true_label}\nPred: {pred_label}"
        # 간단하게 OpenCV를 사용하여 텍스트 추가
        cv_image = cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR)
        cv2.putText(cv_image, f"True: {true_label}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(cv_image, f"Pred: {pred_label}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        final_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    
    save_path = Path(save_dir) / f"{filename}_gradcam.png"
    final_image.save(save_path)
    print(f"Grad-CAM image saved to '{save_path}'")