import torch
import torch.nn as nn
import timm
from typing import List

# 지원하는 모델 목록을 명시적으로 관리합니다.
SUPPORTED_BACKBONES = {
    'resnet50': 'resnet50',
    'mobilenetv2': 'mobilenet_v2',
    'efficientnetv2l': 'efficientnetv2_l',
    'densenet121': 'densenet121'
}

def get_supported_backbones() -> List[str]:
    """지원하는 백본 모델 이름 목록을 반환합니다."""
    return list(SUPPORTED_BACKBONES.keys())

def build(backbone: str, num_classes: int, transfer: bool = True, dropout_rate: float = 0.0):
    """
    지정된 백본을 기반으로 PyTorch 모델을 빌드합니다.

    Args:
        backbone (str): 사용할 백본 모델의 이름.
                        (e.g., 'resnet50', 'mobilenetv2', 'efficientnetb0', 'densenet121')
        num_classes (int): 최종 출력 레이어의 클래스 수. (transfer=False 이면 무시됨)
        transfer (bool, optional): 
            - True (기본값): 전이 학습용 모델을 빌드합니다. ImageNet으로 사전 학습된
              가중치를 불러온 뒤, 최종 분류기(classifier)만 주어진 `num_classes`에 맞게
              새로 초기화하여 교체합니다.
            - False: 특징 추출기(feature extractor)로 사용하기 위한 모델을 빌드합니다.
              사전 학습된 가중치를 불러오고, 최종 분류기는 제거된 상태의 모델을 반환합니다.
        dropout_rate (float, optional): 최종 분류기 직전에 적용할 드롭아웃 비율. 
                                        transfer=True일 때만 유효합니다. Defaults to 0.0.

    Returns:
        torch.nn.Module: 생성된 PyTorch 모델 객체.
        
    Raises:
        ValueError: 지원하지 않는 백본 이름이 입력된 경우.
    """
    backbone_key = backbone.lower().replace('-', '').replace('_', '')
    
    if backbone_key not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone: {backbone}. "
            f"Supported backbones are: {get_supported_backbones()}"
        )
        
    timm_model_name = SUPPORTED_BACKBONES[backbone_key]
    
    if transfer:
        print(f"Building model '{timm_model_name}' for **transfer learning**.")
        print(f"The final classifier will be replaced to have {num_classes} outputs.")
        
        # timm.create_model은 pretrained=True, num_classes=N 으로 설정 시
        # 자동으로 전이학습에 맞게 마지막 레이어를 교체해줍니다. 이것이 바로 우리가 원하는 기능입니다.
        model = timm.create_model(
            timm_model_name,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=dropout_rate
        )
    else:
        print(f"Building model '{timm_model_name}' as a **feature extractor**.")
        print("The final classifier is being removed.")
        
        # num_classes=0 으로 설정하면, timm은 분류기를 제거하고 특징 추출 부분만 반환합니다.
        model = timm.create_model(
            timm_model_name,
            pretrained=True,
            num_classes=0  
        )
    
    print("Model built successfully.")
    return model