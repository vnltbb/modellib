import timm
import torch.nn as nn

# ---- 내부 유틸 ----
def _set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag

def _freeze_all_batchnorm(model: nn.Module):
    # 백본 동결과 함께 BN도 고정(통계 고정 + affine 파라미터 고정)
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()
            if m.affine:
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)

def _get_head_module(model: nn.Module):
    # timm 공통: get_classifier() 제공. 없으면 classifier/fc 순서로 탐색.
    head = model.get_classifier() if hasattr(model, "get_classifier") else None
    if head is None and hasattr(model, "classifier"):
        head = model.classifier
    if head is None and hasattr(model, "fc"):
        head = model.fc
    return head

def _set_cam_target_layer(model: nn.Module):
    # Grad-CAM 타깃: 마지막 Conv2d (EffNetV2는 conv_head가 가장 일반적)
    target = getattr(model, "conv_head", None)
    if target is None:
        # conv_head 없으면 뒤에서부터 첫 Conv2d를 타깃으로
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Conv2d):
                target = m
                break
    model.cam_target_layer = target  # 외부에서 바로 사용 가능
    return target

# ---- 최종 빌더 ----
def build(num_classes: int, drop_rate: float = 0.0):
    """
    - 백본: densenet121.ra_in1k (pretrained)
    - 헤드: timm 기본형(GAP -> Dropout(drop_rate) -> Linear(num_classes))
    - 동결: freeze='all' (헤드만 학습)
    - BN: 백본 동결과 함께 고정(eval + affine no-grad)
    - Grad-CAM: model.cam_target_layer에 마지막 Conv2d 저장
    """
    model = timm.create_model(
        'densenet121.ra_in1k',
        pretrained=True,
        num_classes=num_classes,
        drop_rate=drop_rate,
        global_pool='avg'
    )

    # 1) 전체 동결
    _set_requires_grad(model, False)
    # 2) 헤드만 학습 허용
    head = _get_head_module(model)
    if head is None:
        raise RuntimeError("분류기 헤드를 찾지 못했습니다. timm 모델 구조를 확인하세요.")
    _set_requires_grad(head, True)
    # 3) BN 고정
    _freeze_all_batchnorm(model)
    # 4) Grad-CAM 타깃 지정(외부 라이브러리에서 바로 사용 가능)
    _set_cam_target_layer(model)

    return model

