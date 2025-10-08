import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional, Any

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def _find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    timm CNN 백본에서 Grad-CAM 타깃 레이어 후보를 자동 선택.
    """
    # EfficientNet 계열: conv_head 또는 blocks의 마지막 Conv
    if hasattr(model, "conv_head"):
        return model.conv_head
    if hasattr(model, "blocks"):
        # timm EfficientNet 계열은 blocks[-1].conv_pw 또는 conv_dw가 말단인 경우 多
        try:
            last = model.blocks[-1]
            if hasattr(last, "conv_pw") and isinstance(last.conv_pw, torch.nn.Conv2d):
                return last.conv_pw
            if hasattr(last, "conv_dw") and isinstance(last.conv_dw, torch.nn.Conv2d):
                return last.conv_dw
        except Exception:
            pass

    # ResNet 계열
    if hasattr(model, "layer4"):
        try:
            return model.layer4[-1]
        except Exception:
            return model.layer4

    # DenseNet, MobileNet 등 features 시퀀스
    if hasattr(model, "features"):
        for module in reversed(list(model.features.modules())):
            if isinstance(module, torch.nn.Conv2d):
                return module

    # 최후 보루: 모든 모듈 뒤에서 Conv2d 찾기
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m

    raise ValueError("Grad-CAM target layer를 자동으로 찾지 못했습니다. target_layer를 수동 지정해 주세요.")


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

def _iter_batches(dataloader, x_key: str = "pixel_values", y_key: str = "labels"):
    """dict/tuple 배치를 표준 (inputs, labels)로 통일"""
    for batch in dataloader:
        if isinstance(batch, dict):
            yield batch[x_key], batch[y_key], batch
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            # dict 부가정보 없음
            yield batch[0], batch[1], None
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

def _pil_from_batch(batch_dict, index: int) -> Optional[Image.Image]:
    """
    transformer가 원본 PIL을 batch에 같이 담아줬다면 사용.
    없다면 None 반환(그림 합성 없이 텐서만 사용하게 fallback).
    """
    if batch_dict is None:
        return None
    # 자주 쓰는 키 후보
    for k in ("pil_images", "pil", "originals", "images"):
        if k in batch_dict:
            try:
                return batch_dict[k][index]
            except Exception:
                pass
    return None

def _to_rgb01_from_tensor(x: torch.Tensor) -> np.ndarray:
    """
    텐서(1,C,H,W) 또는 (C,H,W)를 0~1 float32 RGB ndarray로 변환.
    """
    if x.dim() == 4:
        x = x[0]
    x = x.detach().cpu().float().clamp(0, 1)
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    arr = x.permute(1,2,0).numpy()
    return arr

def generate_grad_cam_from_loader(
    model: torch.nn.Module,
    dataloader,
    class_names: List[str],
    save_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    true_per_class: int = 1,
    mis_per_class: int = 1,
    x_key: str = "pixel_values",
    y_key: str = "labels",
    target_layer: torch.nn.Module = None,
) -> Dict[str, int]:
    """
    테스트 로더를 순회하며 클래스별로
      - 올바른 예측 true_per_class 장
      - 오분류 mis_per_class 장
    Grad-CAM 이미지를 저장한다.
    Returns: 저장한 개수 카운트 {"true": n1, "mis": n2}
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.eval(); model.to(device)

    if target_layer is None:
        target_layer = _find_target_layer(model)
        print(f"[GradCAM] Auto-selected target layer: {target_layer.__class__.__name__}")

    cam = GradCAM(model=model, target_layers=[target_layer])

    n_classes = len(class_names)
    need_true = {c: true_per_class for c in range(n_classes)}
    need_mis  = {c: mis_per_class  for c in range(n_classes)}
    saved_true = 0
    saved_mis  = 0

    for inputs, labels, raw in _iter_batches(dataloader, x_key=x_key, y_key=y_key):
        inputs = inputs.to(device)
        labels = labels.to(device) if torch.is_tensor(labels) else torch.tensor(labels, device=device)

        with torch.no_grad():
            logits = model(inputs)
            probas = torch.softmax(logits, dim=1)
            preds  = probas.argmax(dim=1)

        B = inputs.size(0)
        for i in range(B):
            y = int(labels[i].item())
            p = int(preds[i].item())
            # 저장 완료된 클래스는 skip
            need_true_done = all(v <= 0 for v in need_true.values())
            need_mis_done  = all(v <= 0 for v in need_mis.values())
            if need_true_done and need_mis_done:
                break

            # 이 샘플을 저장해야 하는가?
            mode = None
            if y == p and need_true[y] > 0:
                mode = "true"
            elif y != p and need_mis[y] > 0:
                mode = "mis"

            if mode is None:
                continue

            # CAM 대상 1개짜리 텐서로 구성
            x1 = inputs[i:i+1]

            # 원본 이미지 준비 (없으면 텐서 복원)
            pil = _pil_from_batch(raw, i)
            if pil is None:
                rgb01 = _to_rgb01_from_tensor(x1)
                pil = Image.fromarray((rgb01 * 255).astype(np.uint8))

            # 타깃 클래스는 예측 p로 둔다(일반적 사용)
            targets = [ClassifierOutputTarget(p)]
            grayscale_cam = cam(input_tensor=x1, targets=targets)[0, :]

            rgb_img = np.array(pil.convert("RGB")).astype(np.float32) / 255.0
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # 라벨 텍스트
            final_image = Image.fromarray(visualization)
            cv_im = cv2.cvtColor(np.array(final_image), cv2.COLOR_RGB2BGR)
            cv2.putText(cv_im, f"True:{class_names[y]}", (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(cv_im, f"Pred:{class_names[p]}", (10,46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            final_image = Image.fromarray(cv2.cvtColor(cv_im, cv2.COLOR_BGR2RGB))

            # 저장
            sub = "correct" if mode == "true" else "miscls"
            fname = f"{sub}_y{y}-{class_names[y]}_p{p}-{class_names[p]}_{saved_true+saved_mis:05d}.png"
            out_path = Path(save_dir) / fname
            final_image.save(out_path)

            if mode == "true":
                need_true[y] -= 1
                saved_true += 1
            else:
                need_mis[y] -= 1
                saved_mis += 1

        # 전체 충족되었으면 조기 종료
        if all(v <= 0 for v in need_true.values()) and all(v <= 0 for v in need_mis.values()):
            break

    print(f"[GradCAM] saved true={saved_true}, mis={saved_mis} -> {save_dir}")
    return {"true": saved_true, "mis": saved_mis}