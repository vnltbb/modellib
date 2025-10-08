# transformer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Literal, Any
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T


# =============================================================================
# [헬퍼] 0) 백본 매핑 (4종 고정)
# =============================================================================
_BACKBONE_FAMILY_MAP = {
    "resnet":       "timm/wide_resnet50_2.tv2_in1k",
    "efficientnet": "tf_efficientnetv2_l.in1k",
    "efficientnetb0": "timm/efficientnet_b0.ra_in1k",
    "mobilenet":    "timm/mobilenetv3_large_100.ra_in1k",
    "densenet":     "timm/densenet121.tv_in1k",
}

def resolve_checkpoint(backbone: str) -> str:
    """
    config.backbone ∈ {resnet, efficientnet, efficientnetb0, mobilenet, densenet}만 허용.
    timm 리포 아이디로 매핑.
    """
    key = backbone.strip().lower()
    if key not in _BACKBONE_FAMILY_MAP:
        raise ValueError(f"backbone must be one of {list(_BACKBONE_FAMILY_MAP.keys())}, got: {backbone}")
    return _BACKBONE_FAMILY_MAP[key]


# =============================================================================
# [헬퍼] 1) 캐시 인덱스 로딩 (splitter.py가 만든 NPZ + meta.json)
# =============================================================================
@dataclass
class SplitIndices:
    train: List[int]
    val: List[int]
    test: List[int]

@dataclass
class Fold:
    train: List[int]
    val: List[int]

def load_indices_from_cache(
    cache_dir: str | Path,
    mode: Literal["split", "cv"] = "split",
    fold_index: Optional[int] = None,
) -> Tuple[Optional[SplitIndices], Optional[List[Fold]], Dict[str, Any]]:
    """
    splitter.save_split_cache(...) 산출물 로딩.
      - mode="split": split_cache.npz → SplitIndices 반환
      - mode="cv"   : cv_cache.npz    → List[Fold] 반환( fold_index 지정시 해당 폴드만 )
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in: {cache_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kind = meta.get("kind", mode)

    if mode == "split":
        npz_path = cache_dir / "split_cache.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"split_cache.npz not found in: {cache_dir}")
        npz = np.load(npz_path)
        split = SplitIndices(
            train=npz["train"].astype(int).tolist(),
            val=npz["val"].astype(int).tolist(),
            test=npz["test"].astype(int).tolist(),
        )
        return split, None, meta

    # mode == "cv"
    npz_path = cache_dir / "cv_cache.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"cv_cache.npz not found in: {cache_dir}")
    npz = np.load(npz_path)
    keys = sorted(npz.files)
    kmax = max(int(k[5:]) for k in keys if k.startswith("train"))
    folds = [Fold(
        train=npz[f"train{k}"].astype(int).tolist(),
        val=npz[f"val{k}"].astype(int).tolist()
    ) for k in range(kmax + 1)]
    if fold_index is not None:
        if not (0 <= fold_index < len(folds)):
            raise IndexError(f"fold_index out of range: {fold_index}, total folds={len(folds)}")
        folds = [folds[fold_index]]
    return None, folds, meta


# =============================================================================
# [헬퍼] 2) timm 메타 → 전처리 사양 추출 + 전처리/콜레이트 생성
# =============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

_INTERP = {
    "bilinear": T.InterpolationMode.BILINEAR,
    "bicubic":  T.InterpolationMode.BICUBIC,
    "nearest":  T.InterpolationMode.NEAREST,
    "lanczos":  T.InterpolationMode.LANCZOS,
    "box":      T.InterpolationMode.BOX,
    "hamming":  T.InterpolationMode.HAMMING,
}

def _timm_meta(repo_or_name: str) -> Dict[str, Any]:
    """
    repo_or_name: 'timm/<model>' 또는 '<model>'
    반환: {'size','mean','std','interp','crop_pct'}
    """
    name = repo_or_name.split("/", 1)[1] if "/" in repo_or_name else repo_or_name
    try:
        # timm >= 0.9
        from timm.models import get_pretrained_cfg
        cfg = get_pretrained_cfg(name)
        size = int(getattr(cfg, "input_size", (3, 224, 224))[-1])
        mean = tuple(getattr(cfg, "mean", IMAGENET_MEAN))
        std  = tuple(getattr(cfg, "std",  IMAGENET_STD))
        interp = str(getattr(cfg, "interpolation", "bilinear")).lower()
        crop_pct = float(getattr(cfg, "crop_pct", 0.875))
        return {"size": size, "mean": mean, "std": std, "interp": interp, "crop_pct": crop_pct}
    except Exception:
        import timm
        from timm.data import resolve_data_config
        m = timm.create_model(name, pretrained=False)
        dc = resolve_data_config({}, model=m)
        size = int(dc.get("input_size", (3, 224, 224))[-1])
        mean = tuple(dc.get("mean", IMAGENET_MEAN))
        std  = tuple(dc.get("std",  IMAGENET_STD))
        interp = str(dc.get("interpolation", "bilinear")).lower()
        crop_pct = float(dc.get("crop_pct", 0.875))
        return {"size": size, "mean": mean, "std": std, "interp": interp, "crop_pct": crop_pct}

def _build_train_preproc_from_meta(
    meta: Dict[str, Any],
    preset: Literal["none","light","medium","strong"] = "medium",
    override_size: Optional[int] = None,
) -> T.Compose:
    """
    (너가 좋아했던) 다단계 증강 유지: Resize/RandomResizedCrop → Flip → ColorJitter → Affine (preset에 따라)
    최종 크기는 override_size가 있으면 그걸, 없으면 timm size 사용.
    """
    size = int(override_size or meta["size"])
    interp = _INTERP.get(meta["interp"], T.InterpolationMode.BILINEAR)

    if preset == "none":
        return T.Compose([
            T.Resize(int(size / meta["crop_pct"]), interpolation=interp),
            T.CenterCrop(size),
        ])
    elif preset == "light":
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.9, 1.0), interpolation=interp),
            T.RandomHorizontalFlip(0.5),
        ])
    elif preset == "medium":
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0), interpolation=interp),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.1, 0.1, 0.1, 0.02),
            T.RandomAffine(degrees=10, translate=(0.02, 0.02), scale=(0.95, 1.05), interpolation=interp),
        ])
    elif preset == "strong":
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=interp),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.2, 0.2, 0.2, 0.05),
            T.RandomAffine(degrees=15, translate=(0.03, 0.03), scale=(0.9, 1.1), interpolation=interp),
            T.TrivialAugmentWide(num_magnitude_bins=31, interpolation=interp),
        ])
    else:
        raise ValueError(f"unknown preset: {preset}")

def _build_eval_preproc_from_meta(
    meta: Dict[str, Any],
    override_size: Optional[int] = None,
) -> T.Compose:
    """timm eval 정책: Resize(int(size / crop_pct)) → CenterCrop(size)"""
    size = int(override_size or meta["size"])
    interp = _INTERP.get(meta["interp"], T.InterpolationMode.BILINEAR)
    return T.Compose([
        T.Resize(int(size / meta["crop_pct"]), interpolation=interp),
        T.CenterCrop(size),
    ])

def _make_torch_collates(
    train_pre: Optional[T.Compose],
    eval_pre: Optional[T.Compose],
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
):
    """
    ToTensor → Normalize 를 공통으로 적용.
    """
    to_tensor = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    def train_cf(batch):
        imgs = [b["image"] for b in batch]
        if train_pre is not None:
            imgs = [train_pre(im) for im in imgs]
        xs = torch.stack([to_tensor(im) for im in imgs], dim=0)
        ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {"pixel_values": xs, "labels": ys}

    def eval_cf(batch):
        imgs = [b["image"] for b in batch]
        if eval_pre is not None:
            imgs = [eval_pre(im) for im in imgs]
        xs = torch.stack([to_tensor(im) for im in imgs], dim=0)
        ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {"pixel_values": xs, "labels": ys}

    return train_cf, eval_cf

def timm_loader(
    backbone: Literal["resnet","efficientnet","mobilenet","densenet", "efficientnetb0"],
    aug_preset: Literal["none","light","medium","strong"] = "medium",
    image_size: Optional[int] = None,
):
    """
    (헬퍼) timm 메타 기반으로 train/eval collate_fn과 메타 정보를 생성.
    """
    repo = resolve_checkpoint(backbone)
    meta = _timm_meta(repo)
    train_pre = _build_train_preproc_from_meta(meta, preset=aug_preset, override_size=image_size)
    eval_pre  = _build_eval_preproc_from_meta(meta, override_size=image_size)
    train_cf, eval_cf = _make_torch_collates(train_pre, eval_pre, mean=meta["mean"], std=meta["std"])
    return train_cf, eval_cf, meta


# =============================================================================
# [헬퍼] 3) 데이터셋/시드/공통
# =============================================================================
class DictImageFolder(Dataset):
    """
    ImageFolder 래퍼: 변환 없이 {'image': PIL.Image, 'label': int} 반환.
    (전처리는 collate_fn에서 수행)
    """
    def __init__(self, root: str | Path):
        self.base = ImageFolder(root=root)

    @property
    def classes(self) -> List[str]:
        return list(self.base.classes)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pil, label = self.base[idx]
        return {"image": pil, "label": int(label)}

def _seed_worker(worker_id: int):
    import random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed); random.seed(worker_seed)


# =============================================================================
# [메인] 단일 호출 API
#   - build_loaders_from_split_cache(...): train/val/test (선택적으로 생성)
#   - build_loaders_from_cv_cache(...):   train/val (test=None)
# =============================================================================
@dataclass
class LoaderGroup:
    train: Optional[DataLoader]
    val: Optional[DataLoader]
    test: Optional[DataLoader]
    classes: List[str]
    meta: Dict[str, Any]   # timm 메타 요약(size/mean/std/interp/crop_pct)

def _summary_print(prefix: str, classes: List[str], tr_n: int, va_n: int, te_n: int,
                   meta: Dict[str, Any], effective_size: int | None = None):
    use_size = effective_size or meta['size']
    print(f"[{prefix}] classes={len(classes)} | train={tr_n} val={va_n} test={te_n} "
          f"| size={use_size} (meta={meta['size']}) crop_pct={meta['crop_pct']} interp={meta['interp']}")


def build_loaders_from_split_cache(
    data_root: str | Path,
    split_cache_dir: str | Path,
    backbone: Literal["resnet","efficientnet","mobilenet","densenet", "efficientnetb0"],
    batch_size: int = 32,
    aug_preset: Literal["none","light","medium","strong"] = "medium",
    image_size: Optional[int] = None,      # 지정 시 timm size 대신 사용
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    use_val: bool = True,
    use_test: bool = True,
    verbose: bool = True,
) -> LoaderGroup:
    """
    split 캐시를 읽어 '한 번에' train/val/test 로더를 구성.
    - val/test는 use_* 플래그에 따라 선택적으로 생성.
    - 전처리는 timm 메타 기반(증강은 train만).
    """
    split, _, meta_cache = load_indices_from_cache(split_cache_dir, mode="split")
    assert split is not None

    # collate 생성 (timm 메타)
    train_cf, eval_cf, meta = timm_loader(backbone, aug_preset=aug_preset, image_size=image_size)

    # Dataset/Subset
    ds = DictImageFolder(data_root)
    tr_ds = Subset(ds, split.train)
    va_ds = Subset(ds, split.val) if use_val and len(split.val) > 0 else None
    te_ds = Subset(ds, split.test) if use_test and len(split.test) > 0 else None

    # DataLoader 공통 옵션
    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=(num_workers > 0), worker_init_fn=_seed_worker)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                              generator=g, collate_fn=train_cf, **common)
    val_loader   = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=eval_cf, **common) if va_ds is not None else None
    test_loader  = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=eval_cf, **common) if te_ds is not None else None

    if verbose:
       _summary_print("split", ds.classes, len(split.train), len(split.val) if va_ds else 0,
               len(split.test) if te_ds else 0, meta, effective_size=(image_size or None))
    
    return LoaderGroup(train=train_loader, val=val_loader, test=test_loader,
                       classes=ds.classes, meta=meta)


def build_loaders_from_cv_cache(
    data_root: str | Path,
    cv_cache_dir: str | Path,
    fold: int,
    backbone: Literal["resnet","efficientnet","mobilenet","densenet", "efficientnetb0"],
    batch_size: int = 32,
    aug_preset: Literal["none","light","medium","strong"] = "medium",
    image_size: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> LoaderGroup:
    """
    cv 캐시를 읽어 '한 번에' train/val 로더를 구성(test=None).
    """
    _, folds, meta_cache = load_indices_from_cache(cv_cache_dir, mode="cv", fold_index=fold)
    assert folds is not None and len(folds) == 1
    f = folds[0]

    # collate 생성 (timm 메타)
    train_cf, eval_cf, meta = timm_loader(backbone, aug_preset=aug_preset, image_size=image_size)

    # Dataset/Subset
    ds = DictImageFolder(data_root)
    tr_ds = Subset(ds, f.train)
    va_ds = Subset(ds, f.val) if len(f.val) > 0 else None

    common = dict(num_workers=num_workers, pin_memory=pin_memory,
                  persistent_workers=(num_workers > 0), worker_init_fn=_seed_worker)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                              generator=g, collate_fn=train_cf, **common)
    val_loader   = DataLoader(va_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=eval_cf, **common) if va_ds is not None else None

    if verbose:
        _summary_print(f"cv(fold={fold})", ds.classes, len(f.train), len(f.val) if va_ds else 0, 0, meta, effective_size=(image_size or None))

    return LoaderGroup(train=train_loader, val=val_loader, test=None,
                       classes=ds.classes, meta=meta)

# ---- preview: 한 번의 imshow로 "클래스별 3장" 보기 ---------------------------
import math
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from typing import Optional, Tuple, Literal, Dict, List

def preview_classes_imshow(
    lg: "LoaderGroup",
    split: Literal["train","val","test"] = "train",
    per_class: int = 3,
    max_batches: int = 50,
    figsize: Optional[Tuple[float, float]] = None,
    label_x: int = 6,                 # 라벨 x 위치(픽셀)
    label_fontsize: int = 12,         # 라벨 글꼴 크기
    label_box: bool = True,           # 라벨 배경 박스 표시
) -> None:
    """
    LoaderGroup에서 지정 split의 배치들을 모아, 클래스별로 per_class장씩 추출한 뒤
    하나의 큰 그리드 이미지를 imshow '한 번'으로 표시하고, 각 행(클래스) 왼쪽에 라벨을 덧씌웁니다.
    """
    loader = getattr(lg, split, None)
    if loader is None:
        raise ValueError(f"preview_classes_imshow: '{split}' loader is None")

    classes = lg.classes
    K = len(classes)
    buckets: Dict[int, List[torch.Tensor]] = {i: [] for i in range(K)}

    def _denorm(x: torch.Tensor) -> torch.Tensor:
        m = torch.tensor(lg.meta["mean"], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        s = torch.tensor(lg.meta["std"],  dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x * s) + m

    # 샘플 수집
    H = W = None
    it = iter(loader)
    for _ in range(max_batches):
        try:
            b = next(it)
        except StopIteration:
            break
        x = b["pixel_values"]  # (B,3,H,W)
        y = b["labels"].tolist()
        x = _denorm(x).clamp(0, 1).cpu()
        if H is None:
            H, W = x.shape[-2], x.shape[-1]
        for i, yi in enumerate(y):
            if len(buckets[yi]) < per_class:
                buckets[yi].append(x[i])
        if all(len(buckets[i]) >= per_class for i in range(K)):
            break

    # 부족한 클래스는 빈 칸 패딩
    blank = torch.zeros(3, H or 224, W or 224)
    for ci in range(K):
        while len(buckets[ci]) < per_class:
            buckets[ci].append(blank)

    # 클래스별 가로 그리드 → 세로로 이어붙여 하나의 큰 이미지
    row_grids = [make_grid(buckets[ci], nrow=per_class, padding=2) for ci in range(K)]  # (3,h,w)
    row_h = row_grids[0].shape[1]
    grid = torch.cat(row_grids, dim=1)  # (3, H_total = row_h*K, W_row)

    # imshow 한 번
    if figsize is None:
        figsize = (min(16, 2.2 * per_class), min(20, 2.2 * K))
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    ax = fig.add_subplot(111)
    img = grid.permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.set_title(f"{split} preview — {per_class} per class ({K} classes)")
    ax.axis("off")

    # 각 행의 좌측에 클래스 라벨 덧씌우기 (픽셀 좌표계)
    for i, cname in enumerate(classes):
        y_mid = i * row_h + row_h * 0.5
        bbox = dict(facecolor="black", alpha=0.6, pad=2, edgecolor="none") if label_box else None
        ax.text(label_x, y_mid, cname, ha="left", va="center",
                fontsize=label_fontsize, color="white", bbox=bbox)

    plt.show()



# 코드 테스트

# lg = build_loaders_from_cv_cache(
    # data_root="/mnt/c/Users/KHJ/OneDrive/project/torch-bo/dataset-pepper-preprocessed",
    # cv_cache_dir="splits/cache_cv",
    # fold=0,
    # backbone="mobilenet",                # {resnet, efficientnet, mobilenet, densenet, efficientnetb0}
    # batch_size=32,
    # aug_preset="medium",
    # image_size=None,                  # timm size 사용; 지정 시 그 크기로 통일
# )

# print(lg.classes)  
# preview_classes_imshow(lg, split="train", per_class=3)