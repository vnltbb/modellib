import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from typing import Tuple, List, Generator

# 헬퍼 함수 및 클래스는 이전과 동일합니다.
def get_transforms(backbone: str, image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    # ... 이전 코드와 동일 (생략)
    try:
        model_cfg = timm.create_model(backbone, pretrained=False).default_cfg
        mean, std, input_size = model_cfg['mean'], model_cfg['std'], model_cfg['input_size']
        image_size = input_size[-1]
        # print(f"Using model-specific transforms for '{backbone}': mean={mean}, std={std}, size={image_size}")
    except Exception:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        # print(f"Warning: Using ImageNet default transforms.")
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transforms, val_transforms

class TransformedSubset(Subset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
    def __getitem__(self, idx):
        # 1. 원본 데이터셋의 __getitem__을 호출하여 데이터를 가져옵니다.
        #    (full_dataset의 transform이 None이므로 여기서는 PIL Image와 레이블이 반환됩니다.)
        img, label = self.dataset[self.indices[idx]]

        # 2. 이 Subset에 지정된 transform을 적용합니다.
        if self.transform:
            img = self.transform(img)

        return img, label
    def __len__(self):
        return len(self.indices)

# --- Main Function ---
def create_dataloaders(
    data_dir: str,
    backbone: str,
    batch_size: int,
    # --- New CV arguments ---
    cv: bool = False,
    n_splits: int = 5,
    # --- Simple split arguments ---
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    test_dir: str = None,
    num_workers: int = 0,
    random_state: int = 42
):
    """
    데이터셋을 분리하고 PyTorch DataLoader를 생성합니다.
    교차 검증(cv=True)과 단일 분리(cv=False) 모드를 모두 지원합니다.
    """
    train_transform, val_transform = get_transforms(backbone)
    
    # 1. 테스트셋 우선 분리
    test_loader = None
    if test_dir:
        print(f"Using separate test dataset from '{test_dir}'")
        main_data_dir = data_dir
        test_dataset = datasets.ImageFolder(root=test_dir, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        main_data_dir = data_dir

    # PIL Image로 로드하기 위해 transform은 None으로 설정
    # __getitem__에서 transform을 적용하므로, ImageFolder 자체에는 transform을 적용하지 않습니다.
    full_dataset = datasets.ImageFolder(root=main_data_dir, transform=None)
    class_names = full_dataset.classes
    indices = np.arange(len(full_dataset))
    labels = np.array([s[1] for s in full_dataset.samples])

    # test_dir이 없을 경우, main_data_dir에서 test set을 분리
    if not test_dir and test_ratio > 0:
        train_val_indices, test_indices, _, test_labels = train_test_split(
            indices, labels, test_size=test_ratio, stratify=labels, random_state=random_state
        )
        test_subset = TransformedSubset(full_dataset, test_indices, val_transform)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        # CV 또는 train/val 분리는 test를 제외한 나머지 데이터로 수행
        main_indices = train_val_indices
        main_labels = labels[train_val_indices]
    else:
        main_indices = indices
        main_labels = labels
    
    # 2. CV 또는 단일 분리 로직
    if cv:
        print(f"\nSetting up {n_splits}-Fold Cross-Validation.")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # 제너레이터를 반환하는 내부 함수
        def cv_generator():
            for fold, (train_idx, val_idx) in enumerate(skf.split(main_indices, main_labels)):
                # print(f"--- Fold {fold+1}/{n_splits} ---")
                train_actual_indices = main_indices[train_idx]
                val_actual_indices = main_indices[val_idx]

                train_subset = TransformedSubset(full_dataset, train_actual_indices, train_transform)
                val_subset = TransformedSubset(full_dataset, val_actual_indices, val_transform)
                
                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                
                yield fold, train_loader, val_loader

        return cv_generator(), test_loader, class_names

    else:
        print(f"\nSetting up a single train/validation split (val_ratio={val_ratio}).")
        train_indices, val_indices = train_test_split(
            main_indices, test_size=val_ratio, stratify=main_labels, random_state=random_state
        )
        
        train_subset = TransformedSubset(full_dataset, train_indices, train_transform)
        val_subset = TransformedSubset(full_dataset, val_indices, val_transform)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        print_dataset_summary(train_loader, val_loader, test_loader, class_names)
        return train_loader, val_loader, test_loader, class_names


def print_dataset_summary(train_loader, val_loader, test_loader, class_names):
    print("\nDataLoaders created successfully.")
    print(f"  - Train samples: {len(train_loader.dataset)}")
    print(f"  - Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        print(f"  - Test samples: {len(test_loader.dataset)}")
    print(f"  - Classes: {class_names}")