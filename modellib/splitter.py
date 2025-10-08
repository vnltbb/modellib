# splitter.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Union
from pathlib import Path
import hashlib
import json
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold

# =========================
# 데이터 구조
# =========================
@dataclass
class SplitIndices:
    train: List[int]
    val: List[int]
    test: List[int]

@dataclass
class Fold:
    train: List[int]
    val: List[int]


# =========================
# 내부 유틸
# =========================
def _get_targets_from_imagefolder(dataset) -> List[int]:
    """ImageFolder 호환 라벨 벡터 추출"""
    if hasattr(dataset, "targets"):
        return list(map(int, dataset.targets))
    if hasattr(dataset, "samples"):
        return [int(c) for _, c in dataset.samples]
    raise ValueError("dataset.targets 또는 dataset.samples가 필요합니다 (ImageFolder 호환).")

def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=float)
    if np.any(ratios < 0):
        raise ValueError("비율은 음수가 될 수 없습니다.")
    tot = ratios.sum()
    if tot == 0:
        raise ValueError("모든 비율이 0입니다. 최소 하나는 양수여야 합니다.")
    return tuple((ratios / tot).tolist())

def _assert_disjoint(*index_lists: List[int]) -> None:
    all_idx = [i for lst in index_lists for i in lst]
    if len(set(all_idx)) != len(all_idx):
        raise AssertionError("분할 인덱스에 중복이 존재합니다.")

def _class_counts(labels: List[int], indices: List[int]) -> Dict[int, int]:
    from collections import Counter
    return dict(Counter([labels[i] for i in indices]))

def _dataset_signature(dataset) -> Dict[str, Union[str, int]]:
    """
    데이터셋 시그니처: samples (path, class) 리스트를 해시하여 캐시 일관성 확인.
    """
    if not hasattr(dataset, "samples"):
        raise ValueError("dataset.samples가 필요합니다(캐시 시그니처 계산).")
    parts = [f"{p}|{c}" for p, c in dataset.samples]
    joined = "\n".join(parts).encode("utf-8")
    sha = hashlib.sha1(joined).hexdigest()
    return {"num_samples": len(dataset.samples), "sha1": sha}

def _equalize_min_per_class(indices: List[int], labels: List[int]) -> List[int]:
    """
    indices 내에서 각 클래스 샘플 수를 '최소 클래스 수'로 동일화.
    - 재현성을 위해 라운드로빈 방식(섞지 않음)
    - 반환은 정렬된 인덱스 리스트
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for i in indices:
        buckets[labels[i]].append(i)
    if len(buckets) <= 1:
        return sorted(indices)  # 이진분류 미만이면 그대로

    min_n = min(len(v) for v in buckets.values())
    out = []
    for t in range(min_n):
        for c in sorted(buckets.keys()):
            out.append(buckets[c][t])
    return sorted(out)

# =========================
# (a) 비율 분할 — stratified (+옵션 group-aware)
# =========================
def split_by_ratio(
    dataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratified: bool = True,
    group_labels: Optional[List[Union[str, int]]] = None,
    equalize_train_min: bool = False,  # 필요 시 동일 개체가 분할 간 겹치지 않도록
) -> SplitIndices:
    """
    train/val/test 비율 분할(셋 중 0 허용).
    - stratified: 클래스 분포 보존 (기본 True)
    - group_labels: 같은 그룹이 서로 다른 분할에 섞이지 않도록 보장(주어질 때만)
    """
    n = len(dataset)
    all_idx = np.arange(n)
    y = _get_targets_from_imagefolder(dataset)
    tr, va, te = _normalize_ratios(train_ratio, val_ratio, test_ratio)

    # 1) test 홀드아웃
    if te > 0:
        if group_labels is None:
            idx_rest, idx_test = train_test_split(
                all_idx, test_size=te, random_state=seed,
                stratify=y if stratified else None,
            )
        else:
            from sklearn.model_selection import GroupShuffleSplit
            gss = GroupShuffleSplit(n_splits=1, test_size=te, random_state=seed)
            (idx_rest, idx_test) = next(gss.split(all_idx, y, groups=group_labels))
    else:
        idx_rest, idx_test = all_idx, np.array([], dtype=int)

    # 2) 남은 것에서 train/val
    rem = 1.0 - te
    if rem == 0:
        idx_train, idx_val = np.array([], dtype=int), np.array([], dtype=int)
    else:
        rel_val = va / rem
        if rel_val == 0:
            idx_train, idx_val = idx_rest, np.array([], dtype=int)
        elif rel_val == 1:
            idx_train, idx_val = np.array([], dtype=int), idx_rest
        else:
            if group_labels is None:
                idx_train, idx_val = train_test_split(
                    idx_rest, test_size=rel_val, random_state=seed,
                    stratify=[y[i] for i in idx_rest] if stratified else None,
                )
            else:
                from sklearn.model_selection import GroupShuffleSplit
                gss = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
                (idx_train, idx_val) = next(
                    gss.split(idx_rest, [y[i] for i in idx_rest],
                              groups=[group_labels[i] for i in idx_rest])
                )

    
    if equalize_train_min and len(idx_train) > 0:
        y_all = _get_targets_from_imagefolder(dataset)
        idx_train = np.array(_equalize_min_per_class(idx_train.tolist(), y_all), dtype=int)

    _assert_disjoint(idx_train.tolist(), idx_val.tolist(), idx_test.tolist())
    return SplitIndices(
        train=sorted(map(int, idx_train.tolist())),
        val=sorted(map(int, idx_val.tolist())),
        test=sorted(map(int, idx_test.tolist())),
    )


# =========================
# (b) 교차검증 — Stratified K-Fold (+옵션 group-aware), base_indices 지원
# =========================
def make_cv_folds(
    dataset,
    n_splits: int = 5,
    seed: int = 42,
    group_labels: Optional[List[Union[str, int]]] = None,
    base_indices: Optional[List[int]] = None,   # ← (a)의 train 인덱스를 전달하면 그 범위에서만 K-Fold
) -> List[Fold]:
    """
    Stratified K-Fold (기본). group_labels가 주어지면 Group-aware K-Fold.
    base_indices가 주어지면 해당 인덱스 서브셋에서만 폴드를 생성.
    """
    if base_indices is None:
        base_indices = list(range(len(dataset)))
    base_indices = np.array(sorted(base_indices), dtype=int)

    y_all = _get_targets_from_imagefolder(dataset)
    y = [y_all[i] for i in base_indices]

    folds: List[Fold] = []
    if group_labels is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_loc, va_loc in skf.split(base_indices, y):
            folds.append(Fold(
                train=sorted(base_indices[tr_loc].tolist()),
                val=sorted(base_indices[va_loc].tolist())
            ))
    else:
        from sklearn.model_selection import StratifiedGroupKFold
        groups = [group_labels[i] for i in base_indices]
        sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_loc, va_loc in sgk.split(base_indices, y, groups=groups):
            folds.append(Fold(
                train=sorted(base_indices[tr_loc].tolist()),
                val=sorted(base_indices[va_loc].tolist())
            ))
    return folds


# =========================
# (c) 캐시 저장/로드 (NPZ + JSON 메타) — 빠른 로딩 목적
# =========================
def _save_meta(cache_dir: Path, dataset, kind: str, tag: Optional[str]) -> None:
    sig = _dataset_signature(dataset)
    meta = {
        "kind": kind,
        "dataset_signature": sig,
        "num_classes": len(getattr(dataset, "classes", [])),
        "classes": list(getattr(dataset, "classes", [])),
        "tag": tag or "",
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def save_split_cache(
    cache_dir: Union[str, Path],
    dataset,
    split: Union[SplitIndices, List[Fold]],
    kind: Literal["split", "cv"] = "split",
    tag: Optional[str] = None,
) -> None:
    """
    캐시 구성:
      - split: split_cache.npz (train/val/test) + meta.json
      - cv   : cv_cache.npz (train0,val0,train1,val1,...) + meta.json
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _save_meta(cache_dir, dataset, kind, tag)

    if kind == "split":
        assert isinstance(split, SplitIndices)
        np.savez(cache_dir / "split_cache.npz",
                 train=np.array(split.train, dtype=np.int64),
                 val=np.array(split.val, dtype=np.int64),
                 test=np.array(split.test, dtype=np.int64))
    else:
        assert isinstance(split, list)
        arrays = {}
        for k, f in enumerate(split):
            arrays[f"train{k}"] = np.array(f.train, dtype=np.int64)
            arrays[f"val{k}"]   = np.array(f.val, dtype=np.int64)
        np.savez(cache_dir / "cv_cache.npz", **arrays)


def load_split_cache(
    cache_dir: Union[str, Path],
    dataset=None,
    strict: bool = True,
) -> Tuple[str, Union[SplitIndices, List[Fold]], Dict]:
    """
    캐시 로드:
      - meta.json 확인 → dataset_signature 일치 여부 검사(strict=True면 불일치 시 에러)
      - split_cache.npz 또는 cv_cache.npz 로딩
    반환: (kind, split_or_folds, meta)
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json이 없습니다: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kind = meta["kind"]

    if dataset is not None:
        current = _dataset_signature(dataset)
        same = (current["sha1"] == meta["dataset_signature"]["sha1"]) and \
               (current["num_samples"] == meta["dataset_signature"]["num_samples"])
        if strict and not same:
            raise RuntimeError("캐시와 현재 데이터셋이 일치하지 않습니다(sha1/샘플수 불일치). 캐시 갱신 필요.")

    if kind == "split":
        npz = np.load(cache_dir / "split_cache.npz")
        split = SplitIndices(
            train=npz["train"].astype(int).tolist(),
            val=npz["val"].astype(int).tolist(),
            test=npz["test"].astype(int).tolist(),
        )
        return kind, split, meta
    else:
        npz = np.load(cache_dir / "cv_cache.npz")
        keys = sorted(npz.files)
        kmax = max(int(k[5:]) for k in keys if k.startswith("train"))
        folds = []
        for k in range(kmax + 1):
            folds.append(Fold(
                train=npz[f"train{k}"].astype(int).tolist(),
                val=npz[f"val{k}"].astype(int).tolist()
            ))
        return kind, folds, meta


# =========================
# (d) 리포트 & 표/이미지 저장
# =========================
def report_split(dataset, split: SplitIndices) -> Dict:
    y = _get_targets_from_imagefolder(dataset)
    rep = {
        "sizes": {"train": len(split.train), "val": len(split.val), "test": len(split.test)},
        "label_dist": {
            "train": _class_counts(y, split.train),
            "val": _class_counts(y, split.val),
            "test": _class_counts(y, split.test),
        },
    }
    _assert_disjoint(split.train, split.val, split.test)
    return rep

def report_cv(dataset, folds: List[Fold]) -> List[Dict]:
    y = _get_targets_from_imagefolder(dataset)
    out = []
    for k, f in enumerate(folds):
        _assert_disjoint(f.train, f.val)
        out.append({
            "fold": k,
            "sizes": {"train": len(f.train), "val": len(f.val)},
            "label_dist": {
                "train": _class_counts(y, f.train),
                "val": _class_counts(y, f.val),
            }
        })
    return out


# ---- 표 + 이미지 저장 유틸 ----
def data_table_split(rep: Dict, save_basepath: Union[str, Path],
                     font=11, header_font=12, dpi=220) -> Tuple[Path, Path]:
    """
    report_split(rep)를 두 개의 이미지로 저장:
      1) *_top.png    : 데이터 전체 크기 (|dataset|train|val|test|)
      2) *_bottom.png : 클래스별 분포   (|label_dist|train|val|test|)
    반환: (top_path, bottom_path)
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    save_basepath = Path(save_basepath)
    save_basepath.parent.mkdir(parents=True, exist_ok=True)

    # ---------- TOP ----------
    # 원하는 형태: |dataset|train|val|test|, 행은 'count' 1개
    df_top = pd.DataFrame(
        [["count", rep["sizes"]["train"], rep["sizes"]["val"], rep["sizes"]["test"]]],
        columns=["dataset", "train", "val", "test"]
    )

    fig = plt.figure(figsize=(6.2, 1.9), constrained_layout=True)
    ax = fig.add_subplot(111); ax.axis("off")

    tbl = ax.table(cellText=df_top.values,
                   colLabels=df_top.columns,
                   loc="center", cellLoc="center",
                   colWidths=[0.4, 0.7, 0.7, 0.7])  # dataset 칸 좁게
    tbl.auto_set_font_size(False); tbl.set_fontsize(font)
    for _, cell in tbl.get_celld().items(): cell.set_linewidth(0.6)

    top_path = save_basepath.with_suffix("").with_name(save_basepath.stem + "_top.png")
    fig.savefig(top_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    # ---------- BOTTOM ----------
    # 원하는 형태: |label_dist|train|val|test|
    import numpy as np
    classes = sorted(set(list(rep["label_dist"]["train"].keys())
                         + list(rep["label_dist"]["val"].keys())
                         + list(rep["label_dist"]["test"].keys())))
    rows = []
    for c in classes:
        rows.append([
            c,
            rep["label_dist"]["train"].get(c, 0),
            rep["label_dist"]["val"].get(c, 0),
            rep["label_dist"]["test"].get(c, 0),
        ])
    df_bot = pd.DataFrame(rows, columns=["label_dist", "train", "val", "test"])

    h = max(2.0, 1.1 + 0.33 * len(df_bot))  # 행 수에 따라 높이 자동
    fig = plt.figure(figsize=(6.2, h), constrained_layout=True)
    ax = fig.add_subplot(111); ax.axis("off")

    tbl = ax.table(cellText=df_bot.values,
                   colLabels=df_bot.columns,
                   loc="center", cellLoc="center",
                   colWidths=[0.4, 0.7, 0.7, 0.7])  # label_dist 칸 좁게
    tbl.auto_set_font_size(False); tbl.set_fontsize(font)
    for _, cell in tbl.get_celld().items(): cell.set_linewidth(0.6)

    bot_path = save_basepath.with_suffix("").with_name(save_basepath.stem + "_bottom.png")
    fig.savefig(bot_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    return top_path, bot_path



def data_table_cv(rep_list: List[Dict], save_basepath: Union[str, Path],
                  font=11, header_font=12, dpi=220) -> Tuple[Path, Path]:
    """
    report_cv(rep_list)를 두 개 이미지로 저장:
      1) *_top.png    : fold-wise sizes   (|fold|train|val|)
      2) *_bottom.png : label_dist (fold, class) (|fold|class|train|val|)
    fold/class 칸 폭은 다른 칸 대비 절반으로.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    save_basepath = Path(save_basepath)
    save_basepath.parent.mkdir(parents=True, exist_ok=True)

    # ---------- TOP ----------
    df_top = pd.DataFrame([
        {"fold": r["fold"], "train": r["sizes"]["train"], "val": r["sizes"]["val"]}
        for r in rep_list
    ]).sort_values("fold")

    h = max(1.9, 1.1 + 0.33 * len(df_top))
    fig = plt.figure(figsize=(5.6, h), constrained_layout=True)
    ax = fig.add_subplot(111); ax.axis("off")

    tbl = ax.table(cellText=df_top.values,
                   colLabels=["fold", "train", "val"],
                   loc="center", cellLoc="center",
                   colWidths=[0.3, 0.9, 0.9])   # fold = 1/2 폭
    tbl.auto_set_font_size(False); tbl.set_fontsize(font)
    for _, cell in tbl.get_celld().items(): cell.set_linewidth(0.9)

    top_path = save_basepath.with_suffix("").with_name(save_basepath.stem + "_top.png")
    fig.savefig(top_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    # ---------- BOTTOM ----------
    # 표: |fold|class|train|val|
    all_classes = sorted(set().union(*[
        set(r["label_dist"]["train"].keys()) | set(r["label_dist"]["val"].keys())
        for r in rep_list
    ]))
    rows = []
    for r in rep_list:
        k = r["fold"]
        for c in all_classes:
            rows.append([k, c,
                         r["label_dist"]["train"].get(c, 0),
                         r["label_dist"]["val"].get(c, 0)])
    df_bot = pd.DataFrame(rows, columns=["fold", "class", "train", "val"]) \
               .sort_values(["fold", "class"])

    h = max(2.2, 1.1 + 0.30 * len(df_bot))
    fig = plt.figure(figsize=(6.4, min(18, h)), constrained_layout=True)
    ax = fig.add_subplot(111); ax.axis("off")

    tbl = ax.table(cellText=df_bot.values,
                   colLabels=["fold", "class", "train", "val"],
                   loc="center", cellLoc="center",
                   colWidths=[0.25, 0.25, 0.7, 0.7])  # fold/class = 1/2 폭
    tbl.auto_set_font_size(False); tbl.set_fontsize(font)
    for _, cell in tbl.get_celld().items(): cell.set_linewidth(0.6)

    bot_path = save_basepath.with_suffix("").with_name(save_basepath.stem + "_bottom.png")
    fig.savefig(bot_path, dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    return top_path, bot_path



# 코드 테스트
# from torchvision.datasets import ImageFolder
# ds = ImageFolder("/mnt/c/Users/KHJ/OneDrive/project/torch-bo/dataset-pepper-preprocessed")
# split3 = split_by_ratio(ds, train_ratio=0.9, val_ratio=0.0, test_ratio=0.1,
                        # seed=42, stratified=True)

# split = split_by_ratio(
    # ds, 0.7, 0.2, 0.1,
    # seed=42, stratified=True,
    # equalize_train_min=True  # ← 가장 적은 클래스 개수에 맞춰 train만 균등화
# )

# rep = report_split(ds, split)
# print(rep)
# data_table_split(rep, "splits/cache_split/report_split_bal.png")
