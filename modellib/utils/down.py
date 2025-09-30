import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List

def downscale_dataset(
    source_dir: str,
    target_dir: str,
    target_size: Tuple[int, int],
    image_exts: List[str] = ['.jpg', '.jpeg', '.png', '.bmp']
):
    """
    소스 디렉토리의 모든 이미지를 지정된 크기로 리사이징하여
    타겟 디렉토리에 저장합니다.

    Args:
        source_dir (str): 원본 이미지들이 있는 루트 디렉토리 경로.
        target_dir (str): 리사이징된 이미지를 저장할 루트 디렉토리 경로.
        target_size (Tuple[int, int]): 리사이즈할 크기 (width, height).
        image_exts (List[str], optional): 처리할 이미지 확장자 리스트.
                                          Defaults to ['.jpg', '.jpeg', '.png', '.bmp'].
    """
    source_path_obj = Path(source_dir)
    target_path_obj = Path(target_dir)

    print("-" * 50)
    print(f"🖼️  데이터셋 다운스케일을 시작합니다.")
    print(f"원본 경로: {source_path_obj}")
    print(f"저장 경로: {target_path_obj}")
    print(f"리사이즈 크기: {target_size}")
    print("-" * 50)

    # DecompressionBomb 경고 방지
    Image.MAX_IMAGE_PIXELS = None

    all_images = []
    for ext in image_exts:
        all_images.extend(source_path_obj.rglob(f"*{ext}"))
        all_images.extend(source_path_obj.rglob(f"*{ext.upper()}"))
    all_images = sorted(list(set(all_images)))

    if not all_images:
        print(f"오류: '{source_dir}'에서 이미지를 찾을 수 없습니다.")
        return

    for src_img_path in tqdm(all_images, desc="이미지 리사이징 중"):
        try:
            relative_path = src_img_path.relative_to(source_path_obj)
            target_img_path = target_path_obj / relative_path
            
            target_img_path.parent.mkdir(parents=True, exist_ok=True)
            
            with Image.open(src_img_path) as img:
                img_rgb = img.convert("RGB")
                img_resized = img_rgb.resize(target_size)
                img_resized.save(target_img_path)
        except Exception as e:
            print(f"\n파일 처리 중 오류 발생: {src_img_path}, 오류: {e}")

    print("\n🎉 모든 이미지 리사이징이 완료되었습니다!")