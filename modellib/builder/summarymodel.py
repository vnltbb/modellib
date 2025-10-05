import torch.nn as nn
from torchinfo import summary

def summarymodel(
        model,
        input_size,
):
    # 1. 모델 구조 서머리
    result=summary(
    model,
    input_size=input_size,
    depth=1,
    col_names=("input_size", "output_size", "num_params", "trainable"),
    col_width=18,
    row_settings=("var_names",),
    device="cpu",
    )
    print(result)

    # 2. trainable params 체크 > 모델 서머리와 수치 일치하는지 확인
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable_params / total_params if total_params > 0 else 0.0
    print(f"\n[Params] trainable {trainable_params:,} / total {total_params:,}  ({pct:.2f}%)")
    # 어떤 상위 모듈이 학습 대상인지(= parent 관점) 간단 체크
    trainable_parent_names = set()
    for name, p in model.named_parameters():
        if p.requires_grad:
        # "classifier.weight" -> "classifier" 상위 이름만 추출
            parent = name.split(".", 1)[0]
            trainable_parent_names.add(parent)
    print("[Trainable parents]", sorted(trainable_parent_names))

    # 3. BN 동결 검증
    bn_requires_grad = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)):
        # affine이 없을 수 있어 getattr로 안전 접근
            w_req = getattr(m.weight, "requires_grad", False)
            b_req = getattr(m.bias,   "requires_grad", False)
            if w_req or b_req:
                bn_requires_grad.append(type(m).__name__)
    if bn_requires_grad:
        print("[Warn] Some BN have requires_grad=True:", bn_requires_grad)
    else:
        print("[OK] All BN affine params are frozen.")


# 코드 테스트
# num_classes=5
# drop_rate=0.2
# input_size=(3,256,256)
    # input_size_4d=(1, conf['input_size']) >> 리스트 형식으로 'config' 작성

# model = build(
    # num_classes=num_classes,
    # drop_rate=drop_rate
# )

# summarymodel(model, (1, 3, 256, 256))