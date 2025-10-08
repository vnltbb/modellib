from pathlib import Path
import torch

def save_best_weights(model, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)  # 텐서만 저장 → 피클 이슈 없음
    return str(save_path)

def load_weights_for_inference(model, load_path, map_location="cpu", strict=True):
    # PyTorch 2.6+ 기본이 weights_only=True → 가중치 파일은 바로 로드됨
    state = torch.load(load_path, map_location=map_location)
    model.load_state_dict(state, strict=strict)
    return state

# 사용 예시
# state = load_weights_for_inference(model, best_weights_path, map_location=device, strict=True)
# model.eval()
# 필요 시, 별도 메타 기록
# with open(runs_dir / "checkpoints" / "best_meta.txt", "w", encoding="utf-8") as f:
    # f.write(f"epoch={epoch}\nval_f1={va_f1:.6f}\nval_loss={va_loss:.6f}\n")