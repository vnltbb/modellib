import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import math
import statistics

from modellib.build import build as build_model
from modellib.loader import create_dataloaders

# --- 헬퍼 함수: 1 에폭 훈련 및 검증 ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, macro_f1


# --- Objective 클래스  ---
class Objective:
    def __init__(
        self,
        config: dict, 
        data_dir: str,
        backbone: str,
        max_epochs: int,
        n_splits: int,
        metric_to_optimize: str = 'val_loss',
        test_ratio: float = 0.15
    ):
        self.config = config 
        self.data_dir = data_dir
        self.backbone = backbone
        self.max_epochs = max_epochs
        self.n_splits = n_splits
        self.test_ratio = test_ratio
        
        if metric_to_optimize not in ['val_loss', 'macro_f1']:
            raise ValueError("metric_to_optimize must be 'val_loss' or 'macro_f1'")
        self.metric_to_optimize = metric_to_optimize
        self.direction = 'minimize' if metric_to_optimize == 'val_loss' else 'maximize'
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Objective configured to run on device: {self.device}")

    def __call__(self, trial: optuna.Trial) -> float:
        # 1. 설정 파일로부터 search space 동적 생성
        params = {}
        for name, p in self.config['search_space'].items():
            if p['type'] == 'categorical':
                params[name] = trial.suggest_categorical(name, p['choices'])
            elif p['type'] == 'float':
                params[name] = trial.suggest_float(name, p['low'], p['high'], log=p.get('log', False))

        print(f"\n--- Starting Trial {trial.number} ---")
        print(f"  Params: {params}")

        # 2. 데이터 로더 생성 (K-Fold CV)
        try:
            
            cv_generator, _, _ = create_dataloaders(
                data_dir=self.data_dir,
                backbone=self.backbone,
                batch_size=params['batch_size'], 
                cv=True,
                n_splits=self.n_splits,
                test_ratio=self.test_ratio
            )
        except Exception as e:
            print(f"Data loading failed: {e}")
            raise optuna.exceptions.TrialPruned()

        fold_metrics = []
        fold_best_epochs = []
        fold_best_steps = []
        global_step = 0 
        
        # 3. K-Fold 교차 검증 루프
        for fold, train_loader, val_loader in cv_generator:
            
            model = build_model(
                self.backbone, 
                num_classes=len(train_loader.dataset.dataset.classes), 
                dropout_rate=params['dropout_rate']
            ).to(self.device)
            
            optimizer = getattr(optim, params['optimizer'])(
                model.parameters(), 
                lr=params['lr'], 
                weight_decay=params.get('weight_decay', 0.0) # weight_decay는 없을 수 있으므로 .get() 사용
            )
            criterion = nn.CrossEntropyLoss()

            best_fold_metric = -np.inf if self.direction == 'maximize' else np.inf
            best_epoch_in_fold = None
            
            steps_per_epoch_fold = len(train_loader)
            
            # 4. 에폭 루프
            for epoch in range(self.max_epochs):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, self.device)
                val_loss, macro_f1 = validate_one_epoch(model, val_loader, criterion, self.device)
                
                print(f"  [Trial {trial.number}/Fold {fold+1}] Epoch {epoch+1}/{self.max_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {macro_f1:.4f}")

                # 5. Pruning (가지치기)
                metric_for_pruning = val_loss if self.metric_to_optimize == 'val_loss' else macro_f1
                
                trial.report(metric_for_pruning, global_step)
                global_step += 1 # global_step 증가
                
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                if self.direction == 'maximize':
                    if macro_f1 > best_fold_metric:
                        best_fold_metric = macro_f1
                        best_epoch_in_fold = epoch
                else:
                    if val_loss < best_fold_metric:
                        best_fold_metric = val_loss
                        best_epoch_in_fold = epoch

            fold_metrics.append(best_fold_metric)
            
            if best_epoch_in_fold is None:
                best_epoch_in_fold = self.max_epochs -1
            fold_best_epochs.append(best_epoch_in_fold + 1)
            fold_best_steps.append((best_epoch_in_fold+1)*steps_per_epoch_fold)

        # 6. K-Fold 결과의 평균을 최종 점수로 반환
        final_score = np.mean(fold_metrics)
        
        target_steps_median = int(statistics.median(fold_best_steps))
        trial.set_user_attr("fold_best_epochs", fold_best_epochs)
        trial.set_user_attr("fold_best_steps", fold_best_steps)
        trial.set_user_attr("target_steps_median", target_steps_median)
        
        return final_score