import os
import sys
import shutil
import json
import datetime
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

class ExperimentTracker:
    def __init__(self, experiment_name="SAILER_MSP"):
        # 1. 建立時間戳記資料夾
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = f"experiments/{now}_{experiment_name}"
        
        self.weights_dir = os.path.join(self.exp_dir, "weights")
        self.plots_dir = os.path.join(self.exp_dir, "plots")
        self.logs_dir = os.path.join(self.exp_dir, "logs")
        
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # 2. 啟動 TensorBoard
        self.writer = SummaryWriter(log_dir=self.logs_dir)

        # 3. 啟動 Weights & Biases (雲端同步)
        wandb.init(
            project="SAILER_Emotion_Recognition",
            name=f"{now}_{experiment_name}",      
            sync_tensorboard=True               
        )
        print(f"實驗資料夾與 W&B 已建立: {self.exp_dir}")

        # 4. 註冊全局崩潰攔截器 (Auto Cleanup on Crash)
        self._setup_exception_hook()

    def _setup_exception_hook(self):
        """
        攔截 Python 的未處理錯誤 (Unhandled Exceptions) 與 Ctrl+C (KeyboardInterrupt)。
        如果程式崩潰，且尚未儲存任何有效權重，則自動清理產生的空資料夾與 W&B 錯誤連線。
        """
        original_hook = sys.excepthook
        
        def cleanup_hook(exc_type, exc_value, exc_traceback):
            print(f"\n[ExperimentTracker] 🚨 偵測到程式中止 ({exc_type.__name__})")
            
            # 把 TensorBoard 寫入流關閉
            if hasattr(self, 'writer'):
                self.writer.close()
                
            # 關閉 WandB 並標記為失敗 (Failed)
            if wandb.run is not None:
                wandb.finish(exit_code=1)
                
            # 判斷是否要刪除本地資料夾 (若 weights_dir 內無任何檔案，代表連 1 個 Epoch 都沒跑完)
            try:
                if os.path.exists(self.weights_dir) and len(os.listdir(self.weights_dir)) == 0:
                    print(f"[ExperimentTracker] 此實驗未產生任何有用權重，啟動清理機制...")
                    shutil.rmtree(self.exp_dir)
                    print(f"[ExperimentTracker] 已刪除殘留垃圾資料夾: {self.exp_dir}")
                else:
                    print(f"[ExperimentTracker] 保留實驗資料夾，裡面已經存放了儲存好的權重檔案。")
            except Exception as e:
                pass
                
            # 呼叫原始的 Python 錯誤處理機制，讓錯誤訊息 (Traceback 字串) 原生態地印在終端機上
            original_hook(exc_type, exc_value, exc_traceback)
            
        sys.excepthook = cleanup_hook

    def save_config(self, config_dict):
        with open(os.path.join(self.exp_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)
        wandb.config.update(config_dict)    

    def log_metrics(self, epoch, train_loss, val_loss, val_acc):
        self.writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_loss}, epoch)
        self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)

    def plot_loss_curve(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', color='blue')
        plt.plot(val_losses, label='Validation Loss', color='red')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plots_dir, 'loss_curve.png'))
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, class_names):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'))
        plt.close()

    def close(self):
        self.writer.close()
        wandb.finish()