import os
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