import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_individual_roc(y_true, y_proba, model_name, filename):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {model_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def plot_comparison_roc(y_true, y_proba_lgbm, y_proba_ncf, filename='comparison_roc.png'):
    fpr_l, tpr_l, _ = roc_curve(y_true, y_proba_lgbm)
    fpr_n, tpr_n, _ = roc_curve(y_true, y_proba_ncf)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_l, tpr_l, lw=2, label=f'LightGBM (AUC = {auc(fpr_l, tpr_l):.3f})')
    plt.plot(fpr_n, tpr_n, lw=2, label=f'PyTorch NCF (AUC = {auc(fpr_n, tpr_n):.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('Model Comparison: Content vs. Collaborative')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")