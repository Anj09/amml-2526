import torch
import os
import sys
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy import stats
from model import VariationalAutoencoder 
import visuals  # visual.py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


############################ DATA LOADING ########################################


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)
PROJECT_ROOT = "/Users/anjali98/Desktop/SUSSEX_LABS/amml-2526-main"

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

MODEL_WEIGHTS = {
    "model0": os.path.join(DATA_ROOT, "amml_model0_weights.pth"),
    "model1": os.path.join(DATA_ROOT, "amml_model1_weights.pth"),
    "model2": os.path.join(DATA_ROOT, "amml_model2_weights.pth"),
}

TEST_DATA_PATH = os.path.join(DATA_ROOT, "test_dataset.pt")
HOLDOUT_DATA_PATH = os.path.join(DATA_ROOT, "holdout_dataset.pt")



BATCH_SIZE = 128


def load_vae(weights_path):
    model = VariationalAutoencoder().to(device)
    state = torch.load(weights_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

model0 = load_vae(MODEL_WEIGHTS["model0"])
model1 = load_vae(MODEL_WEIGHTS["model1"])
model2 = load_vae(MODEL_WEIGHTS["model2"])


torch.serialization.add_safe_globals([Subset])

def load_pt_dataset(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, Subset):
        return data
    if isinstance(data, (list, tuple)):
        x, y = data
    elif isinstance(data, dict):
        x, y = data["x"], data["y"]
    else:
        raise ValueError(f"Unexpected .pt format: {type(data)}")
    if x.ndim == 3:
        x = x.unsqueeze(1)
    x = x.float() / 255.0
    y = y.long()
    return TensorDataset(x, y)

test_dataset = load_pt_dataset(TEST_DATA_PATH)
holdout_dataset = load_pt_dataset(HOLDOUT_DATA_PATH)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
holdout_dataloader = DataLoader(holdout_dataset, batch_size=BATCH_SIZE, shuffle=False)


################################################ MODEL LOADING #################################

def load_vae(weights_path):
    model = VariationalAutoencoder().to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    return model

model0 = load_vae(MODEL_WEIGHTS["model0"])
model1 = load_vae(MODEL_WEIGHTS["model1"])
model2 = load_vae(MODEL_WEIGHTS["model2"])


################################## CORE VAE / METRICS ####################

def mse_per_image(x, recon):
    return F.mse_loss(recon, x, reduction="none").view(x.size(0), -1).mean(dim=1)

def psnr_from_mse(mse, max_val=1.0):
    return 10.0 * torch.log10((max_val ** 2) / (mse + 1e-8))

def ssim_batch(x, recon, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(recon, kernel_size=3, stride=1, padding=1)
    sigma_x = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(recon * recon, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * recon, 3, 1, 1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2) + 1e-8)
    return ssim_map.mean(dim=[1, 2, 3])

def kl_divergence_mu_logvar(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

def vae_forward(model, x):
    recon, mu, logvar = model(x)
    std = logvar.mul(0.5).exp_()
    eps = torch.empty_like(std).normal_()
    z = eps.mul(std).add_(mu)
    return recon, mu, logvar, z

def evaluate_model_on_loader(model, loader, split_name="test"):
    model.eval()
    all_targets, all_mse, all_psnr, all_ssim, all_kl, all_z, all_mu, all_logvar = [], [], [], [], [], [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            recon, mu, logvar, z = vae_forward(model, x)
            recon = recon.clamp(0, 1)
            mse_vals = mse_per_image(x, recon)
            psnr_vals = psnr_from_mse(mse_vals)
            ssim_vals = ssim_batch(x, recon)
            kl_vals = kl_divergence_mu_logvar(mu, logvar)

            all_targets.append(y.cpu())
            all_mse.append(mse_vals.cpu())
            all_psnr.append(psnr_vals.cpu())
            all_ssim.append(ssim_vals.cpu())
            all_kl.append(kl_vals.cpu())
            all_z.append(z.cpu())
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())

    targets = torch.cat(all_targets)
    mse_all = torch.cat(all_mse)
    psnr_all = torch.cat(all_psnr)
    ssim_all = torch.cat(all_ssim)
    kl_all = torch.cat(all_kl)
    z_all = torch.cat(all_z)
    mu_all = torch.cat(all_mu)
    logvar_all = torch.cat(all_logvar)

    metrics = {
        "targets": targets,
        "mse_per_image": mse_all,
        "psnr_per_image": psnr_all,
        "ssim_per_image": ssim_all,
        "kl_per_image": kl_all,
        "z": z_all,
        "mu": mu_all,
        "logvar": logvar_all,
        "mse_mean": mse_all.mean().item(),
        "mse_std": mse_all.std().item(),
        "psnr_mean": psnr_all.mean().item(),
        "psnr_std": psnr_all.std().item(),
        "ssim_mean": ssim_all.mean().item(),
        "ssim_std": ssim_all.std().item(),
        "kl_mean": kl_all.mean().item(),
        "kl_std": kl_all.std().item(),
    }

    mu_std_per_dim = mu_all.numpy().std(axis=0)
    metrics["latent_std_global"] = float(mu_std_per_dim.mean())
    metrics["latent_std_min"] = float(mu_std_per_dim.min())
    metrics["latent_std_max"] = float(mu_std_per_dim.max())

    print(f"\n#### {split_name.upper()} RECON / KL ####")
    print(f"MSE mean±std: {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"PSNR mean±std: {metrics['psnr_mean']:.3f} ± {metrics['psnr_std']:.3f}")
    print(f"SSIM mean±std: {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
    print(f"KL mean±std: {metrics['kl_mean']:.4f} ± {metrics['kl_std']:.4f}")
    return metrics

def run_svm_on_latent(z_tensor, y_tensor, C=1.0, kernel="rbf"):
    z_np = z_tensor.numpy()
    y_np = y_tensor.numpy()
    n = z_np.shape[0]
    idx = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, X_test = z_np[train_idx], z_np[test_idx]
    y_train, y_test = y_np[train_idx], y_np[test_idx]
    clf = SVC(C=C, kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
    print(f"SVM latent accuracy: {acc:.4f}")
    return acc, cm

def paired_ttest_metric(metric_a, metric_b, alpha=0.05, name_a="modelA", name_b="modelB"):
    a = np.array(metric_a)
    b = np.array(metric_b)
    t_stat, p_val = stats.ttest_rel(a, b)
    print(f"Paired t-test {name_a} vs {name_b}: t={t_stat:.4f}, p={p_val:.6e}")
    return t_stat, p_val

def save_summary_csv(summary_dict, filename="vae_evaluation_summary.csv"):
    df = pd.DataFrame.from_dict(summary_dict, orient="index")
    df.to_csv(filename)
    print("Saved summary CSV to", filename)


################################ Main evaluation ################################
if __name__ == "__main__":
    models = {"model0": model0, "model1": model1, "model2": model2}
    summary = {}
    test_metrics_all = {}
    holdout_metrics_all = {}

    ######################## EVALUATION METRICES ########################

    for name, model in models.items():
        print("\n###################################")
        print("Evaluating", name, "on TEST")
        
        test_metrics = evaluate_model_on_loader(model, test_dataloader, split_name=f"{name}_test")
        test_metrics_all[name] = test_metrics

        print("\n####################################")
        print("Evaluating", name, "on HOLDOUT")
        
        holdout_metrics = evaluate_model_on_loader(model, holdout_dataloader, split_name=f"{name}_holdout")
        holdout_metrics_all[name] = holdout_metrics


        print("\nSVM on latent (test) for", name)
        svm_acc_test, svm_cm_test = run_svm_on_latent(test_metrics["z"], test_metrics["targets"])


        visuals.show_reconstructions_packed( model, test_dataloader, holdout_dataloader, device,title_prefix=name,filename=f"{name}_recon_test_holdout.png")
        visuals.tsne_and_confusion(name,test_metrics["z"],test_metrics["targets"],svm_cm_test,filename=f"{name}_tsne_confusion.png")
        visuals.generate_samples_from_prior(model, device, latent_dim=model.latent_dims,title=f"{name} prior samples",filename=f"{name}_prior_samples.png")


        summary[name] = {
            "test_mse_mean": test_metrics["mse_mean"],
            "test_psnr_mean": test_metrics["psnr_mean"],
            "test_ssim_mean": test_metrics["ssim_mean"],
            "test_kl_mean": test_metrics["kl_mean"],
            "test_latent_std_global": test_metrics["latent_std_global"],
            "test_svm_acc_latent": svm_acc_test,
            "holdout_mse_mean": holdout_metrics["mse_mean"],
            "holdout_psnr_mean": holdout_metrics["psnr_mean"],
            "holdout_ssim_mean": holdout_metrics["ssim_mean"],
            "holdout_kl_mean": holdout_metrics["kl_mean"],
            "holdout_latent_std_global": holdout_metrics["latent_std_global"],
        }
        
    
    visuals.grouped_bar_comparison( summary, metric_keys=["test_mse_mean"], split_name="test", filename="comparison_test_mse_only.png")

    metric_keys = ["test_psnr_mean", "test_ssim_mean", "test_kl_mean","test_svm_acc_latent"]

    visuals.grouped_bar_comparison(summary,metric_keys,split_name="test",filename="comparison_test_metrics.png")

    ###############################################################################

    pairs = [("model0", "model1"), ("model0", "model2"), ("model1", "model2")]
    print("\n#### Paired t-tests on TEST per-image MSE ####")
    for a, b in pairs:
        mA = test_metrics_all[a]["mse_per_image"]
        mB = test_metrics_all[b]["mse_per_image"]
        paired_ttest_metric(mA, mB, name_a=a, name_b=b)

    print("\n#### Paired t-tests on TEST per-image KL ####")
    for a, b in pairs:
        kA = test_metrics_all[a]["kl_per_image"]
        kB = test_metrics_all[b]["kl_per_image"]
        paired_ttest_metric(kA, kB, name_a=a, name_b=b)


    save_summary_csv(summary, filename="vae_evaluation_summary.csv")

    model_names = list(summary.keys())

    print("\n#### Model ranking by test SVM latent accuracy ####")
    for m in sorted(model_names, key=lambda x: summary[x]["test_svm_acc_latent"], reverse=True):
        print(
            f"{m}: SVM latent acc = {summary[m]['test_svm_acc_latent']:.4f}, "
            f"test MSE = {summary[m]['test_mse_mean']:.6f}, "
            f"test KL = {summary[m]['test_kl_mean']:.4f}"
        )
