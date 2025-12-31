# visual.py
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torch
import os

PLOT_DIR = "/Users/anjali98/Desktop/SUSSEX_LABS/amml-2526-main/plots_ml"  

os.makedirs(PLOT_DIR, exist_ok=True)

def tsne_and_confusion(model_name, z_tensor, y_tensor, cm,
                       filename=None, n_samples=2000):
    z_np = z_tensor.numpy()
    y_np = y_tensor.numpy()
    if z_np.shape[0] > n_samples:
        idx = np.random.choice(z_np.shape[0], n_samples, replace=False)
        z_np = z_np[idx]
        y_np = y_np[idx]
    tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
    z_2d = tsne.fit_transform(z_np)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # t-SNE
    sc = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=y_np, cmap="tab10", s=5, alpha=0.7)
    cb = fig.colorbar(sc, ax=axes[0], ticks=range(10))
    axes[0].set_title(f"{model_name} t-SNE (test)")
    axes[0].set_xlabel("t-SNE 1")
    axes[0].set_ylabel("t-SNE 2")

    # confusion matrix
    im = axes[1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=axes[1])
    classes = [str(i) for i in range(10)]
    axes[1].set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"{model_name} SVM confusion (test)",
    )
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )

    plt.tight_layout()
    _save_or_show(fig, filename)

def plot_confusion_matrix(cm, classes, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=7,
            )
    fig.tight_layout()
    plt.show()

def run_tsne_and_plot(z_tensor, y_tensor, title_prefix="model_tsne", n_samples=2000):
    z_np = z_tensor.numpy()
    y_np = y_tensor.numpy()
    if z_np.shape[0] > n_samples:
        idx = np.random.choice(z_np.shape[0], n_samples, replace=False)
        z_np = z_np[idx]
        y_np = y_np[idx]
    tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
    z_2d = tsne.fit_transform(z_np)
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(z_2d[:, 0], z_2d[:, 1], c=y_np, cmap="tab10", s=5, alpha=0.7)
    plt.colorbar(sc, ticks=range(10))
    plt.title(f"{title_prefix} t-SNE latent space")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.show()

def to_img(x):
    return x.clamp(0, 1)

def show_reconstructions_packed(model, test_loader, holdout_loader, device,
                                title_prefix="model0", num_images=8,
                                filename=None):

    model.eval()
    # one batch from each loader
    for x_t, _ in test_loader:
        x_test = x_t.to(device)
        break
    for x_h, _ in holdout_loader:
        x_hold = x_h.to(device)
        break

    from main import vae_forward
    with torch.no_grad():
        recon_test, _, _, _ = vae_forward(model, x_test)
        recon_hold, _, _, _ = vae_forward(model, x_hold)

    x_test = x_test[:num_images].cpu()
    r_test = recon_test[:num_images].clamp(0, 1).cpu()
    x_hold = x_hold[:num_images].cpu()
    r_hold = recon_hold[:num_images].clamp(0, 1).cpu()

    # build 4 grids
    grid_test_orig = vutils.make_grid(x_test, nrow=int(np.sqrt(num_images)))
    grid_test_recon = vutils.make_grid(r_test, nrow=int(np.sqrt(num_images)))
    grid_hold_orig = vutils.make_grid(x_hold, nrow=int(np.sqrt(num_images)))
    grid_hold_recon = vutils.make_grid(r_hold, nrow=int(np.sqrt(num_images)))

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    axes[0, 0].imshow(np.transpose(grid_test_orig.numpy(), (1, 2, 0)), cmap="gray")
    axes[0, 0].set_title("Test originals")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(np.transpose(grid_test_recon.numpy(), (1, 2, 0)), cmap="gray")
    axes[0, 1].set_title("Test reconstructions")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(np.transpose(grid_hold_orig.numpy(), (1, 2, 0)), cmap="gray")
    axes[1, 0].set_title("Holdout originals")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.transpose(grid_hold_recon.numpy(), (1, 2, 0)), cmap="gray")
    axes[1, 1].set_title("Holdout reconstructions")
    axes[1, 1].axis("off")

    fig.suptitle(f"{title_prefix}: reconstruction comparison", fontsize=12)
    plt.tight_layout()
    _save_or_show(fig, filename)



def generate_samples_from_prior(model, device, latent_dim=20,
                                num_samples=16, title="Samples from prior",
                                filename=None):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decoder(z)
        samples = samples.clamp(0, 1).cpu()

    grid = vutils.make_grid(samples, nrow=int(np.sqrt(num_samples)))
    fig = plt.figure(figsize=(4, 4))
    plt.title(title)
    plt.imshow(np.transpose(grid.numpy(), (1, 2, 0)), cmap="gray")
    plt.axis("off")
    _save_or_show(fig, filename)


def grouped_bar_comparison(summary, metric_keys, split_name="test", filename=None):

    models = list(summary.keys())           # ['model0','model1','model2']
    n_models = len(models)
    x = np.arange(len(metric_keys))
    width = 0.8 / n_models
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(models):
        y = [summary[m][k] for k in metric_keys]
        ax.bar(x + i*width, y, width=width,
               label=m, color=colors[i % len(colors)])

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_keys, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(f"Model comparison ({split_name})")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    _save_or_show(fig, filename)


def metric_comparison_plot(summary, metric_keys, split_name="test", filename=None):

    models = list(summary.keys())
    x = np.arange(len(metric_keys))
    colors = ["tab:blue", "tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, m in enumerate(models):
        y = [summary[m][k] for k in metric_keys]
        ax.plot(x, y, marker="o", color=colors[i % len(colors)], label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, rotation=45, ha="right")
    ax.set_title(f"Model comparison ({split_name})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, filename)



def _save_or_show(fig, filename=None):
    if filename is not None:
        path = os.path.join(PLOT_DIR, filename)
        fig.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()
