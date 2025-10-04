import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.metrics import accuracy_score
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
import numpy as np
import os
from tqdm import tqdm
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_CLASSES = 40

def denormalize_epsilon(eps_normalized, std):
    std_tensor = torch.tensor(std, dtype=torch.float32)
    return float(torch.mean(eps_normalized * std_tensor))

def compute_clip_values(mean, std):
    min_vals = [(0.0 - m) / s for m, s in zip(mean, std)]
    max_vals = [(1.0 - m) / s for m, s in zip(mean, std)]
    return float(min(min_vals)), float(max(max_vals))

def get_file(dir_path):
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    print("\nChoose a model:")
    for i, f in enumerate(files):
        print(f"{i+1}. {f}")
    idx = int(input("File number: ")) - 1
    return os.path.join(dir_path, files[idx])

def get_model_arch(filename):
    name = os.path.basename(filename).lower()
    if "resnet" in name:
        return "resnet34.tv_in1k"
    elif "efficientnet" in name:
        return "efficientnet_b0.ra_in1k"
    elif "densenet" in name:
        return "densenet121.tv_in1k"
    elif "convnext" in name:
        return "convnext_tiny.in12k_ft_in1k_384"
    elif "visiontransformer" in name:
        return "vit_small_patch16_384"
    else:
        raise ValueError("Unknown model.")

def train_fgsm(model, train_loader, classifier, epsilons, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    training_log = []

    for epoch in range(epochs):
        epsilon = np.random.choice(epsilons)
        print(f"[FGSM] Epoch {epoch+1}/{epochs} | ε = {epsilon:.4f}")
        attack = FastGradientMethod(estimator=classifier, eps=denormalize_epsilon(epsilon, model_std))
        model.train()
        total_loss = 0
        batch_count = 0

        for x, y in tqdm(train_loader, desc=f"FGSM Epoch {epoch+1}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = torch.tensor(attack.generate(x=x.cpu().numpy())).to(DEVICE)
            x_comb = torch.cat([x, x_adv])
            y_comb = torch.cat([y, y])
            optimizer.zero_grad()
            out = model(x_comb)
            loss = loss_fn(out, y_comb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"[FGSM] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        training_log.append((epoch+1, epsilon, avg_loss))

def train_pgd(model, train_loader, classifier, epsilons, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    training_log = []

    for epoch in range(epochs):
        epsilon = np.random.choice(epsilons)
        print(f"[PGD] Epoch {epoch+1}/{epochs} | ε = {epsilon:.4f}")
        attack = ProjectedGradientDescent(
            estimator=classifier,
            eps=denormalize_epsilon(epsilon, model_std),
            eps_step=denormalize_epsilon(0.01, model_std),
            max_iter=10
        )
        model.train()
        total_loss = 0
        batch_count = 0
        for x, y in tqdm(train_loader, desc=f"PGD Epoch {epoch+1}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_adv = torch.tensor(attack.generate(x=x.cpu().numpy())).to(DEVICE)
            x_comb = torch.cat([x, x_adv])
            y_comb = torch.cat([y, y])
            optimizer.zero_grad()
            out = model(x_comb)
            loss = loss_fn(out, y_comb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"[FGSM] Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")
        training_log.append((epoch+1, epsilon, avg_loss))

def test_model(model, test_loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test"):
            x = x.to(DEVICE)
            outputs = model(x)
            preds.extend(outputs.argmax(dim=1).cpu().numpy())
            labels.extend(y.numpy())
    acc = accuracy_score(labels, preds)
    print(f"[TEST] Accuracy: {acc * 100:.2f}%")

def set_transform(model, input_size_override=384):
    global transform, model_mean, model_std
    config = resolve_data_config({}, model=model, pretrained_cfg=getattr(model, 'pretrained_cfg', None))
    config['input_size'] = (3, input_size_override, input_size_override)
    model_mean = config['mean']
    model_std = config['std']
    transform = create_transform(**config)
    print("[DEBUG] Used config:", config)

def get_model_by_type(type):
    model_names = {
        'resnet': 'resnet34.tv_in1k',
        'efficientnet': 'efficientnet_b0.ra_in1k',
        'densenet': 'densenet121.tv_in1k',
        'convnext': 'convnext_tiny.in12k_ft_in1k_384',
        'visiontransformer': 'vit_small_patch16_384'
    }
    if type in model_names:
        return lambda: timm.create_model(model_names[type], pretrained=True, num_classes=40)
    return None

def save_model(model, filename):
    save_dict = {
        'state_dict': model.state_dict()
    }
    if hasattr(model, 'pretrained_cfg'):
        save_dict['pretrained_cfg'] = model.pretrained_cfg
    torch.save(save_dict, filename)
    print(f"Model and config saved to {filename}")

def load_model(model, filename, device):
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' does not exist.")
        return None

    try:
        checkpoint = torch.load(filename, map_location=device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        if 'pretrained_cfg' in checkpoint and hasattr(model, 'pretrained_cfg'):
            model.pretrained_cfg = checkpoint['pretrained_cfg']
        model.to(device)
        print(f"Model loaded from {filename}")
    except Exception as e:
        print(f"[ERROR] Cannot load the model: {e}")
        return None

    return model

if __name__ == "__main__":
    torch.set_num_threads(12)
    os.environ["OMP_NUM_THREADS"] = "12"
    print("=== Adversarial Training ===")

    model_path = get_file("models")
    type = os.path.basename(model_path).split('_')[0]
    model_class = get_model_by_type(type)
    if model_class:
        model = model_class()
        model = load_model(model, model_path, DEVICE)
        set_transform(model)

        train_dataset = datasets.ImageFolder("training", transform=transform)
        test_dataset = datasets.ImageFolder("testing", transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        state_dict = torch.load(model_path, map_location=DEVICE)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE)

        eps_input = input("\nInput epsilon(s), np. 0.01,0.03,0.05: ")
        epsilons = [float(e.strip().replace(",", ".")) for e in eps_input.split(",") if e.strip()]
        multi = len(epsilons) > 1
        epochs = int(input("Number of epochs: "))

        clip_values = compute_clip_values(model_mean, model_std)
        classifier = PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
            input_shape=(3, 384, 384),
            nb_classes=NUM_CLASSES,
            clip_values=clip_values,
            preprocessing=None,
            device_type="gpu"
        )

        print("\nChoose adversarial training type:")
        print("1 - FGSM")
        print("2 - PGD")
        choice = input("Your choice (1/2): ").strip()

        if choice == "1":
            train_fgsm(model, train_loader, classifier, epsilons, epochs)
            method = "fgsm"
        elif choice == "2":
            train_pgd(model, train_loader, classifier, epsilons, epochs)
            method = "pgd"
        else:
            print("Wrong option.")
            exit()

        os.makedirs("AT_models", exist_ok=True)
        if multi:
            out_path = f"AT_models/{os.path.basename(model_path).split('.')[0]}_{method}_adv_multi.pth"
        else:
            out_path = f"AT_models/{os.path.basename(model_path).split('.')[0]}_{method}_adv.pth"
        save_model(model, out_path)
        print(f"Model loaded to {out_path}")

        test_model(model, test_loader)
    else:
        print("Error: wrong model")