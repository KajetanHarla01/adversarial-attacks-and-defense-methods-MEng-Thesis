import torch
import timm
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, CarliniL2Method, DeepFool, ZooAttack
from art.estimators.classification import PyTorchClassifier
from torchvision import datasets
import os
from os import listdir
from os.path import isfile, join
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from collections import defaultdict
from itertools import product
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import threading

class TimeoutExpired(Exception):
    pass

def input_with_timeout(prompt, timeout):
    result = [None]

    def ask():
        result[0] = input(prompt)

    thread = threading.Thread(target=ask)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("(No input received — continuing training...)\n")
        return None
    return result[0]

timm_models = {
    1: 'resnet34.tv_in1k',
    2: 'efficientnet_b0.ra_in1k',
    3: 'densenet121.tv_in1k',
    4: 'convnext_tiny.in12k_ft_in1k_384',
    5: 'vit_small_patch16_384'
}
DEFAULT_PARAMS = {
    'eps': 0.03,
    'eps_step': 0.01,
    'max_iter': 10,
    'nb_grads': 10,
    'confidence': 0.0,
    'initial_const': 0.1,
    'binary_search_steps': 5
}

def get_attack(attack_name, classifier, eps=0.03, targeted=False, max_iter=10, nb_grads=10, eps_step=None, confidence=0.0, initial_const=0.1, binary_search_steps=5):
    if attack_name == 'FGSM':
        return FastGradientMethod(estimator=classifier, eps=denormalize_epsilon(eps, model_std), targeted=targeted, verbose=False)
    elif attack_name == 'PGD':
        return ProjectedGradientDescent(estimator=classifier, eps=denormalize_epsilon(eps, model_std), eps_step=denormalize_epsilon(eps_step, model_std), max_iter=max_iter, targeted=targeted, verbose=False)
    elif attack_name == 'CW':
        return CarliniL2Method(classifier=classifier, targeted=targeted, max_iter=max_iter, confidence=confidence, initial_const=initial_const, binary_search_steps=binary_search_steps, verbose=False)
    elif attack_name == 'DeepFool':
        return DeepFool(classifier=classifier, max_iter=max_iter, verbose=False, nb_grads=nb_grads)
    elif attack_name == 'ZOO':
        return ZooAttack(classifier=classifier, max_iter=max_iter, nb_parallel=1, verbose=False, batch_size=1)
    else:
        raise ValueError(f"Unsupported attack: {attack_name}")

def set_transform(model, input_size_override=384):
    global transform, model_mean, model_std
    config = resolve_data_config({}, model=model, pretrained_cfg=getattr(model, 'pretrained_cfg', None))
    config['input_size'] = (3, input_size_override, input_size_override)
    model_mean = config['mean']
    model_std = config['std']
    transform = create_transform(**config)
    print("[DEBUG] Used config:", config)

def denormalize(img, mean, std):
    img = img.clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    return img

def denormalize_epsilon(eps_normalized, std):
    if isinstance(eps_normalized, float) or isinstance(eps_normalized, int):
        std_tensor = torch.tensor(std, dtype=torch.float32)
        return float(torch.mean(eps_normalized * std_tensor))
    elif isinstance(eps_normalized, list):
        std_tensor = torch.tensor(std, dtype=torch.float32)
        eps_tensor = torch.tensor(eps_normalized, dtype=torch.float32)
        return (eps_tensor.unsqueeze(1) * std_tensor).mean(dim=1).tolist()
    else:
        raise ValueError("eps_normalized must be a float, int, or list of floats")

def compute_clip_values_from_mean_std(mean, std):
    min_vals = [(0.0 - m) / s for m, s in zip(mean, std)]
    max_vals = [(1.0 - m) / s for m, s in zip(mean, std)]
    clip_min = float(min(min_vals))
    clip_max = float(max(max_vals))
    return (clip_min, clip_max)

def save_adversarial_example(img_orig, img_adv, class_idx, image_idx, classifier, psnr_val):
    import imageio
    import torch.nn.functional as F

    os.makedirs("saved_adv_examples", exist_ok=True)
    orig_path = f"saved_adv_examples/original_class{class_idx}_img{image_idx}.png"
    adv_path = f"saved_adv_examples/adversarial_class{class_idx}_img{image_idx}.png"
    noise_path = f"saved_adv_examples/noise_class{class_idx}_img{image_idx}.png"

    imageio.imwrite(orig_path, (img_orig * 255).astype(np.uint8))
    imageio.imwrite(adv_path, (img_adv * 255).astype(np.uint8))
    noise = np.clip(img_adv - img_orig + 0.5, 0, 1)
    imageio.imwrite(noise_path, (noise * 255).astype(np.uint8))

    print(f"[INFO] Images saved:\n - {orig_path}\n - {adv_path}\n - {noise_path}")
    print(f"[INFO] PSNR: {psnr_val:.2f} dB")

    img_orig_tensor = torch.tensor(img_orig.transpose(2, 0, 1)).unsqueeze(0).float().to(classifier._device)
    img_adv_tensor = torch.tensor(img_adv.transpose(2, 0, 1)).unsqueeze(0).float().to(classifier._device)

    for i in range(3):
        img_orig_tensor[0, i] = (img_orig_tensor[0, i] - model_mean[i]) / model_std[i]
        img_adv_tensor[0, i] = (img_adv_tensor[0, i] - model_mean[i]) / model_std[i]

    with torch.no_grad():
        logits_orig = torch.tensor(classifier.predict(img_orig_tensor.cpu().numpy())).squeeze()
        logits_adv = torch.tensor(classifier.predict(img_adv_tensor.cpu().numpy())).squeeze()

        probs_orig = F.softmax(logits_orig, dim=0).cpu().numpy()
        probs_adv = F.softmax(logits_adv, dim=0).cpu().numpy()

    top3_orig = np.argsort(probs_orig)[::-1][:3]
    top3_adv = np.argsort(probs_adv)[::-1][:3]

    print("\nTop-3 predictions for original image (softmax):")
    for rank, idx in enumerate(top3_orig, start=1):
        print(f" {rank}. Class {idx} – Probability: {probs_orig[idx]:.4f}")

    print("\nTop-3 predictions for adversarial image (softmax):")
    for rank, idx in enumerate(top3_adv, start=1):
        print(f" {rank}. Class {idx} – Probability: {probs_adv[idx]:.4f}")

def get_file(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
    if len(files) == 0:
        print("Error: \"" + dir + "\" directory is empty")
        exit()
    print("Choose file: ")
    for i, f in enumerate(files, start=1):
        print(f"{i} - {f}")
    file_choose = int(input("File number: "))
    if file_choose > len(files):
        print("Error: invalid file number")
        exit()
    return join(dir, files[file_choose - 1])

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

def create_subset_per_class(dataset, samples_per_class=10):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)
    selected_indices = [idx for indices in class_indices.values() for idx in indices]
    return torch.utils.data.Subset(dataset, selected_indices)

def parse_param_list(input_str, cast_fn=float):
    return [cast_fn(x.strip()) for x in input_str.split(',') if x.strip() != '']

def get_param_combinations(params_dict):
    keys = params_dict.keys()
    values = [v if isinstance(v, list) else [v] for v in params_dict.values()]
    combos = [dict(zip(keys, combination)) for combination in product(*values)]
    return combos

def get_params(attack_type):
    print("\n[INFO] Choose how to define test parameters:")
    print("1 - all combinations of the given values")
    print("2 - manually define individual parameter sets")
    mode = input("Choose mode (1/2): ").strip()
    if mode == '2':
        n = int(input("How many parameter sets do you want to define? "))
        all_combos = []
        for i in range(n):
            print(f"\n[Set {i+1}] Enter parameters:")
            params = ask_for_params(attack_type)
            all_combos.append(get_param_combinations(params)[0])
        return all_combos
    else:
        params = ask_for_params(attack_type)
        return get_param_combinations(params)

def ask_for_params(attack_type):
    params = {}

    if attack_type in ['FGSM', 'PGD']:
        eps_input = input(f"Epsilon (default {DEFAULT_PARAMS['eps']}). Leave empty for default: ").strip()
        if eps_input:
            params['eps'] = parse_param_list(eps_input)

    if attack_type == 'PGD':
        eps_step_input = input(f"Epsilon step (default {DEFAULT_PARAMS['eps_step']}). Leave empty for default: ").strip()
        if eps_step_input:
            params['eps_step'] = parse_param_list(eps_step_input)

    if attack_type in ['CW', 'DeepFool', 'ZOO', 'PGD']:
        iter_input = input(f"Max iterations (default {DEFAULT_PARAMS['max_iter']}). Leave empty for default: ").strip()
        if iter_input:
            params['max_iter'] = parse_param_list(iter_input, int)

    if attack_type == 'CW':
        conf_input = input(f"Confidence (default {DEFAULT_PARAMS['confidence']}). Leave empty for default: ").strip()
        if conf_input:
            params['confidence'] = parse_param_list(conf_input)

        const_input = input(f"Initial const (default {DEFAULT_PARAMS['initial_const']}). Leave empty for default: ").strip()
        if const_input:
            params['initial_const'] = parse_param_list(const_input)

        binary_input = input(f"Binary search steps (default {DEFAULT_PARAMS['binary_search_steps']}). Leave empty for default: ").strip()
        if binary_input:
            params['binary_search_steps'] = parse_param_list(binary_input, int)

    if attack_type == 'DeepFool':
        grads_input = input(f"Number of classes (nb_grads) (default {DEFAULT_PARAMS['nb_grads']}). Leave empty for default: ").strip()
        if grads_input:
            params['nb_grads'] = parse_param_list(grads_input, int)

    return params

def run_attack_on_dataset(model, dataloader, attack_type='FGSM', user_params={}, show_predictions=False):
    device = torch.device("cuda")
    model.to(device)
    clip_values = compute_clip_values_from_mean_std(model_mean, model_std)
    classifier = PyTorchClassifier(
        model=model,
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.0001),
        input_shape=(3, 384, 384),
        nb_classes=40,
        clip_values=clip_values,
        preprocessing=None,
        device_type="gpu"
    )
    attack_params = {**DEFAULT_PARAMS, **user_params}
    print(f"\n[INFO] Running attack {attack_type} with parameters: {user_params}")
    attack = get_attack(
        attack_name=attack_type,
        classifier=classifier,
        **attack_params
    )
    run_single_attack(classifier, dataloader, attack, show_predictions)
    print(f"\n[FINISHED] Attack: {attack_type}")
    print("[PARAMETERS]:")
    for k, v in user_params.items():
        print(f"  {k} = {v}")

def run_single_attack(classifier, data_loader, attack, show_predictions):
    all_preds_adv = []
    all_labels = []
    all_ssim = []
    all_psnr = []
    total_images = len(data_loader.dataset)

    save_example = False
    stop = input_with_timeout("Do you want to save a specific adversarial example? (y/N): ", 20)
    if stop and stop.strip().lower() == 'y':
        save_example = True
    selected_class = None
    selected_idx_in_class = None
    if save_example:
        selected_class = int(input("Enter class index to save (e.g., 5): "))
        selected_idx_in_class = int(input("Enter image number within class (starting from 0): "))

    class_counters = defaultdict(int)
    global_idx = 0

    with tqdm(total=total_images, desc="Processing images") as pbar:
        for images, labels in data_loader:
            images = images.to(classifier._device)
            labels = labels.to(classifier._device)
            adv_images = attack.generate(x=images.cpu().numpy())
            preds_adv = classifier.predict(adv_images)
            adv_images = torch.tensor(adv_images).to(classifier._device)
            preds_adv_classes = torch.tensor(preds_adv).argmax(dim=1)
            all_preds_adv.extend(preds_adv_classes.tolist())
            all_labels.extend(labels.cpu().tolist())
            images_np = images.cpu().numpy()
            adv_images_np = adv_images.cpu().numpy()

            for i, (orig, adv) in enumerate(zip(images_np, adv_images_np)):
                label = labels[i].item()
                class_counters[label] += 1

                img_orig = denormalize(torch.tensor(orig), model_mean, model_std).permute(1, 2, 0).numpy()
                img_orig = np.clip(img_orig, 0, 1)
                img_adv = denormalize(torch.tensor(adv), model_mean, model_std).permute(1, 2, 0).numpy()
                img_adv = np.clip(img_adv, 0, 1)

                if save_example and label == selected_class and (class_counters[label] - 1) == selected_idx_in_class:
                    save_adversarial_example(img_orig, img_adv, selected_class, selected_idx_in_class, classifier, psnr_val)


                if preds_adv_classes[i].item() != label:
                    ssim_val = ssim(img_orig, img_adv, data_range=1.0, channel_axis=-1)
                    psnr_val = psnr(img_orig, img_adv, data_range=1.0)
                    all_ssim.append(ssim_val)
                    all_psnr.append(psnr_val)

                if show_predictions:
                    preds_orig = torch.argmax(classifier.predict(images), axis=1)
                    print(f"Image {global_idx}: {preds_orig[i]} -> {preds_adv_classes[i]}")

                global_idx += 1
            pbar.update(images.shape[0])

    acc_adv = accuracy_score(all_labels, all_preds_adv)
    f1_adv = f1_score(all_labels, all_preds_adv, average='macro')
    precision_adv = precision_score(all_labels, all_preds_adv, average='macro', zero_division=1)
    recall_adv = recall_score(all_labels, all_preds_adv, average='macro', zero_division=1)

    print(f"Adversarial Accuracy: {acc_adv * 100:.2f}%, F1: {f1_adv * 100:.2f}%, Precision: {precision_adv * 100:.2f}%, Recall: {recall_adv * 100:.2f}%")
    print(f"Mean SSIM: {np.mean(all_ssim):.4f}, Mean PSNR: {np.mean([val for val in all_psnr if np.isfinite(val)]):.2f} dB")

if __name__ == '__main__':
    print("Choose a model:")
    model_filename = get_file("models")
    device = torch.device("cuda")
    torch.set_num_threads(12)
    os.environ["OMP_NUM_THREADS"] = "12"
    type = os.path.basename(model_filename).split('_')[0]
    model_class = get_model_by_type(type)
    if model_class:
        model = model_class()
        model = load_model(model, model_filename, device)
        set_transform(model)
        test_data = datasets.ImageFolder(root='testing', transform=transform)

        attack_options = ['FGSM', 'PGD', 'CW', 'DeepFool', 'ZOO']
        print("\nChoose an attack:")
        for i, name in enumerate(attack_options):
            print(f"{i + 1}. {name}")
        attack_choice = int(input("Your choice (1-5): ")) - 1
        attack_name = attack_options[attack_choice]

        samples_per_class = int(input("\nHow many samples per class? (e.g., 10): "))

        class_names = test_data.classes
        print("\n[Optional] Choose classess for testing (np. 0,2,5) or press Enter to use all.")
        class_input = input("Classes (Enter = all): ").strip()

        if class_input:
            selected_classes = set(int(x.strip()) for x in class_input.split(',') if x.strip().isdigit())
            class_indices = defaultdict(list)
            for idx, (_, label) in enumerate(test_data.samples):
                if label in selected_classes and len(class_indices[label]) < samples_per_class:
                    class_indices[label].append(idx)
            selected_indices = [idx for indices in class_indices.values() for idx in indices]
            dataset = torch.utils.data.Subset(test_data, selected_indices)
        else:
            dataset = create_subset_per_class(test_data, samples_per_class=samples_per_class)

        show_predictions_input = input("Show predictions (y/N)?").strip().lower()
        show_predictions = show_predictions_input == 'y'

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
        param_combos = get_params(attack_name)

        for i, combo in enumerate(param_combos):
            print(f"\n[TEST {i+1}/{len(param_combos)}] Using parameters: {combo}")
            run_attack_on_dataset(model, dataloader, attack_type=attack_name, user_params=combo, show_predictions=show_predictions)

    else:
        print("Unknown model type: ", type)