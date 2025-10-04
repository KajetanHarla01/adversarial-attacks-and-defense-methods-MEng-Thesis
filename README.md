<h1>Adversarial Attacks and Defense Methods in Image Recognition Systems</h1>

<p>
This repository contains the experimental framework developed for my master's thesis on
<strong>adversarial robustness</strong> in image classification models.
The project systematically evaluates multiple <strong>adversarial attack algorithms</strong>,
<strong>defensive training</strong>, and <strong>detection methods</strong> across several deep learning architectures
using the <a href="https://pytorch.org/">PyTorch</a>,
<a href="https://huggingface.co/docs/timm/">timm</a>,
and <a href="https://github.com/Trusted-AI/adversarial-robustness-toolbox">Adversarial Robustness Toolbox (ART)</a> libraries.
</p>

<hr>

<h2>Overview</h2>

<p>
The goal of this project is to explore the <strong>vulnerability of deep convolutional and transformer-based architectures</strong>
to adversarial perturbations and to assess <strong>defense mechanisms</strong> designed to mitigate such attacks.
Experiments were conducted primarily on the <strong>Stanford 40 Human Actions</strong> dataset,
though the code supports any folder-structured image dataset. Link to download the dataset: http://vision.stanford.edu/Datasets/40actions.html
</p>

<h3>Implemented Components</h3>

<table>
  <thead>
    <tr><th>Category</th><th>Description</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>Model Training</strong></td><td>Training and validation scripts for models such as ResNet-34, EfficientNet-B0, DenseNet-121, ConvNeXt-Tiny, and ViT-Small.</td></tr>
    <tr><td><strong>Adversarial Attacks</strong></td><td>Implementation and testing of FGSM, PGD, CW (Carlini–Wagner), DeepFool, and ZOO attacks with adjustable parameters.</td></tr>
    <tr><td><strong>Adversarial Training</strong></td><td>Robust retraining using FGSM and PGD examples.</td></tr>
    <tr><td><strong>Input Transformations</strong></td><td>Median filtering and JPEG compression applied to adversarial images.</td></tr>
    <tr><td><strong>Binary Logit Detector</strong></td><td>A secondary neural network trained on model logits to distinguish between clean and adversarial inputs.</td></tr>
    <tr><td><strong>Metrics</strong></td><td>Accuracy, Precision, Recall, F1, PSNR, SSIM, AUC, and Balanced Accuracy.</td></tr>
  </tbody>
</table>

<hr>

<h2>Repository Structure</h2>

<pre>
├── train_models.py            # Training and validation of base models
├── attacks.py                 # Generation and evaluation of adversarial examples
├── adversarial_training.py    # FGSM and PGD-based adversarial training
├── input_transformations.py   # Defensive transformations (median, JPEG)
├── binary_classifier.py       # Logit-based adversarial detection model
├── models/                    # Directory for saved model checkpoints
├── AT-models/                 # Directory for saved Adversarial Training models
├── training/ testing/ ...     # Dataset folders (ImageFolder structure)
└── README.md
</pre>

<hr>

<h2>Setup</h2>

<h3>Requirements</h3>
<ul>
  <li>Python 3.10+</li>
  <li>PyTorch (with CUDA support)</li>
  <li>timm</li>
  <li>torchvision</li>
  <li>adversarial-robustness-toolbox (ART)</li>
  <li>scikit-learn</li>
  <li>scikit-image</li>
  <li>numpy, tqdm, matplotlib, opencv-python</li>
</ul>

<p>Install all dependencies:</p>

<pre><code>pip install -r requirements.txt</code></pre>

<hr>

<h2>Usage</h2>

<h3>1. Train a Base Model</h3>
<pre><code>python train_models.py</code></pre>
<p>You’ll be prompted to select an architecture and number of epochs.
The script automatically handles dataset splitting, checkpoint saving, and evaluation.</p>

<h3>2. Generate Adversarial Examples</h3>
<pre><code>python attacks.py</code></pre>
<p>Choose a model and attack type (FGSM, PGD, CW, DeepFool, or ZOO).
Metrics (accuracy, F1, PSNR, SSIM) are computed automatically.</p>

<h3>3. Perform Adversarial Training</h3>
<pre><code>python adversarial_training.py</code></pre>
<p>Supports FGSM- and PGD-based robust training with customizable epsilon values.</p>

<h3>4. Apply Input Defenses</h3>
<pre><code>python input_transformations.py</code></pre>
<p>You can test the effect of preprocessing transformations (median, JPEG) on adversarial examples.</p>

<h3>5. Train and Evaluate a Binary Detector</h3>
<pre><code>python binary_classifier.py</code></pre>
<p>
Trains a secondary MLP classifier on model logits to distinguish between clean and adversarial samples.
Reports ROC-AUC, precision-recall metrics, and threshold calibration for specific FPR values.
</p>

<hr>

<h2>Models</h2>

<ul>
  <li><code>resnet34.tv_in1k</code></li>
  <li><code>efficientnet_b0.ra_in1k</code></li>
  <li><code>densenet121.tv_in1k</code></li>
  <li><code>convnext_tiny.in12k_ft_in1k_384</code></li>
  <li><code>vit_small_patch16_384</code></li>
</ul>

<hr>

<h2>Evaluation Metrics</h2>

<table>
  <thead>
    <tr><th>Metric</th><th>Description</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>Accuracy / F1</strong></td><td>Classification performance under attack.</td></tr>
    <tr><td><strong>PSNR / SSIM</strong></td><td>Perceptual similarity between original and adversarial images.</td></tr>
    <tr><td><strong>ROC-AUC / AP</strong></td><td>Detection quality for binary classifiers.</td></tr>
    <tr><td><strong>Attack Success Rate (ASR)</strong></td><td>Fraction of successful misclassifications.</td></tr>
  </tbody>
</table>

<hr>

<h2>Thesis Context</h2>

<p>
This code accompanies the research presented in my
<strong>Master’s Thesis on Adversarial Attacks and Defense Methods in Image Recognition Systems</strong>,
which analyzes:
</p>
<ul>
  <li>The relative robustness of CNN vs. Transformer architectures.</li>
  <li>The effectiveness of different attack strategies (FGSM, PGD, CW, DeepFool, ZOO).</li>
  <li>Defensive approaches: adversarial training, preprocessing transformations, and adversarial samples detection.</li>
</ul>

<h2>License</h2>

<p>This project is released under the <strong>MIT License</strong>.</p>
