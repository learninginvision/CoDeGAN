# CoDeGAN

Disentanglement, as an important issue of interpretable AI, has attracted vast attention of computer vision community. In contrast to VAE-based disentanglement, GAN-based disentanglement is extremely hard as GANs do not have explicit sample likelihood and forbidden posterior inference. Most existing studies, such as InfoGAN and its variants, maximize the mutual information (MI) between an image and its latent codes to disentangle image variation in a unsupervised manner. A major problem of these methods is that they force the network to generate the same images for the same latent factor and thus may seriously destroy the equilibrium of GANs. To alleviate this problem, we propose **Co**ntrastive **D**is**e**ntanglement for Generative **A**dversarial Networks (**CoDeGAN**), where we relax the domain of similarity constraints to be the feature domain rather than the image domain, so as to improve GANs equilibrium and constrain disentanglement. Besides, we provide a theoretical analysis of why CoDeGAN can effectively alleviate GANs disequilibrium. Furthermore, we introduce self-supervised pre-training into CoDeGAN for learning semantic representation to guide unsupervised disentanglement. The extensive experimental results have shown that our method achieves the SOTA performance on multiple benchmarks.

## Approach



![](./Figure/structure.png)
=======

## Dependencies

- Python 3.6.13
- Pytorch 1.7.1
- Numpy 1.19.2
- TensorFlow 1.12.0

## Datasets

The MNIST, Fashion-MNIST and CIFAR-10 datasets needn't to be downloaded in advance, the code automatically downloads the data to directory ``"./CoDeGAN/data/<dataset_name>"`` during operation , if you download the data yourself, make sure they are on the same directory.

## Training

You can train your own models on the datasets mentioned, the few labels used in our experiments and the corresponding pretrained models are saved in directory ``"../CoDeGAN/<dataset_name>/few_labels"``, they are used only in few labels experiments. When selecting few labels images, we only make sure that the number of images for each class is equal, without additional filtering.

We also provide [pre trained model weights](https://drive.google.com/drive/folders/1KrIAhsEd3BOKAZOPIHJY3MW9-kw3oAgS?usp=sharing), you can download this file and unzip to base directory.

Each model can be trained by the following formats:

- **MNIST**

  Changing directory to :

  ```bash
  cd ./CoDeGAN/MNIST/
  ```

  Opening ``param.yml`` to set hyparameters:

  ```python
  beta_1: 75 
  beta_2: 0.0005
  ```

  Runing:

  ```bash
  $ python train.py
  ```

- **Fashion-MNIST**

  Changing directory to:

  ```bash
  cd ./CoDeGAN/Fashion-MNIST/
  ```

  Opening ``param.yml`` to set hyparameters:

  - For unsupervised experiments:

  ```python
  beta_1: 100
  beta_2: 0.0005
  max_step: 600
  chain_step: 0
  few_labels: False
  pretrain: False
  meta: False
  ```

  - For unsupervised witch pretraining experiments:

  ```python
  beta_1: 100
  beta_2: 0.0005
  max_step: 600
  chain_step: 200
  few_labels: False
  pretrain: True
  meta: False
  ```

  - For few labels experiments(100 labels):

  ```python
  beta_1: 100
  beta_2: 0.0005
  max_step: 600
  chain_step: 0
  few_labels: True
  pretrain: False
  meta: False
  ```

  - For few labels with meta pretraining experiments(100 labels):

  ```python
  beta_1: 100
  beta_2: 0.0005
  max_step: 600
  chain_step: 200
  few_labels: True
  pretrain: True
  meta: True
  ```

  - For few labels with pretraining experiments(100 labels):

  ```python
  beta_1: 100
  beta_2: 0.0005
  max_step: 300
  chain_step: 200
  few_labels: True
  pretrain: True
  meta: False
  ```

  Runing:

  ```bash
  $ python train.py
  ```

- **CIFAR-10**

  Changing directory to:

  ```bash
  cd ./CoDeGAN/CIFAR-10/
  ```

  Opening ``param.yml`` to set hyparameters:

  - For unsupervised experiments:

  ```python
  beta_1: 10
  beta_2: 5
  chain_step: 0
  max_step  : 350
  few_labels: False
  pretrain: False
  meta: False
  ```

  - For unsupervised with pretrain experiments:

  ```python
  beta_1: 1.5
  beta_2: 0.5
  chain_step: 0
  max_step  : 300
  few_labels: False
  pretrain: True
  meta: False
  ```

  - For few labels experiments(500 labels):

  ```python
  beta_1: 1
  beta_2: 0.5
  chain_step: 0
  max_step  : 350
  few_labels: True
  pretrain: False
  meta: False
  ```

  - For few labels with meta pretraining experiments(500 labels):

  ```python
  beta_1: 0.25
  beta_2: 2.5
  chain_step: 100
  max_step  : 300
  few_labels: True
  pretrain: True
  meta: True
  ```

  - For few labels with pretraining experiments(500 labels):

  ```python
  beta_1: 0.25
  beta_2: 0.5
  chain_step: 100
  max_step  : 250
  few_labels: True
  pretrain: True
  meta: False
  ```

  Runing:

  ```bash
  $ python train.py
  ```

**COIL-20**

Changing directory to:

```bash
cd ./CoDeGAN/COIL-20/
```

Opening ``param.yml`` to set hyparameters:

- For unsupervised experiments:

```python
beta_1: 150
beta_2: 0.0005
chain_step: 0
max_step  : 4500
pretrain: False
```

- For unsupervised with pretrain experiments:

```python
beta_1: 100
beta_2: 0.0005
chain_step: 1500
max_step  : 4500
pretrain: True
```

- Runing:

  ```bash
  $ python train.py
  ```

**3D-Chairs**

Changing directory to:

```bash
cd ./CoDeGAN/3D-Chairs/
```

Opening ``param.yml`` to set hyparameters:

- For unsupervised experiments:

```python
beta_1: 100
beta_2: 0.0005
chain_step: 0
max_step  : 3000
pretrain: False
```

- For unsupervised with pretrain experiments:

```python
beta_1: 75
beta_2: 0.0005
chain_step: 1000
max_step  : 3000
pretrain: True
```

- Runing:

  ```bash
  $ python train.py
  ```

**3D-Cars**

Changing directory to:

```bash
cd ./CoDeGAN/3D-Cars/
```

Opening ``param.yml`` to set hyparameters:

- For unsupervised experiments:

```python
beta_1: 100
beta_2: 0.0005
chain_step: 0
max_step  : 25000
pretrain: False
```

- For unsupervised with pretrain experiments:

```python
beta_1: 80
beta_2: 0.0005
chain_step: 200
max_step  : 2500
pretrain: True
```

- Runing:

  ```bash
  python train.py
  ```

## Testing

- **ACC, NMI, ARI**

  The ACC, NMI, ARI are calculated by ``./CoDeGAN/<dataset_name>/test.py`` in MNIST and Fashion-MNIST, ``./CoDeGAN/<dataset_name>/test/test_acc.py`` in CIFAR-10, which will be calculated automatically during training, the test result will be saved in  ``./CoDeGAN/<dataset_name>/result/<rand int>/test_result.txt``.

- **IS, FID**

  For IS and FID testing, we follow the work of [LDAGAN](https://github.com/Sumching/LDAGAN), the code is written by TensorFlow, if you want to calculate IS and FID score for CIFAR-10 experiments, you can do it by the following steps:

  Downloading 

  Opening  ``./CoDeGAN/CIFAR-10/utils/sample_fake_images2npy.py``

  Setting ``model_dir`` to be the directory where the final trained models are saved.

  Running:

  In Pytorch environment:

  ```bash
  cd ./CoDeGAN/CIFAR-10/utils/
  $ python sample_fake_images2npy.py
  ```

  In TensorFlow environment:

  ```bash
  cd ./CoDeGAN/CIFAR-10/test/
  $ python test_IS&FID.py
  ```

## Result

![](./Figure/codegan.png)

<center></center>Qualitative comparison with state-of-the-art methods on the Fashion-MNIST, COIL-20 and CIFAR-10 datasets.</center>





