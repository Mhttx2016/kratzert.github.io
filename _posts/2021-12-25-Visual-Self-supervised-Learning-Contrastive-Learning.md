# Visual Self-supervised Learning(Contrastive Learning)


## With negative examples (Contrastive method)

### InstDisc:Unsupervised Feature Learning via Non-Parametric Instance Discrimination

<p align="center">
<img src="/images/Visual-Self-supervised-Learning-assets/image-20220306164622407.png" alt="image-20220306164622407" style="width:100%;" />
</p>

- **Motivation:**learn a good feature representation that captures apparent similarity among instances, instead of classes,

- **How:**
  - memory bank store reprsentation
  - NCE loss + Proximal Regularization
  - <img src="/images/Visual-Self-supervised-Learning-assets/image-20220306165410633.png" alt="image-20220306165410633" style="width:50%;" />
  - <img src="/images/Visual-Self-supervised-Learning-assets/image-20220306165250790.png" alt="image-20220306165250790" style="width:75%;" />
- Adv vs Disadv
  - Memory bank update every epoch

### CPC: Representation Learning withContrastive Predictive Coding
<p align="center">
<img src="/images/Visual-Self-supervised-Learning-assets/image-20220306171843768.png" alt="image-20220306171843768" style="width:75%;" />
</p>

- Motivation:

  - learn representations by predicting the future in latent space by using powerful autoregressive models. We use a probabilistic contrastive loss which induces the latent space to capture information that is maximally useful to predict future samples
  - learn the representations that encode the underlying shared information between different parts of the (high-dimensional) signal. At the same time it discards low-level information and noise that is more local.

- How:

  - 最大化context和prediction 的互信息

  - When predicting future information we instead encode the target x (future) and context c (present) into a compact distributed vector representations (via non-linear learned mappings) in a way that maximally preserves the mutual information of the original signals x and c defined as

    <img src="/images/Visual-Self-supervised-Learning-assets/image-20220306172459042.png" alt="image-20220306172459042" style="width:50%;" />

  - we do not predict future observations <img src="http://latex.codecogs.com/svg.latex?x_{t&plus;k}" title="http://latex.codecogs.com/svg.latex?x_{t+k}" /> directly with a generative model <img src="http://latex.codecogs.com/svg.latex?p_k(x_{t&plus;k}|c_t)" title="http://latex.codecogs.com/svg.latex?p_k(x_{t+k}|c_t)" />. Instead we model a density ratio which preserves the mutual information between <img src="http://latex.codecogs.com/svg.latex?x_{t&plus;k}" title="http://latex.codecogs.com/svg.latex?x_{t+k}" /> and <img src="http://latex.codecogs.com/svg.latex?c_t" title="http://latex.codecogs.com/svg.latex?c_t" />

  - contrastive loss: q: predictions with context <img src="http://latex.codecogs.com/svg.latex?f(x_{t&plus;k}|c_t)" title="http://latex.codecogs.com/svg.latex?f(x_{t+k}|c_t)" />, k+: enbedding of ground truth, k-: sample out of context

  

### CMC:Contrastive Multiview Coding
<p align="center">
<img src="/images/Visual-Self-supervised-Learning-assets/image-20220306175348574.png" alt="image-20220306175348574" style="width:75%;" />
</p>
- Motivation

  - we learn a representation that aims to maximize mutual information between different views of the same scene but is otherwise compact

  - classic hypothesis that a powerful representation is one that models view-invariant

    factors.

- How: positive:same image of different view,e.g. depth image/segmentation image

- Disadvantage: encoder not shared

### MoCo
<p align="center">
<img src="/images/Visual-Self-supervised-Learning-assets/image-20220306181423380.png" alt="image-20220306181423380" style="width:100%;" />
</p>
- we hypothesize that it is desirable to build dictionaries that are: **(i) large and (ii) consistent** as they evolve during training. Intuitively, a larger dictionary may better sample the underlying continuous, highdimensional visual space, while the keys in the dictionary should be represented by the same or similar encoder so that their comparisons to the query are consistent.
- Shuffling BN：The model appears to “cheat” the pretext task and easily finds a low-loss solution. This is possibly because the intra-batch communication among samples (caused by BN) leaks information.
- 

### SimCLR
<p align="center">
<img src="/images/Visual-Self-supervised-Learning-assets/image-20220306184747940.png" alt="image-20220306184747940" style="width:75%;" />
</p>

- In-batch negative
- composition of data augmentations
  - spatial/geometric transformation of data: cropping and resizing (with horizontal flipping), rotation
  - appearance transformation, such as color distortion (including color dropping, brightness, contrast, saturation, hue)
  - it is critical to *compose cropping with color distortion** in order to learn generalizable features.
- global BN:
- Large batch size:
- nonlinear projection head:

​	We conjecture that the importance of **using the representation before the nonlinear projection is due to loss of information induced by the contrastive loss**. **In particular, <img src="http://latex.codecogs.com/svg.latex?z&space;=&space;g(h)" title="http://latex.codecogs.com/svg.latex?z = g(h)" /> is trained to be invariant to data transformation. Thus, <img src="http://latex.codecogs.com/svg.latex?g" title="http://latex.codecogs.com/svg.latex?g" /> can remove information that may be useful for the downstream task, such as the color or orientation of objects**. By leveraging the nonlinear transformation <img src="http://latex.codecogs.com/svg.latex?g(\cdot)" title="http://latex.codecogs.com/svg.latex?g(\cdot)" />, more information can be formed and maintained in <img src="http://latex.codecogs.com/svg.latex?h" title="http://latex.codecogs.com/svg.latex?h" />.

- **L2 normalization** (i.e. cosine similarity) along with temperature effectively weights different examples, and an **appropriate temperature** can help the model learn from **hard negatives**; and 2) unlike cross-entropy, other objective functions do not weigh the negatives by their relative hardness.
<p align="center">
  <img src="/images/Visual-Self-supervised-Learning-assets/image-20220306184454609.png" alt="image-20220306184454609" style="width:100%;" />
  </p>

### SwAV:Unsupervised Learning of Visual Features by Contrasting Cluster Assignments
<p align="center">
<img src="/images/Visual-Self-supervised-Learning-assets/image-20220306195148828.png" alt="image-20220306195148828" style="width:100%;" />
</p>
- Motivation:
  - SwAV, that takes advantage of contrastive methods without requiring to compute pairwise comparisons. Specifically, our method simultaneously clusters the data while **enforcing consistency between cluster assignments produced for different augmentations (or “views”) of the same image**, instead of comparing features directly as in contrastive learning.

- How

  <img src="/images/Visual-Self-supervised-Learning-assets/image-20220306201843121.png" alt="image-20220306201843121" style="width:75%;" />

  - Multi-Crop
  - 



## w/o negative examples

### BYOL
<p align="center">
<img src="/images/Visual-Self-supervised-Learning-assets/image-20220306220817154.png" alt="image-20220306220817154" style="width:100%;" />
</p>
<img src="/images/Visual-Self-supervised-Learning-assets/image-20220306223236207.png" alt="image-20220306223236207" style="width:75%;" />

<img src="/images/Visual-Self-supervised-Learning-assets/image-20220306223304632.png" alt="image-20220306223304632" style="width:50%;" />

### SimSiam:Exploring Simple Siamese Representation Learning
<p align="center">
<img src="/images/Visual-Self-supervised-Learning-assets/image-20220306225205266.png" alt="image-20220306225205266" style="width:75%;" />
</p>
- Motivation:that collapsing solutions do exist for the loss and structure, but a **stop-gradient** operation plays an essential role in preventing collapsing.

- SimSiam as EM algorithm

  考虑以下损失函数:
  <img src="http://latex.codecogs.com/svg.latex?\mathcal&space;L=\mathbb&space;E_{x,&space;\mathcal&space;T}\left[||\mathcal&space;F_\theta(\mathcal&space;T(x))-\eta_x||^2_2\right]" title="http://latex.codecogs.com/svg.latex?\mathcal L=\mathbb E_{x, \mathcal T}\left[||\mathcal F_\theta(\mathcal T(x))-\eta_x||^2_2\right]" />
  <img src="http://latex.codecogs.com/svg.latex?\eta_x" title="http://latex.codecogs.com/svg.latex?\eta_x" />是图片<img src="http://latex.codecogs.com/svg.latex?x" title="http://latex.codecogs.com/svg.latex?x" />的表示，是一组可学习的参数，<img src="http://latex.codecogs.com/svg.latex?\theta" title="http://latex.codecogs.com/svg.latex?\theta" />是网络参数。我们采用EM算法优化这两组参数。目标是:
  
  <img src="http://latex.codecogs.com/svg.latex?\theta^*,\eta^*=\min&space;\mathcal&space;L(\theta,&space;\eta)" title="http://latex.codecogs.com/svg.latex?\theta^*,\eta^*=\min \mathcal L(\theta, \eta)" />
  
  我们交替优化:
  
  <img src="http://latex.codecogs.com/svg.latex?\theta^t&space;\leftarrow&space;\arg&space;\min_\theta&space;\mathcal&space;L(\theta,&space;\eta^{t-1})" title="http://latex.codecogs.com/svg.latex?\theta^t \leftarrow \arg \min_\theta \mathcal L(\theta, \eta^{t-1})" />和<img src="http://latex.codecogs.com/svg.latex?\eta^t&space;\leftarrow&space;\arg&space;\min_\eta&space;\mathcal&space;L(\theta^t,&space;\eta)" title="http://latex.codecogs.com/svg.latex?\eta^t \leftarrow \arg \min_\eta \mathcal L(\theta^t, \eta)" />
  
  - E步：优化<img src="http://latex.codecogs.com/svg.latex?\theta" title="http://latex.codecogs.com/svg.latex?\theta" />：采用sgd优化<img src="http://latex.codecogs.com/svg.latex?\theta" title="http://latex.codecogs.com/svg.latex?\theta" /> 需要采用stop gradient防止梯度回传到<img src="http://latex.codecogs.com/svg.latex?\eta" title="http://latex.codecogs.com/svg.latex?\eta" />，因为<img src="http://latex.codecogs.com/svg.latex?\eta" title="http://latex.codecogs.com/svg.latex?\eta" />被视为常量
  - M步：优化<img src="http://latex.codecogs.com/svg.latex?\eta_x" title="http://latex.codecogs.com/svg.latex?\eta_x" /> :给定每个x并且固定<img src="http://latex.codecogs.com/svg.latex?\theta=\theta_t" title="http://latex.codecogs.com/svg.latex?\theta=\theta_t" />优化<img src="http://latex.codecogs.com/svg.latex?\mathcal&space;L" title="http://latex.codecogs.com/svg.latex?\mathcal L" />（求导）:
  
    <img src="http://latex.codecogs.com/svg.latex?\eta_x^t=\arg&space;\min_{\eta_x}\mathbb&space;E_{\mathcal&space;T}[||\mathcal&space;F_{\theta^t}(\mathcal&space;T(x))-\eta_x||^2_2]=\mathbb&space;E_{\mathcal&space;T}[\mathcal&space;F_{\theta^t}(\mathcal&space;T(x))]" title="http://latex.codecogs.com/svg.latex?\eta_x^t=\arg \min_{\eta_x}\mathbb E_{\mathcal T}[||\mathcal F_{\theta^t}(\mathcal T(x))-\eta_x||^2_2]=\mathbb E_{\mathcal T}[\mathcal F_{\theta^t}(\mathcal T(x))]" />

    说明<img src="http://latex.codecogs.com/svg.latex?\eta_x" title="http://latex.codecogs.com/svg.latex?\eta_x" />的最优解是<img src="http://latex.codecogs.com/svg.latex?x" title="http://latex.codecogs.com/svg.latex?x" />经过augmentation后表征的期望(关于augmentation)

- One-step alternation

  - 通过采样**一次**augmentation <img src="http://latex.codecogs.com/svg.latex?\mathcal&space;T'" title="http://latex.codecogs.com/svg.latex?\mathcal T'" />对<img src="http://latex.codecogs.com/svg.latex?\eta_x^t" title="http://latex.codecogs.com/svg.latex?\eta_x^t" />估计：
    <img src="http://latex.codecogs.com/svg.latex?\eta_x^t=\mathcal&space;F_{\theta^t}(\mathcal&space;T'(x))" title="http://latex.codecogs.com/svg.latex?\eta_x^t=\mathcal F_{\theta^t}(\mathcal T'(x))" />

  - 对<img src="http://latex.codecogs.com/svg.latex?\theta" title="http://latex.codecogs.com/svg.latex?\theta" />优化,带入上式,采用sgd优化
    <img src="http://latex.codecogs.com/svg.latex?\theta^{t&plus;1}=\arg&space;\min&space;\mathbb&space;E_{x,&space;\mathcal&space;T}[||\mathcal&space;F_{\theta}(\mathcal&space;T'(x))-\eta_x^t||]" title="http://latex.codecogs.com/svg.latex?\theta^{t+1}=\arg \min \mathbb E_{x, \mathcal T}[||\mathcal F_{\theta}(\mathcal T'(x))-\eta_x^t||]" />

- Predictor

  predictor h是对<img src="http://latex.codecogs.com/svg.latex?\mathbb&space;E_{\mathcal&space;T}[\mathcal&space;F_{\theta^t}(\mathcal&space;T(x))]" title="http://latex.codecogs.com/svg.latex?\mathbb E_{\mathcal T}[\mathcal F_{\theta^t}(\mathcal T(x))]" />的估计：

  <img src="http://latex.codecogs.com/svg.latex?h^*(z_1)&space;=&space;\arg&space;\min_{h(z_1)}&space;\mathbb&space;E_z[||h(z_1)-z_2||_2^2]=\mathbb&space;E_z[z_2]=\mathbb&space;E_{\mathcal&space;T}[f(\mathcal&space;T(x))]" title="http://latex.codecogs.com/svg.latex?h^*(z_1) = \arg \min_{h(z_1)} \mathbb E_z[||h(z_1)-z_2||_2^2]=\mathbb E_z[z_2]=\mathbb E_{\mathcal T}[f(\mathcal T(x))]" />

  在单步优化中,<img src="http://latex.codecogs.com/svg.latex?\mathbb&space;E_{\mathcal&space;T}&space;[\cdot]" title="http://latex.codecogs.com/svg.latex?\mathbb E_{\mathcal T} [\cdot]" />被忽略了，h视为对其的估计(Expectation over augmentations)，除了用h估计<img src="http://latex.codecogs.com/svg.latex?\mathbb&space;E_{\mathcal&space;T}&space;[\cdot]" title="http://latex.codecogs.com/svg.latex?\mathbb E_{\mathcal T} [\cdot]" />，也可以采用动量更新的方式

- Symmetrization

  Actually, the SGD optimizer computes the empirical expectation of <img src="http://latex.codecogs.com/svg.latex?\mathbb&space;E_{x,&space;\mathcal&space;T}[\cdot]" title="http://latex.codecogs.com/svg.latex?\mathbb E_{x, \mathcal T}[\cdot]" /> by sampling a batch of images and one pair of augmentations <img src="http://latex.codecogs.com/svg.latex?(\mathcal&space;T_1,&space;\mathcal&space;T_2)" title="http://latex.codecogs.com/svg.latex?(\mathcal T_1, \mathcal T_2)" />. In principle, the empirical expectation should be more precise with **denser sampling**,Symmetrization supplies an extra pair  <img src="http://latex.codecogs.com/svg.latex?(\mathcal&space;T_2,&space;\mathcal&space;T_1)" title="http://latex.codecogs.com/svg.latex?(\mathcal T_2, \mathcal T_1)" /> This explains that symmetrization is **not necessary** for our method to work, yet it is able to improve accuracy



### DINO
<p align="center">
<img src="/images/Visual-Self-supervised-Learning-assets/image-20220307210626613.png" alt="image-20220307210626613" style="width:75%;" />
</p>
