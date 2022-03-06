---
layout: "post"
title: "变分推断理解生成模型(VAE/GAN/AAE/ALI)"
date: "2022.02.04"
excerpt: "从变分推断角度理解EM算法、VAE、GAN、AAE、ALI(BiGAN)"
comments: true
---
# 变分推断理解生成模型(VAE/GAN/AAE/ALI)

从变分推断角度理解EM算法、VAE、GAN、AAE、ALI(BiGAN)



## 1.变分推断

假设<img src="http://latex.codecogs.com/svg.latex?x" title="http://latex.codecogs.com/svg.latex?x" />为显变量，<img src="http://latex.codecogs.com/svg.latex?x" title="http://latex.codecogs.com/svg.latex?z" />为隐变量，<img src="http://latex.codecogs.com/svg.latex?\widetilde&space;p(x)" title="http://latex.codecogs.com/svg.latex?\widetilde p(x)" />为<img src="http://latex.codecogs.com/svg.latex?x" title="http://latex.codecogs.com/svg.latex?x" />的真实分布,我们学习带参数的<img src="http://latex.codecogs.com/svg.latex?q_\theta(x)" title="http://latex.codecogs.com/svg.latex?q_\theta(x)" />估计<img src="http://latex.codecogs.com/svg.latex?\widetilde&space;p(x)" title="http://latex.codecogs.com/svg.latex?\widetilde p(x)" />：

<img src="http://latex.codecogs.com/svg.latex?q(x)=q_\theta(x)=\int_{z}q_\theta(x,z)dz" title="http://latex.codecogs.com/svg.latex?q(x)=q_\theta(x)=\int_{z}q_\theta(x,z)dz" />

我们希望<img src="http://latex.codecogs.com/svg.latex?q_\theta(x)" title="http://latex.codecogs.com/svg.latex?q_\theta(x)" />能逼近<img src="http://latex.codecogs.com/svg.latex?\widetilde&space;p(x)" title="http://latex.codecogs.com/svg.latex?\widetilde p(x)" />，所以一般情况下我们会去最大化对数似然函数:

<img src="http://latex.codecogs.com/svg.latex?\theta=\arg&space;\max_{\theta}\int_x\widetilde&space;p(x)\log&space;q(x)&space;dx" title="http://latex.codecogs.com/svg.latex?\theta=\arg \max_{\theta}\int_x\widetilde p(x)\log q(x) dx" />

等价于最小化KL散度<img src="http://latex.codecogs.com/svg.latex?KL(\widetilde&space;p(x)||q(x))" title="http://latex.codecogs.com/svg.latex?KL(\widetilde p(x)||q(x))" />：

<img src="http://latex.codecogs.com/svg.latex?KL(\widetilde&space;p(x)||q(x))=\int_x&space;\widetilde&space;p(x)&space;\log&space;\frac{\widetilde&space;p(x)}{q(x)}dx" title="http://latex.codecogs.com/svg.latex?KL(\widetilde p(x)||q(x))=\int_x \widetilde p(x) \log \frac{\widetilde p(x)}{q(x)}dx" />


该积分难以计算，变分推断中引入联合分布<img src="http://latex.codecogs.com/svg.latex?p(x,z)" title="http://latex.codecogs.com/svg.latex?p(x,z)" />使得<img src="http://latex.codecogs.com/svg.latex?\widetilde&space;p(x)=\int_z&space;p(x,z)dz" title="http://latex.codecogs.com/svg.latex?\widetilde p(x)=\int_z p(x,z)dz" />，变分推断的本质，将边缘分布的KL散度<img src="http://latex.codecogs.com/svg.latex?KL(\widetilde&space;p(x)||q(x))" title="http://latex.codecogs.com/svg.latex?KL(\widetilde p(x)||q(x))" />改为联合分布的KL散度<img src="http://latex.codecogs.com/svg.latex?KL(p(x,z)||q(x,z))" title="http://latex.codecogs.com/svg.latex?KL(p(x,z)||q(x,z))" />

<img src="http://latex.codecogs.com/svg.latex?KL(p(x,z)||q(x,z))=\int_x&space;\int_z&space;p(x,z)&space;\log&space;\frac{p(x,z)}{q(x,z)}&space;dz&space;dx&space;\\=&space;\int_x&space;\int_z&space;\widetilde&space;p(x)p(z|x)&space;\log&space;\frac{\widetilde&space;p(x)p(z|x)}{q(x)q(z|x)}dz&space;dx&space;\\&space;=&space;\int_x&space;\widetilde&space;p(x)\int_z&space;p(z|x)log&space;\left(\frac{\widetilde&space;p(x)p(z|x)}{q(x)q(z|x)}\right)dzdx&space;\\=&space;\int_x&space;\widetilde&space;p(x)\int_z&space;p(z|x)\left(\log&space;\frac{\widetilde&space;p(x)}{q(x)}&space;&plus;&space;\log&space;\frac{p(z|x)}{q(z|x)}\right)dzdx&space;\\=&space;\int_x&space;\widetilde&space;p(x)&space;\log&space;\frac{\widetilde&space;p(x)}{q(x)}&space;dx&space;&plus;&space;\int_x&space;\widetilde&space;p(x)&space;\int_z&space;p(z|x)&space;\log&space;\frac{p(z|x)}{q(z|x)}dzdx&space;\\=&space;KL(\widetilde&space;p(x)||q(x))&space;&plus;&space;\int_x&space;KL(p(z|x)||q(z|x))dx&space;\\=&space;KL(\widetilde&space;p(x)||q(x))&space;&plus;&space;\mathbb{E}_{x&space;\sim&space;p(x)}\left[&space;KL(p(z|x)||q(z|x))\right]&space;\\\geq&space;KL(\widetilde&space;p(x)||q(x))" title="http://latex.codecogs.com/svg.latex?KL(p(x,z)||q(x,z))=\int_x \int_z p(x,z) \log \frac{p(x,z)}{q(x,z)} dz dx \\= \int_x \int_z \widetilde p(x)p(z|x) \log \frac{\widetilde p(x)p(z|x)}{q(x)q(z|x)}dz dx \\ = \int_x \widetilde p(x)\int_z p(z|x)log \left(\frac{\widetilde p(x)p(z|x)}{q(x)q(z|x)}\right)dzdx \\= \int_x \widetilde p(x)\int_z p(z|x)\left(\log \frac{\widetilde p(x)}{q(x)} + \log \frac{p(z|x)}{q(z|x)}\right)dzdx \\= \int_x \widetilde p(x) \log \frac{\widetilde p(x)}{q(x)} dx + \int_x \widetilde p(x) \int_z p(z|x) \log \frac{p(z|x)}{q(z|x)}dzdx \\= KL(\widetilde p(x)||q(x)) + \int_x KL(p(z|x)||q(z|x))dx \\= KL(\widetilde p(x)||q(x)) + \mathbb{E}_{x \sim p(x)}\left[ KL(p(z|x)||q(z|x))\right] \\\geq KL(\widetilde p(x)||q(x))" />


<img src="http://latex.codecogs.com/svg.latex?\int_zp(z|x)dz&space;=&space;1&space;\\\int_z&space;p(x,z)&space;=&space;\int_z&space;p(z|x)p(x)dz&space;=&space;p(x)" title="http://latex.codecogs.com/svg.latex?\int_zp(z|x)dz = 1 \\\int_z p(x,z) = \int_z p(z|x)p(x)dz = p(x)" />

联合分布的KL散度是一个更强的条件（上界）。所以一旦优化成功，那么我们就得到<img src="http://latex.codecogs.com/svg.latex?q(x,z)\rightarrow&space;p(x,z)" title="http://latex.codecogs.com/svg.latex?q(x,z)\rightarrow p(x,z)" />，从而<img src="http://latex.codecogs.com/svg.latex?\int_z&space;q(x,z)dz&space;\rightarrow&space;\int_z&space;p(x,z)&space;dz&space;=&space;\widetilde&space;p(x)" title="http://latex.codecogs.com/svg.latex?\int_z q(x,z)dz \rightarrow \int_z p(x,z) dz = \widetilde p(x)" />进而得到<img src="http://latex.codecogs.com/svg.latex?\widetilde&space;p(x)" title="http://latex.codecogs.com/svg.latex?\widetilde p(x)" />的近似。



## 2.VAE

​	在VAE中，我们设<img src="http://latex.codecogs.com/svg.latex?q(x,z)=q(x|z)q(z),p(x,z)=\widetilde&space;p(x)p(z|x)" title="http://latex.codecogs.com/svg.latex?q(x,z)=q(x|z)q(z),p(x,z)=\widetilde p(x)p(z|x)" />，其中<img src="http://latex.codecogs.com/svg.latex?q(x|z),p(z|x)" title="http://latex.codecogs.com/svg.latex?q(x|z),p(z|x)" />分别是decoder和encoder,假设为带有未知参数的高斯分布而<img src="http://latex.codecogs.com/svg.latex?q(x|z),p(z|x)" title="http://latex.codecogs.com/svg.latex?q(z)" />是标准高斯分布。最小化的目标是:

<img src="http://latex.codecogs.com/svg.latex?KL&space;\left(&space;p(x,z)||q(x,z)&space;\right)=&space;\int_x\int_z&space;\widetilde&space;p(x)p(z|x)&space;\log&space;\frac{\widetilde&space;p(x)p(z|x)}{q(x|z)q(z)}dzdx&space;\\=&space;\int_x&space;\widetilde&space;p(x)&space;\int_z&space;p(z|x)&space;\left(\log&space;\widetilde&space;p(x)&space;&plus;&space;\log\frac{p(z|x)}{q(z)}&space;-&space;\log&space;q(x|z)&space;\right)dzdx&space;\\=&space;\int_x&space;\widetilde&space;p(x)&space;\log&space;\widetilde&space;p(x)&space;&plus;\int_x&space;\left(KL(p(z|x)||q(z))&space;-&space;\int_z&space;p(z|x)&space;\log&space;q(x|z)dz&space;\right)dx&space;\\=&space;C&space;&plus;&space;\mathbb&space;E_{x&space;\sim&space;\widetilde&space;p(x)}[KL(p(z|x)||q(z))-\int_z&space;p(z|x)&space;\log&space;q(x|z)dz]" title="http://latex.codecogs.com/svg.latex?KL \left( p(x,z)||q(x,z) \right)= \int_x\int_z \widetilde p(x)p(z|x) \log \frac{\widetilde p(x)p(z|x)}{q(x|z)q(z)}dzdx \\= \int_x \widetilde p(x) \int_z p(z|x) \left(\log \widetilde p(x) + \log\frac{p(z|x)}{q(z)} - \log q(x|z) \right)dzdx \\= \int_x \widetilde p(x) \log \widetilde p(x) +\int_x \left(KL(p(z|x)||q(z)) - \int_z p(z|x) \log q(x|z)dz \right)dx \\= C + \mathbb E_{x \sim \widetilde p(x)}[KL(p(z|x)||q(z))-\int_z p(z|x) \log q(x|z)dz]" />

其中<img src="http://latex.codecogs.com/svg.latex?log&space;\widetilde&space;p(x)" title="http://latex.codecogs.com/svg.latex?log \widetilde p(x)" />没有包含优化目标，可以视为常数，而对<img src="http://latex.codecogs.com/svg.latex?log&space;\widetilde&space;p(x)" title="http://latex.codecogs.com/svg.latex?\widetilde p(x)" />的积分则转化为对样本的采样.

因为<img src="http://latex.codecogs.com/svg.latex?q(x|z),p(z|x)" title="http://latex.codecogs.com/svg.latex?q(x|z),p(z|x)" />为带有神经网络的高斯分布，这时候<img src="http://latex.codecogs.com/svg.latex?KL(p(z|x||q(z))" title="http://latex.codecogs.com/svg.latex?KL(p(z|x||q(z))" />可以显式地算出，而通过重参数技巧来**采样一个点**完成积分<img src="http://latex.codecogs.com/svg.latex?\int_z&space;p(z|x)\log&space;q(x|z)dz" title="http://latex.codecogs.com/svg.latex?\int_z p(z|x)\log q(x|z)dz" />的估算，可以得到VAE最终要最小化的loss：

<img src="http://latex.codecogs.com/svg.latex?\mathbb&space;E_{x&space;\sim&space;\widetilde&space;p(x)}&space;\left[&space;-&space;\log&space;q(z|x)&space;&plus;&space;KL(p(z|x)||q(z))&space;\right]" title="http://latex.codecogs.com/svg.latex?\mathbb E_{x \sim \widetilde p(x)} \left[ - \log q(z|x) + KL(p(z|x)||q(z)) \right]" />


式中<img src="http://latex.codecogs.com/svg.latex?KL(p(z|x)||q(z))" title="http://latex.codecogs.com/svg.latex?KL(p(z|x)||q(z))" />反应<img src="http://latex.codecogs.com/svg.latex?KL(p(z|x)||q(z))" title="http://latex.codecogs.com/svg.latex?z" />的辨识度，过小则<img src="http://latex.codecogs.com/svg.latex?-\log&space;q(z|x)" title="http://latex.codecogs.com/svg.latex?-\log q(z|x)" />不会小，而如果<img src="http://latex.codecogs.com/svg.latex?-\log&space;q(z|x)" title="http://latex.codecogs.com/svg.latex?-\log q(z|x)" />小则<img src="http://latex.codecogs.com/svg.latex?q(x|z)" title="http://latex.codecogs.com/svg.latex?q(x|z)" />大，预测准确，这时候<img src="http://latex.codecogs.com/svg.latex?p(z|x)" title="http://latex.codecogs.com/svg.latex?p(z|x)" />不会太随机，即<img src="http://latex.codecogs.com/svg.latex?KL(p(z|x||q(z))" title="http://latex.codecogs.com/svg.latex?KL(p(z|x||q(z))" />不会小，所以这两部分的loss其实是相互拮抗的。

​	重参数技巧：从<img src="http://latex.codecogs.com/svg.latex?\mathbf&space;N(\mu,\delta^2)" title="http://latex.codecogs.com/svg.latex?\mathbf N(\mu,\delta^2)" />中采样一个<img src="http://latex.codecogs.com/svg.latex?z" title="http://latex.codecogs.com/svg.latex?z" />，相当于从<img src="http://latex.codecogs.com/svg.latex?\mathbf&space;N(0,I)" title="http://latex.codecogs.com/svg.latex?\mathbf N(0,I)" />中采样一个<img src="http://latex.codecogs.com/svg.latex?\epsilon" title="http://latex.codecogs.com/svg.latex?\epsilon" />，然后让<img src="http://latex.codecogs.com/svg.latex?z=\mu&space;&plus;&space;\epsilon*\delta" title="http://latex.codecogs.com/svg.latex?z=\mu + \epsilon*\delta" />。

## 3.EM算法

在VAE中我们对后验分布做了约束，仅假设它是高斯分布，所以我们优化的是高斯分布的参数。如果不作此假设，那么直接优化原始目标<img src="http://latex.codecogs.com/svg.latex?KL(p(x,z)||q(x,z))" title="http://latex.codecogs.com/svg.latex?KL(p(x,z)||q(x,z))" />，在某些情况下也是可操作的,EM 采用交替优化的方式.

<img src="http://latex.codecogs.com/svg.latex?KL&space;\left(&space;p(x,z)||q(x,z)&space;\right)=&space;\int_x\int_z&space;\widetilde&space;p(x)p(z|x)&space;\log&space;\frac{\widetilde&space;p(x)p(z|x)}{q(x|z)q(z)}dzdx&space;\\=&space;\int_x\int_z&space;\widetilde&space;p(x)p(z|x)&space;\log&space;\frac{\widetilde&space;p(x)p(z|x)}{q(x,z)}dzdx&space;\\=&space;\int_x&space;\widetilde&space;p(x)&space;\left(\int_z&space;p(z|x)&space;\log&space;\frac{\widetilde&space;p(x)p(z|x)}{q(x,z)}&space;dz\right)dx&space;\\=&space;\int_x&space;\widetilde&space;p(x)&space;\left(\int_z&space;p(z|x)&space;\left(&space;\log&space;p(z|x)&space;-&space;\log&space;q(x,z)&space;&plus;&space;\log&space;\widetilde&space;p(x)&space;\right)&space;dz\right)dx&space;\\" title="http://latex.codecogs.com/svg.latex?KL \left( p(x,z)||q(x,z) \right)= \int_x\int_z \widetilde p(x)p(z|x) \log \frac{\widetilde p(x)p(z|x)}{q(x|z)q(z)}dzdx \\= \int_x\int_z \widetilde p(x)p(z|x) \log \frac{\widetilde p(x)p(z|x)}{q(x,z)}dzdx \\= \int_x \widetilde p(x) \left(\int_z p(z|x) \log \frac{\widetilde p(x)p(z|x)}{q(x,z)} dz\right)dx \\= \int_x \widetilde p(x) \left(\int_z p(z|x) \left( \log p(z|x) - \log q(x,z) + \log \widetilde p(x) \right) dz\right)dx \\" />

**E步**：先固定<img src="http://latex.codecogs.com/svg.latex?p(z|x)" title="http://latex.codecogs.com/svg.latex?p(z|x)" />，优化<img src="http://latex.codecogs.com/svg.latex?q(x|z)" title="http://latex.codecogs.com/svg.latex?q(x|z)" />，那么就有

<img src="http://latex.codecogs.com/svg.latex?q(x|z)=\arg&space;\min_{q(x|z)}&space;\mathbb&space;E_{x&space;\sim&space;\widetilde&space;p(x)}[\int_z&space;p(z|x)&space;\log&space;q(x,z)dz]" title="http://latex.codecogs.com/svg.latex?q(x|z)=\arg \min_{q(x|z)} \mathbb E_{x \sim \widetilde p(x)}[\int_z p(z|x) \log q(x,z)dz]" />

**M步**：完成这一步后，我们固定<img src="http://latex.codecogs.com/svg.latex?q(x,z)" title="http://latex.codecogs.com/svg.latex?q(x,z)" />，优化<img src="http://latex.codecogs.com/svg.latex?q(x,z)" title="http://latex.codecogs.com/svg.latex?p(z|x)" />，先将<img src="http://latex.codecogs.com/svg.latex?q(x|z)q(z)" title="http://latex.codecogs.com/svg.latex?q(x|z)q(z)" />写成<img src="http://latex.codecogs.com/svg.latex?q(z|x)q(x)" title="http://latex.codecogs.com/svg.latex?q(z|x)q(x)" />的形式：

<img src="http://latex.codecogs.com/svg.latex?KL&space;\left(&space;p(x,z)||q(x,z)&space;\right)=&space;\int_x\int_z&space;\widetilde&space;p(x)p(z|x)&space;\log&space;\frac{\widetilde&space;p(x)p(z|x)}{q(x,z)}&space;\\=&space;\int_x&space;\widetilde&space;p(x)&space;\left(\int_z&space;p(z|x)&space;\left(&space;\log&space;p(z|x)&space;-&space;\log&space;q(x,z)&space;&plus;&space;\log&space;\widetilde&space;p(x)&space;\right)&space;dz\right)dx&space;\\=&space;\int_x&space;\widetilde&space;p(x)&space;\left(\int_z&space;p(z|x)&space;\left(&space;\log&space;p(z|x)&space;-&space;\log&space;q(z|x)&space;q(x)&space;&plus;&space;\log&space;\widetilde&space;p(x)&space;\right)&space;dz\right)dx&space;\\=&space;\int_x&space;\widetilde&space;p(x)&space;\left(\int_z&space;p(z|x)&space;\left(&space;\log&space;p(z|x)&space;-&space;\log&space;q(z|x)&space;-&space;\log&space;q(x)&space;&plus;&space;\log&space;\widetilde&space;p(x)&space;\right)&space;dz\right)dx&space;\\=&space;\int_x&space;\widetilde&space;p(x)&space;\left(\int_z&space;p(z|x)&space;\left(&space;\log&space;p(z|x)&space;-&space;\log&space;q(z|x)&space;&space;\right)&space;dz&space;-&space;\log&space;q(x)&space;&plus;&space;\log&space;\widetilde&space;p(x)&space;\right)dx&space;\\&space;=&space;\int_x&space;\widetilde&space;p(x)&space;\left(\int_z&space;p(z|x)&space;\log&space;\frac{p(z|x)}{q(z|x)}&space;dz&space;-&space;\log&space;q(x)&space;&plus;&space;\log&space;\widetilde&space;p(x)&space;\right)dx&space;\\=&space;\mathbb&space;E_{x&space;\sim&space;\widetilde&space;p(x)}[KL(p(z|x)||q(z|x))]&space;&plus;&space;C" title="http://latex.codecogs.com/svg.latex?KL \left( p(x,z)||q(x,z) \right)= \int_x\int_z \widetilde p(x)p(z|x) \log \frac{\widetilde p(x)p(z|x)}{q(x,z)} \\= \int_x \widetilde p(x) \left(\int_z p(z|x) \left( \log p(z|x) - \log q(x,z) + \log \widetilde p(x) \right) dz\right)dx \\= \int_x \widetilde p(x) \left(\int_z p(z|x) \left( \log p(z|x) - \log q(z|x) q(x) + \log \widetilde p(x) \right) dz\right)dx \\= \int_x \widetilde p(x) \left(\int_z p(z|x) \left( \log p(z|x) - \log q(z|x) - \log q(x) + \log \widetilde p(x) \right) dz\right)dx \\= \int_x \widetilde p(x) \left(\int_z p(z|x) \left( \log p(z|x) - \log q(z|x) \right) dz - \log q(x) + \log \widetilde p(x) \right)dx \\ = \int_x \widetilde p(x) \left(\int_z p(z|x) \log \frac{p(z|x)}{q(z|x)} dz - \log q(x) + \log \widetilde p(x) \right)dx \\= \mathbb E_{x \sim \widetilde p(x)}[KL(p(z|x)||q(z|x))] + C" />



<img src="http://latex.codecogs.com/svg.latex?p(z|x)=\arg&space;\min_{p(z|x)}&space;\mathbb&space;E_{x&space;\sim&space;\widetilde&space;p(x)}[KL(p(z|x)||q(z|x))]=q(z|x)" title="http://latex.codecogs.com/svg.latex?p(z|x)=\arg \min_{p(z|x)} \mathbb E_{x \sim \widetilde p(x)}[KL(p(z|x)||q(z|x))]=q(z|x)" />


由于现在对<img src="http://latex.codecogs.com/svg.latex?p(z|x)" title="http://latex.codecogs.com/svg.latex?p(z|x)" />没有约束，因此可以直接让<img src="http://latex.codecogs.com/svg.latex?p(z|x)=q(z|x)" title="http://latex.codecogs.com/svg.latex?p(z|x)=q(z|x)" />使得loss等于0.

<img src="http://latex.codecogs.com/svg.latex?p(z|x)" title="http://latex.codecogs.com/svg.latex?p(z|x)" />有理论最优解：

<img src="http://latex.codecogs.com/svg.latex?p(z|x)=&space;q(z|x)&space;=&space;\frac{q(x|z)q(z)}{q(x)}=\frac{q(x|z)q(z)}{\int_z&space;q(x|z)q(z)dz}" title="http://latex.codecogs.com/svg.latex?p(z|x)= q(z|x) = \frac{q(x|z)q(z)}{q(x)}=\frac{q(x|z)q(z)}{\int_z q(x|z)q(z)dz}" />


## 4.GAN

## Reference:
苏剑林. (Jul. 18, 2018). 《用变分推断统一理解生成模型（VAE、GAN、AAE、ALI） 》[Blog post]. Retrieved from https://kexue.fm/archives/5716

