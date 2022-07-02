# MobileOne PyTorch

Unofficial PyTorch implementation of
[**An Improved One millisecond Mobile Backbone**](https://arxiv.org/pdf/2206.04040.pdf) paper.

## Quickstart

Install with `pip install mobileone_pytorch` and create a MobileOne with:

```python
from mobileone_pytorch import mobileone_s1
model = mobileone_s1()
```

## Overview

This repository contains an implementation of MobileOne.

Features:

- Implementation of all MobileOne versions
- Reparametrization for model deployment

*Upcomining features*:

- Squeeze-and-Excitation block for MobileOne S4

**Help wanted**:

- Training models on ImageNet

## Table of contents

1. [About MobileOne](#about-mobileone)
2. [Installation](#installation)
3. [Usage](#usage)
   - [Create models](#create-models)
   - [Deployment via reparametrization](#deployment)

### About MobileOne

MobileOne is a novel architecture that with variants achieves an inference time
under 1 ms on an iPhone12 with 75.9% top-1 accuracy on ImageNet.

- MobileOne achieves state-of-the-art performance
within the efficient architectures while being many times faster
on mobile.

- The best model (S4) obtains similar performance on ImageNet
as [Mobile-Former](https://arxiv.org/abs/2108.05895) while being 38× faster.
Moreover it obtains 2.3% better top-1 accuracy on ImageNet
than [EfficientNet](https://arxiv.org/abs/1905.11946) at similar latency.

### Installation

Install via pip:

```bash
pip install mobileone_pytorch
```

Or install from source:

```bash
git clone https://github.com/federicopozzi33/MobileOne-PyTorch.git
cd mobileone_pytorch
pip install -e .
```

### Usage

#### Create models

Create MobileOne models:

```python
from mobileone_pytorch import (
   mobileone_s0, 
   mobileone_s1, 
   mobileone_s2, 
   mobileone_s3, 
   mobileone_s4
)

model_s0 = mobileone_s0()
model_s1 = mobileone_s1()
model_s2 = mobileone_s2()
model_s3 = mobileone_s3()
model_s4 = mobileone_s4()
```

#### Deployment

Deploy a MobileOne through reparametrization:

```python
import torch
from mobileone_pytorch import mobileone_s1

x = torch.rand(1, 3, 224, 224)

model = mobileone_s1()
deployed = model.reparametrize()

model.eval()
deployed.eval()

out1 = model(x)
out2 = deployed(x)

torch.testing.assert_close(out1, out2)
```

### Contributing

If you find a bug, create a GitHub issue.
Similarly, if you have questions, simply post them as GitHub issues.
