# Convolution & Filter 🔧

Convolution 연산의 수학적 기초부터 CNN 설계, 효율화까지 상세하게 다룹니다.

## 📚 목차

- [Convolution 기본 개념](#1-convolution-기본-개념)
- [이미지 필터 종류](#2-이미지-필터-종류)
- [CNN Convolution 연산](#3-cnn-convolution-연산)
- [실전 코드 예제](#5-실전-코드-예제)

## 1. Convolution 기본 개념

### 1.1 수학적 정의

**연속 시간 컨벌루션**:
```
(y * x)(t) = ∫₋∞⁺∞ x(τ) · h(t-τ) dτ
```

**이산 시간 컨벌루션**:
```
y[n] = (x * h)[n] = Σₖ₌₋∞⁺∞ x[k] · h[n-k]
```

**2D 이미지 컨벌루션**:
```
(I * K)(i, j) = Σₘ₌₀ᴹ⁻¹ Σₙ₌₀ᴺ⁻¹ I(i-m, j-n) · K(m, n)
```

### 1.2 핵심 파라미터

| 파라미터 | 설명 |
|------|--|
| **Kernel Size** | 커널 크기 (3×3, 5×5, 7×7 등) |
| **Stride** | 슬라이딩 간격 |
| **Padding** | 주변 패딩 (same/valid) |
| **Dilation** | dilation rate |

**출력 크기 공식**:
```
Output = (Input - Kernel + 2×Padding - 1) / Stride + 1
```

## 2. 이미지 필터 종류

### 2.1 저역통과 필터 (Low-pass)

#### Gaussian Blur

```python
import torch
import torch.nn.functional as F

def gaussian_kernel(size=5, sigma=1.0):
    coords = torch.arange(size, dtype=torch.float)
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-((x - size//2)**2 + **(y - size//2)2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)
```

### 2.2 고역통과 필터 (High-pass)

#### Sobel Operator

```python
class SobelFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float()
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float()
    
    def forward(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(gx**2 + gy**2)
```

## 3. CNN Convolution 연산

```python
import torch.nn as nn

# Standard Conv2D
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, 
                 stride=1, padding=1, bias=True)

# 1×1 Convolution (Channel mixing)
conv_1x1 = nn.Conv2d(128, 64, kernel_size=1)

# Depthwise Convolution (Efficient)
depthwise = nn.Conv2d(256, 256, 3, padding=1, groups=256)

# Dilated Convolution
dilated = nn.Conv2d(256, 256, 3, dilation=2)
```

## 4. 효율적 Convolution 기법

### 4.1 Depthwise Separable Convolution

```python
class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.pw(x)
        x = self.bn2(x)
        return F.relu(x)
```

### 4.2 FFT-based Convolution

```python
def fft_convolution(x, kernel):
    x_fft = torch.fft.fft2(x)
    k_fft = torch.fft.fft2(kernel)
    result_fft = x_fft * k_fft
    return torch.fft.ifft2(result_fft).real
```

## 📊 성능 비교

| Model | Type | Params (M) | FLOPs (G) |
|-------|-----|--|---|
| **ResNet-50** | Standard | 25.6 | 4.1 |
| **MobileNet-V2** | Depthwise | 3.5 | 0.35 |
| **EfficientNet-B0** | Depthwise | 5.3 | 0.4 |

---

*마지막 업데이트: 2026-03-30*
