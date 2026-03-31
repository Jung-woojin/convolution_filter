# Convolution & Filter 🔧

Convolution 연산의 수학적 기초부터 CNN 설계, 효율화까지 상세하게 다룹니다.

---

## 📚 목차

- [Convolution 기본 개념](#1-convolution-기본-개념)
- [이미지 필터 종류](#2-이미지-필터-종류)
- [CNN Convolution 연산](#3-cnn-convolution-연산)
  - [Standard & 1×1 Convolution](#31-cnn-의-convolution-구조)
  - [Group & Depthwise Convolution](#33-group-convolution)
  - [Dilated Convolution](#35-dilated-convolution)
  - **🆕 Deformable Convolution (DCN)**
  - **🆕 Dynamic Convolution**
- [효율적 Convolution 기법](#4-효율적-convolution-기법)
- [실전 코드 예제](#5-실전-코드-예제)
- [성능 최적화 가이드](#6-성능-최적화-가이드)

---

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

여기서:
- `I`: 입력 이미지
- `K`: 커널 (필터)
- `(m, n)`: 커널 좌표
- `(i, j)`: 출력 좌표

### 1.2 기하학적 직관

**Convolution = 슬라이딩 윈도우 연산**

```
입력 이미지:          커널:            출력:
┌──────┐           ┌──┐              ┌──┐
│1 2 3 │   ●       │3 1 │     =       │●│
│4 5 6 │   +       │1 -2│     →       │●│
│7 8 9 │           │-1 3│              │●│
└──────┘           └──┘              └──┘

각 위치에서:
- 커널을 이미지 위에 슬라이딩
- 해당 영역과 점곱 (element-wise multiply)
- 합산하여 출력 픽셀 계산
```

### 1.3 핵심 파라미터

| 파라미터 | 설명 | 영향 |
|------|--|----|
| **Kernel Size** | 커널 크기 (3×3, 5×5, 7×7 등) | receptive field, 계산량 |
| **Stride** | 슬라이딩 간격 | 출력 크기, downsampling |
| **Padding** | 주변 패딩 (same/valid) | 출력 크기 보존 여부 |
| **Dilation** | dilation rate | receptive field 확장 |

**출력 크기 공식**:
```
Output = (Input - Kernel + 2×Padding - 1) / Stride + 1
```

**예시**:
```python
# Input: 224×224, Kernel: 3×3, Stride: 1, Padding: 1
Output = (224 - 3 + 2×1 - 1) / 1 + 1 = 224×224 (크기 보존)

# Input: 224×224, Kernel: 3×3, Stride: 2, Padding: 1
Output = (224 - 3 + 2×1 - 1) / 2 + 1 = 112×112 (downsampling)
```

### 1.4 Convolution vs Cross-Correlation

**수학적으로 다름**:
- **Convolution**: 커널을 반전 후 연산
- **Cross-correlation**: 커널 반전 없이 연산

**CNN 에서는 Cross-correlation**:
```python
# 실제 CNN 은 convolution 이 아니라 cross-correlation 사용
# 하지만 관례상 "convolution" 이라고 부름
```

---

## 2. 이미지 필터 종류

### 2.1 저역통과 필터 (Low-pass Filters)

#### Gaussian Blur

**수식**:
```
G(x, y) = (1/(2πσ²)) · exp(-(x²+y²)/(2σ²))
```

**PyTorch 구현**:
```python
import torch
import torch.nn.functional as F

def gaussian_kernel(size=5, sigma=1.0):
    """Generate Gaussian kernel"""
    coords = torch.arange(size, dtype=torch.float)
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    
    # Gaussian function
    kernel = torch.exp(-((x - size//2)**2 + **(y - size//2)2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    return kernel.unsqueeze(0).unsqueeze(0)

def apply_gaussian_blur(image, sigma=1.0):
    """Apply Gaussian blur"""
    kernel = gaussian_kernel(size=5, sigma=sigma)
    kernel = kernel.repeat(image.shape[1], 1, 1)  # Channels first
    
    # Padding
    padding = 2
    padded = F.pad(image, (padding, padding, padding, padding), mode='replicate')
    
    # Convolution
    output = F.conv2d(padded, kernel, padding=0, groups=image.shape[1])
    
    return output
```

**용도**:
- 노이즈 제거
- 이미지 축소 전 anti-aliasing
- Smoothness regularization

#### Box Filter (Moving Average)

```python
def box_kernel(size=3):
    """Generate uniform box filter"""
    return torch.ones(size, size) / (size * size)
```

**용도**:
- Simple blurring
- Pre-processing for edge detection

---

### 2.2 고역통과 필터 (High-pass Filters)

#### **📊 X 축/ Y 축 관점: 필터를 축별로 분리해서 보는 이유**

**핵심 개념**:
```
이미지 = 수평 방향 변화 (Y 축 변화) + 수직 방향 변화 (X 축 변화)

X 축 필터 → Y 방향 gradient (수직 에지 검출)
Y 축 필터 → X 방향 gradient (수평 에지 검출)
```

**왜 축별로 분리할까?**

1. **방향성 특징 추출**
   - 수평선/수직선 구분
   - 텍스처 방향 분석
   - 객체 윤곽 방향 이해

2. **문제별 최적화**
   - OCR: 수평선 강조 (문자 baseline)
   - 건축: 수직선 강조 (빌딩 라인)
   - 지질: 방향성 패턴 분석

3. **컴퓨팅 효율**
   - 필요 방향만 처리
   - 분리 필터링 후 fusion
   - 병렬 처리 최적화

---

#### Sobel Operator - 축별 분리

**X 방향 (수직 에지 검출)**:
```
Gx = [ -1  0  1 ]
     [ -2  0  2 ]
     [ -1  0  1 ]

이 필터는 Y 축을 따라 변화하는 에지 (수직 에지) 검출
예시: |   |   |   |   |   (문자, 기둥, 문)
```

**Y 방향 (수평 에지 검출)**:
```
Gy = [ -1 -2 -1 ]
     [  0  0  0 ]
     [  1  2  1 ]

이 필터는 X 축을 따라 변화하는 에지 (수평 에지) 검출
예시: ━━━━
      ━━━━  (바닥, 하늘 경계, 선)
```

**PyTorch 구현**:
```python
class SobelFilter(nn.Module):
    """Sobel edge detection - X/Y 축 분리"""
    def __init__(self):
        super().__init__()
        
        # X 축 필터: 수직 에지 검출 (Y 방향 변화)
        self.sobel_x = torch.tensor([[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
        
        # Y 축 필터: 수평 에지 검출 (X 방향 변화)
        self.sobel_y = torch.tensor([[-1, -2, -1],
                                      [ 0,  0,  0],
                                      [ 1,  2, 1]]).float().unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        # 각 축별 gradient 계산
        gx = F.conv2d(x, self.sobel_x.to(x.device), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(x.device), padding=1)
        
        # 축별 통계
        x_magnitude = torch.abs(gx)  # 수직 에지 강도
        y_magnitude = torch.abs(gy)  # 수평 에지 강도
        
        # 전체 magnitude
        magnitude = torch.sqrt(gx**2 + gy**2)
        
        # Gradient direction
        direction = torch.atan2(gy, gx)
        
        return magnitude, direction, x_magnitude, y_magnitude
```

**실전 예시: 축별 특징 분석**

```python
def analyze_edge_directions(image):
    """이미지 내 수평/수직 에지 비율 분석"""
    
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]]).float()
    
    sobel_y = torch.tensor([[-1, -2, -1],
                            [ 0,  0,  0],
                            [ 1,  2, 1]]).float()
    
    # 각 축별 gradient
    gx = F.conv2d(image, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    gy = F.conv2d(image, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    
    x_strength = torch.mean(torch.abs(gx))  # 수직 에지 강도
    y_strength = torch.mean(torch.abs(gy))  # 수평 에지 강도
    
    # 방향성 비율
    if y_strength > 0:
        vertical_ratio = x_strength / y_strength
    else:
        vertical_ratio = 0
    
    print(f"수직 에지 강도: {x_strength:.4f}")
    print(f"수평 에지 강도: {y_strength:.4f}")
    print(f"방향 비율 (수직/수평): {vertical_ratio:.2f}")
    
    # 이미지 타입 분류
    if vertical_ratio > 1.5:
        print("→ 수직 구조물 강조 (빌딩, 문자, 기둥)")
    elif vertical_ratio < 0.7:
        print("→ 수평 구조물 강조 (지평선, 바닥)")
    else:
        print("→ 균형 잡힌 구조")
```

**적용 사례**:

| Application | 주요 축 | 필터 활용 |
|------|--|----|
| **OCR** | 수평 | 문자 baseline 감지, 줄 간격 분석 |
| **건축** | 수직 | 건물 라인, 창문 프레임 |
| **도로** | 수평 | 차선, 도로 경계 |
| **지질** | 방향성 | 층리 방향 분석 |
| **의료** | 방향성 | 혈관, 신경 방향 |

---

#### **Advanced: Direction-Specific Filters**

#### Orientation-Specific Filters

**45° 필터** (주성분 분석):
```
D1 = [ -1 -1  0 ]
     [ -1  0  1 ]
     [  0  1  1 ]

D2 = [  0 -1 -1 ]
     [  1  0 -1 ]
     [  1  1  0 ]
```

**Orthogonal 2 차 필터**:
```python
def create_directional_filters():
    """8 방향 에지 검출 필터"""
    
    directions = [
        # 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
        torch.tensor([[-1, 0, 1],
                      [-2, 0, 2],
                      [-1, 0, 1]]),  # 90° (수직)
        
        torch.tensor([[-1, -1, 0],
                      [-1, 0, 1],
                      [0, 1, 1]]),  # 45°
        
        torch.tensor([[-1, -2, -1],
                      [ 0,  0,  0],
                      [ 1,  2, 1]]),  # 0° (수평)
        
        # ... 나머지 5 방향
    ]
    
    return directions
```

**Use Case: 텍스처 분석**

```python
def texture_analysis(image, kernel_size=3):
    """텍스처 방향성 분석"""
    
    filters = create_directional_filters()
    
    # 각 방향별 response
    responses = []
    for i, f in enumerate(filters):
        response = F.conv2d(image, f.unsqueeze(0).unsqueeze(0), padding=1)
        responses.append(torch.mean(torch.abs(response)))
    
    # dominant direction 찾기
    dominant_axis = torch.argmax(torch.tensor(responses))
    
    print(f"주 텍스처 방향: {dominant_axis}°")
    
    return responses
```

---

#### Sobel Operator

**PyTorch 구현**:
```python
class SobelFilter(nn.Module):
    """Sobel edge detection"""
    def __init__(self):
        super().__init__()
        
        self.sobel_x = torch.tensor([[-1, 0, 1],
                                      [-2, 0, 2],
                                      [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0)
        
        self.sobel_y = torch.tensor([[-1, -2, -1],
                                      [ 0,  0,  0],
                                      [ 1,  2, 1]]).float().unsqueeze(0).unsqueeze(0)
    
    def forward(self, x):
        # Convolve
        gx = F.conv2d(x, self.sobel_x.to(x.device), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(x.device), padding=1)
        
        # Magnitude
        magnitude = torch.sqrt(gx**2 + gy**2)
        
        # Gradient direction
        direction = torch.atan2(gy, gx)
        
        return magnitude, direction
```

**용도**:
- Edge detection
- Feature extraction
- 이미지 pre-processing

#### Laplacian Operator - 2 차 미분

**수식**:
```
L = [ 0  -1   0 ]
    [ -1   4  -1 ]
    [ 0  -1   0 ]

4 방향 모두 균등 (방향성 없음)
X 축: -1, 4, -1 → 2 차 미분
Y 축: -1, 4, -1 → 2 차 미분
```

**축별 분리 Laplacian**:

```python
def separable_laplacian_kernel():
    """X 축과 Y 축을 분리한 Laplacian"""
    
    # X 축 2 차 미분
    kernel_x = torch.tensor([[-1],
                             [4],
                             [-1]]).float().unsqueeze(0).unsqueeze(0)
    
    # Y 축 2 차 미분
    kernel_y = torch.tensor([[-1, 4, -1]]).float().unsqueeze(0).unsqueeze(0)
    
    return kernel_x, kernel_y

def separable_laplacian(image):
    """2 차 미분을 축별로 계산"""
    
    kx, ky = separable_laplacian_kernel()
    
    # X 축 2 차 미분
    lx = F.conv2d(image, kx.to(image.device), padding=1)
    
    # Y 축 2 차 미분
    ly = F.conv2d(image, ky.to(image.device), padding=1)
    
    # 합치기
    laplacian = lx + ly
    
    return laplacian, lx, ly
```

**장점**:
- ✅ **계산 효율**: 2 차 convolution → 2 차 convolution
- ✅ **축별 분석**: X/Y 방향 2 차 미분 개별 분석
- ✅ **병렬 처리**: 축별 독립적 계산

**PyTorch 구현**:
```python
def laplacian_kernel():
    """Laplacian filter kernel"""
    return torch.tensor([[0, -1, 0],
                         [-1, 4, -1],
                         [0, -1, 0]]).float().unsqueeze(0).unsqueeze(0)

def apply_laplacian(image):
    """Apply Laplacian filter for sharpening"""
    kernel = laplacian_kernel()
    kernel = kernel.repeat(image.shape[1], 1, 1)
    
    laplacian = F.conv2d(image, kernel, padding=1)
    
    # Sharpening: original - laplacian
    sharpened = image - laplacian
    
    return sharpened
```

**용도**:
- Image sharpening
- Feature detection
- Second derivative approximation

---

### 2.3 특수 필터 (Specialized Filters)

#### Canny Edge Detector

**Steps**:
1. Gaussian blur
2. Gradient calculation (Sobel)
3. Non-maximum suppression
4. Double thresholding
5. Edge tracking by hysteresis

```python
class CannyEdgeDetector(nn.Module):
    def __init__(self, sigma=1.5, thresholds=(0.1, 0.2)):
        super().__init__()
        self.sigma = sigma
        self.high_threshold, self.low_threshold = thresholds
```

#### Prewitt Operator

```
Px = [ -1  0  1 ]
     [ -1  0  1 ]
     [ -1  0  1 ]

Py = [ -1 -1 -1 ]
     [  0  0  0 ]
     [  1  1  1 ]
```

---

### 2.4 고급 필터 (Advanced Filters)

#### Gabor Filter

**수식**:
```
G(x, y; λ, θ, φ, σ, γ) = exp(-π²·(x'² + γ²·y'²) / (2·σ²))
                         · cos(2π·x'/λ + φ)

where:
  x' = x·cosθ + y·sinθ
  y' = -x·sinθ + y·cosθ
```

**PyTorch 구현**:
```python
def gabor_kernel(frequency=0.1, theta=0, sigma=7, phase=0):
    """Generate Gabor filter kernel"""
    size = 21
    half = size // 2
    
    x = np.arange(-half, half + 1)
    y = np.arange(-half, half + 1)
    X, Y = np.meshgrid(x, y)
    
    # Rotation
    theta_rad = np.deg2rad(theta)
    x_rot = X * np.cos(theta_rad) + Y * np.sin(theta_rad)
    y_rot = -X * np.sin(theta_rad) + Y * np.cos(theta_rad)
    
    # Gabor function
    kernel = np.exp(-np.pi**2 * (x_rot**2 + sigma**2 * y_rot**2) / (2 * sigma**2))
    kernel *= np.cos(2 * np.pi * frequency * x_rot + phase)
    kernel = kernel / np.abs(kernel).max()
    
    return torch.from_numpy(kernel).float()
```

**용도**:
- Texture analysis
- Orientation detection
- Human vision modeling

#### Bilateral Filter

**특징**:
- Spatial proximity 고려
- Intensity similarity 고려
- Edge-preserving smoothing

---

## 3. CNN Convolution 연산

### 3.1 CNN 의 Convolution 구조

**Input**: (B, C, H, W)  
**Kernel**: (O, C, K, K)  
**Output**: (B, O, H', W')

```python
# PyTorch Conv2D
import torch.nn as nn

conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, 
                 stride=1, padding=1, bias=True)
```

### 3.2 1×1 Convolution

**Purpose**:
- Channel mixing
- Dimensionality reduction
- Non-linearity 추가

```python
# 1×1 conv
conv_1x1 = nn.Conv2d(128, 64, kernel_size=1)
# Equivalent to: Linear layer applied per pixel
```

**Math**:
```
For each spatial position (i, j):
output[i,j] = W @ input[i,j] + b
where W: (out_channels, in_channels)
```

### 3.3 Group Convolution

**Purpose**: Reduce computation, increase parameter independence

```python
# Group convolution
conv = nn.Conv2d(
    in_channels=256,
    out_channels=256,
    kernel_size=3,
    groups=4  # 4 separate groups
)
```

**Compute reduction**:
- Standard: (C×C×K×K) FLOPs
- Group=4: (C×(C/4)×K×K) × 4 = (C×C×K×K) FLOPs (same) but independent

### 3.4 Depthwise Convolution

**Purpose**: Parameter and computation efficiency

```python
# Depthwise separable convolution
conv_dw = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256)
conv_pw = nn.Conv2d(256, 256, kernel_size=1)
```

**Efficiency**:
- Standard: C×C×K×K
- Depthwise: C×K×K (depthwise) + C×C (pointwise)
- **Speedup**: ~8-9× for 3×3 kernel

### 3.5 Dilated Convolution

**Purpose**: Increase receptive field without increasing parameters

```python
# Dilated convolution
conv = nn.Conv2d(256, 256, kernel_size=3, dilation=2)
```

**Receptive field**:
```
Normal 3×3: RF = 3×3
Dilated=2:  RF = 5×5
Dilated=4:  RF = 9×9
```

---

### 3.6 Deformable Convolution (DCN)

**Purpose**: Adaptive sampling locations for geometric variations

**Standard Convolution**:
```
Fixed grid:
[x x x]
[x x x]
[x x x]
```

**Deformable Convolution**:
```
Learning offsets → Adaptive sampling positions
[x x .]
[. x x]
[x . x]
```

**수학적 정의**:
```
Standard:  y[p] = Σₖ wₖ · x[p + k]

Deformable: y[p] = Σₖ wₖ · x[p + k + Δpₖ]

여기서:
- k: 기본 grid points
- Δpₖ: 학습 가능한 offset
- p: output feature map position
```

**PyTorch 구현 (핵심)**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableConv2d(nn.Module):
    """Deformable Convolutional Layer v2
    
    Adds learnable offsets to sampling locations
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 stride=1, padding=1, deformable_groups=1):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.deformable_groups = deformable_groups
        
        # Offset conv layer
        offset_conv = nn.Conv2d(
            in_channels, 
            self.deformable_groups * kernel_size**2 * 2,
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        nn.init.constant_(offset_conv.weight, 0)
        nn.init.constant_(offset_conv.bias, 0)
        self.offset_conv = offset_conv
        
        # Regular conv weights
        self.w_conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride, 
            padding=padding
        )
        nn.init.normal_(self.w_conv.weight, 0, 0.01)
        
        # Modulation
        self.modulation = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract offsets
        offset = self.offset_conv(x)
        B, C, H, W = x.shape
        
        # Modulation weights
        modulation = self.modulation(x)
        
        # Note: This is a simplified DCN implementation
        # For production, use MMCV's DCNv2
        # See: https://github.com/open-mmlab/mmcv
        
        return F.conv2d(x, self.w_conv.weight, 
                       stride=self.stride, 
                       padding=self.padding)
```

**핵심 특징**:
- ✅ **Adaptive receptive field**: 객체 shape 에 따라 유연한 sampling
- ✅ **Rotation/Scale invariant**: 다양한 변환에 강인
- ✅ **Better alignment**: misaligned features 처리

**Use Cases**:
- Object detection (YOLOv4, Faster R-CNN + DCN)
- Face alignment
- Pose estimation
- Any task with geometric variations

**성능 비교**:

| Model | Architecture | COCO mAP | FLOPs (G) |
|-------|---|----|--|
| **Faster R-CNN** | Standard Conv | 39.5 | 136 |
| **Faster R-CNN + DCN** | Deformable Conv | **42.0** | 136 |
| **Mask R-CNN** | Standard Conv | 37.0 | 188 |
| **Mask R-CNN + DCN** | Deformable Conv | **40.4** | 188 |
| **YOLOv3** | Standard Conv | 33.6 | 61 |
| **YOLOv3 + DCN** | Deformable Conv | **35.2** | 61 |

**Key insight**: Deformable Convolution 은 FLOPs 증가 없이 정확도 향상! 📈

---

### 3.7 Dynamic Convolution

**Purpose**: Adaptive filters based on input content

**Standard Convolution**:
```
Fixed weights for all inputs:
y = x * W  (W is constant)
```

**Dynamic Convolution**:
```
Weights change based on input:
y = x * W(x)  (W varies with x)
```

**Implementation**:
```python
class DynamicConv2d(nn.Module):
    """Dynamic Convolutional Layer
    
    Generates conv weights dynamically based on input
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Weight generation network
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, out_channels * in_channels * kernel_size**2, 1)
        )
        
        # Base weights
        self.base_weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) / 100
        )
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate dynamic weights
        dynamic_weights = self.weight_net(x)
        dynamic_weights = dynamic_weights.view(B, self.out_channels, 
                                               self.in_channels, 
                                               self.kernel_size**2)
        
        # Reshape to convolution kernel format
        dynamic_weights = dynamic_weights.view(B, self.out_channels, 
                                               self.in_channels, 
                                               self.kernel_size, 
                                               self.kernel_size)
        
        # Apply to input
        y = F.conv2d(x, dynamic_weights)
        
        return y
```

**핵심 특징**:
- ✅ **Input-dependent**: 입력에 따라 적응적 필터링
- ✅ **Content-aware**: semantic 정보 활용
- ✅ **Flexible**: 다양한 변형 가능

**Use Cases**:
- Image classification (Dynamic Sparse Conv)
- Object detection
- Attention mechanisms

---

## 4. 효율적 Convolution 기법

### 4.1 Depthwise Separable Convolution

**Structure**:
```
Standard Conv:
Input (256) → [Conv 256→256] → Output (256)
              3×3 kernel × 256² parameters

Depthwise Separable:
Input (256) → [Depthwise 3×3] → [Pointwise 1×1] → Output (256)
              3×3×256 params       1×1×256² params
              Total: ~1/8 params
```

**MobileNet architecture**:
```python
class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        # Depthwise convolution
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, 
                           groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU6()
        
        # Pointwise convolution
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU6()
    
    def forward(self, x):
        x = self.dw(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pw(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
```

### 4.2 Fast Fourier Transform (FFT) based Convolution

**Concept**: 
```
Spatial: Convolution = O(n²)
Frequency: Multiplication = O(n log n)
```

**PyTorch 구현**:
```python
def fft_convolution(x, kernel):
    """Convolution using FFT"""
    # FFT
    x_fft = torch.fft.fft2(x)
    k_fft = torch.fft.fft2(kernel)
    
    # Element-wise multiplication
    result_fft = x_fft * k_fft
    
    # IFFT
    result = torch.fft.ifft2(result_fft).real
    
    return result
```

**Use cases**:
- Large kernels (15×15 or larger)
- Fixed kernel (filtering)
- When kernel size ≫ image size

### 4.3 Winograd Minimal Filtering Algorithm

**Purpose**: Optimize small kernels (3×3, 5×5)

**Speedup**: 
- 3×3: ~25% faster
- 5×5: ~40% faster

**Note**: More complex implementation, typically in optimized libraries (cuDNN, MKLDNN)

### 4.4 Im2Col + GEMM

**Concept**: Reshape convolution to matrix multiplication

```python
def im2col_convolution(x, kernel, kernel_size=3, stride=1, padding=1):
    """Convert convolution to GEMM using im2col"""
    B, C, H, W = x.shape
    out_H = (H + 2*padding - kernel_size) // stride + 1
    out_W = (W + 2*padding - kernel_size) // stride + 1
    
    # Im2Col
    cols = F.unfold(x, kernel_size, padding=padding, stride=stride)
    # cols: (B, C*kernel*kernel, out_H*out_W)
    
    # GEMM
    kernel_col = kernel.view(kernel.shape[0], -1)
    output = torch.matmul(kernel_col, cols)
    
    # Reshape
    output = output.view(B, -1, out_H, out_W)
    
    return output
```

---

## 5. 실전 코드 예제

### 5.1 Custom Convolution Layer

```python
class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2, bias=False)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out')
    
    def forward(self, x):
        return F.relu(self.conv(x))
```

### 5.2 Adaptive Filter Selection

```python
class AdaptiveFilterSelection(nn.Module):
    """Select appropriate filter based on local image characteristics"""
    
    def __init__(self):
        super().__init__()
        
        # Edge detector
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        
        # Blur filter
        self.gaussian = nn.Conv2d(1, 1, 5, padding=2, bias=False)
        
        # Initialize
        self.sobel_x.weight.data = torch.tensor([[[-1,0,1],[-2,0,2],[-1,0,1]]])
        self.sobel_y.weight.data = torch.tensor([[[-1,-2,-1],[0,0,0],[1,2,1]]])
        self.gaussian.weight.data = gaussian_kernel(5).unsqueeze(0)
    
    def forward(self, x):
        # Detect edges
        edges_x = torch.abs(self.sobel_x(x))
        edges_y = torch.abs(self.sobel_y(x))
        edges = edges_x + edges_y
        
        # Apply blur to smooth areas
        blur = self.gaussian(x)
        
        # Adaptive: edge regions keep original, smooth regions blurred
        alpha = torch.sigmoid(edges * 10)
        output = alpha * x + (1 - alpha) * blur
        
        return output
```

### 5.3 Multi-scale Convolution Ensemble

```python
class MultiScaleConv(nn.Module):
    """Combine convolutions at multiple scales"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.conv7 = nn.Conv2d(in_channels, out_channels, 7, padding=3)
        
        self.fusion = nn.Conv2d(out_channels * 3, out_channels, 1)
    
    def forward(self, x):
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)
        
        # Concatenate and fuse
        combined = torch.cat([c3, c5, c7], dim=1)
        output = self.fusion(combined)
        
        return F.relu(output)
```

---

## 6. 성능 최적화 가이드

### 6.1 Kernel Size 선택

| 사용 목적 | 추천 크기 | 이유 |
|------|-------|----|
| Edge detection | 3×3 | Local gradient |
| Texture analysis | 5×5~7×7 | Larger context |
| Object detection | 3×3 (stacked) | Deeper network |
| Semantic segmentation | 3×3, 5×5 | Balance |

**Rule of thumb**:
- Small objects: 3×3, 5×5
- Large objects: 7×7, or use dilated conv

### 6.2 Padding 전략

**Same padding** (output = input):
```
padding = (kernel_size - 1) / 2
```

**Valid padding** (no padding):
- Smaller output
- Less boundary artifacts

**Recommendation**:
- **Training**: Same padding for most layers
- **Boundary-sensitive tasks**: Reflect/pad replication

### 6.3 Stride 튜닝

**Downsampling**:
```
Stride 1: No downsampling
Stride 2: 50% spatial reduction
Stride 3: ~67% spatial reduction
```

**Guidelines**:
- **Early layers**: stride=1 (preserve details)
- **Later layers**: stride=2 (spatial reduction)
- **Object detection**: Additional stride=2 or FPN

### 6.4 Dilation 전략

```python
# Multi-scale receptive field
dilation_rates = [1, 2, 4, 8]

# ASPP (Atrous Spatial Pyramid Pooling)
aspp_modules = nn.ModuleList([
    nn.Conv2d(channels, channels, 1, dilation=1),
    nn.Conv2d(channels, channels, 3, dilation=2),
    nn.Conv2d(channels, channels, 3, dilation=4),
    nn.Conv2d(channels, channels, 3, dilation=8)
])
```

**Use cases**:
- Semantic segmentation
- Dense prediction tasks
- Multi-scale feature extraction

### 6.5 Memory Optimization

**Gradient checkpointing**:
```python
from torch.utils.checkpoint import checkpoint

def conv_block(x):
    return F.relu(conv(x))

# Memory efficient
output = checkpoint(conv_block, x)
```

**Reduced precision**:
```python
# Mixed precision training
from apex.amp import initialize, amp

amp.init(model, optimizer, opt_level='O1')
```

---

## 📊 성능 비교

### Convolution Types FLOPs Comparison

| Model | Type | Params (M) | FLOPs (G) | Top-1 Accuracy (%) |
|-------|-----|--|--------|---------|--|-------|
| **ResNet-50** | Standard | 25.6 | 4.1 | 76.1 |
| **MobileNet-V2** | Depthwise | 3.5 | 0.35 | 72.0 |
| **ShuffleNet-V2** | Shuffle | 2.3 | 0.15 | 72.0 |
| **EfficientNet-B0** | Depthwise | 5.3 | 0.4 | 77.0 |

### Kernel Size Impact

| Kernel | Params (3×3 vs 5×5) | Compute (3×3 vs 5×5) | mAP Gain |
|--------|--|----|----|
| **3×3** | 9 units | 9 units | Baseline |
| **5×5** | 25 units | 25 units | +0.5% |
| **7×7** | 49 units | 49 units | +0.3% |
| **2× 3×3** | 18 units | 18 units | +0.2% |

### Deformable Convolution Performance

| Model | Architecture | COCO mAP | FLOPs (G) |
|-------|---|----|--|
| **Faster R-CNN** | Standard Conv | 39.5 | 136 |
| **Faster R-CNN + DCN** | Deformable Conv | **42.0** | 136 |
| **Mask R-CNN** | Standard Conv | 37.0 | 188 |
| **Mask R-CNN + DCN** | Deformable Conv | **40.4** | 188 |
| **YOLOv3** | Standard Conv | 33.6 | 61 |
| **YOLOv3 + DCN** | Deformable Conv | **35.2** | 61 |

**Key insight**: Deformable Convolution 은 FLOPs 증가 없이 정확도 향상! 📈

---

## 📚 참고 자료

- **Deep Learning Book**, Goodfellow et al.
- **Fast AI**, Jeremy Howard
- **PyTorch Documentation**: torch.nn.Conv2d
- **TensorFlow**: tf.nn.conv2d
- **Deformable Conv**: [DCN Paper](https://arxiv.org/abs/1803.08669)
- **MMCV DCN**: [MMCV Library](https://github.com/open-mmlab/mmcv)

---

*마지막 업데이트: 2026-03-31*
*Created for deep understanding of convolution operations*
*Added: X/Y axis perspective, Direction-specific filters, Deformable & Dynamic Convolution*
