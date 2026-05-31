# 主动脉中心线分析系统

从CT扫描中自动分割主动脉、三维表面重建、中心线提取和直径测量的端到端流程工具。它结合了基于深度学习的主动脉分割模型（3D RUNet）与血管建模工具包（VMTK），并提供交互式3D可视化界面。

> [English Documentation](README.md)

## 功能特性

- **DICOM 转 NIfTI** — 批量加载CT切片并转换为体数据NIfTI格式
- **深度学习分割** — 使用3D残差U-Net（RUNet）自动分割主动脉
- **3D网格生成** — 使用Marching Cubes提取表面并进行拉普拉斯平滑，导出为STL格式
- **交互式中心线提取** — 在3D主动脉表面上通过点击选择起点和终点，由VMTK计算中心线
- **实时直径测量** — 鼠标悬停在中心线上任意位置即可查看该处的主动脉直径
- **多模式3D可视化** — 支持切换仅表面、仅中心线和同时显示三种模式
- **CSV数据导出** — 中心线坐标和直径数据保存为CSV文件，方便后续分析
- **日志记录** — 完整的会话日志记录到 `results/log.txt`

## 界面截图

| 步骤 | 说明 |
|------|------|
| ![分割](img/1.png) | **第一步：主动脉分割** — 使用RUNet深度学习模型从CT DICOM数据中自动分割出主动脉。 |
| ![选点](img/2.png) | **第二步：选择起点和终点** — 在3D主动脉表面上，将鼠标放在目标位置，按空格键分别选择起点（蓝色球体）和终点（黄色立方体）。 |
| ![中心线和直径](img/3.png) | **第三步：中心线和直径** — 计算得到的中心线以绿色显示。鼠标悬停在中心线上任意位置即可查看该处的主动脉直径。 |

## 项目结构

```
aorta-centerline/
├── main.py                  # PyQt5 GUI应用（主入口）
├── model/
│   └── Iteration_3500       # 预训练RUNet模型权重
├── networks/
│   ├── runet.py             # 3D残差U-Net网络架构
│   └── building_blocks.py   # 残差块和工具函数
├── inference/
│   ├── inference_sega.py    # 分割推理引擎
│   └── meshing.py           # NIfTI → 网格转换
├── preprocessing/
│   └── preprocessing_volumetric.py  # 重采样和归一化
├── augmentation/
│   ├── volumetric.py        # 体数据增强变换
│   ├── aug.py               # 变换组合工具
│   └── torchio.py           # 基于TorchIO的增强流程
├── helpers/
│   ├── utils.py             # 归一化和数据读写工具
│   ├── cost_functions.py    # 训练损失函数（Dice、Hausdorff等）
│   └── hausdorff.py         # Hausdorff距离损失实现
├── paths/
│   └── paths.py             # 路径配置
├── results/                 # 输出目录（运行时自动创建）
│   ├── output_volume.nii.gz # 转换后的NIfTI体数据
│   ├── pred_mask.nii.gz     # 分割掩码
│   ├── pred_mask_3d_model.stl  # 3D表面网格
│   ├── centerline_diameters.csv # 中心线坐标和直径数据
│   └── log.txt              # 会话日志
└── img/                     # 文档图片
```

## 安装步骤

### 步骤一：创建 Conda 环境

```bash
conda create -n vmtk -c conda-forge python=3.10 itk=5.3 vmtk vtk pyqt pyvista pyvistaqt scikit-image nibabel SimpleITK pandas
```

> **注意：** 该命令会安装 vmtk-1.5.0 及配套的 ITK 5.3 环境。VMTK 官网最新版本为 1.4.0，conda-forge 提供的 1.5.0 版本包含了本项目所需的 `vmtkCenterlines` 模块。

### 步骤二：安装 PyTorch 和其他依赖

**CPU 版本：**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**GPU 版本（CUDA 12.6）：**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**其他依赖包：**
```bash
pip install torchsummary torchio
```

### 步骤三：准备主动脉分割模型

将训练好的主动脉分割模型放入 `model/` 目录：

```
model/Iteration_3500
```

默认模型文件名为 `Iteration_3500`，代表模型训练了 3500 轮。

- **模型训练源码：** [SEGA_MW_2023](https://github.com/MWod/SEGA_MW_2023/tree/main)
- **预训练模型下载：** [百度网盘](https://pan.baidu.com/s/1f3R1wVMRo6gShNpKLl_h0g?pwd=AORT) 提取码：`AORT`

## 使用方法

### 启动 GUI

```bash
python main.py
```

**双显卡系统（NVIDIA + Intel 双 GPU）：**
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only python main.py
```

### 操作步骤

1. **加载数据** — 点击"📁 加载DICOM数据"，选择包含 CT DICOM 文件的文件夹。系统将自动：
   - 将 DICOM 转换为 NIfTI 格式
   - 运行深度学习分割
   - 生成 3D STL 表面网格
   - 加载并平滑处理网格用于显示

   也可以点击"🔍 加载STL模型"直接加载已有的 STL 文件。

2. **选择起点和终点** — 将鼠标移动到主动脉表面，可看到预览标记：
   - 按**空格键**选择**起点**（蓝色球体）
   - 移动到另一位置，再按**空格键**选择**终点**（黄色立方体）
   - 按 **R 键**重新选择

3. **计算中心线** — 点击"📐 计算中心线"按钮或按 **C 键**，VMTK 将计算两点之间通过血管腔内的最短路径。

4. **查看结果** — 计算完成后：
   - **"👁️ 全部显示" / 按键 1** — 同时显示主动脉表面和中心线
   - **"📈 仅中心线" / 按键 2** — 仅显示中心线
   - **"🔍 仅主动脉" / 按键 3** — 仅显示主动脉表面
   - 在"仅中心线"模式下，将鼠标悬停在中心线上可实时查看任意位置的直径

### 输出文件

所有结果保存在 `results/` 目录中：

| 文件 | 说明 |
|------|------|
| `output_volume.nii.gz` | 由 DICOM 转换得到的 NIfTI 体数据 |
| `pred_mask.nii.gz` | 二值分割掩码 |
| `pred_mask_3d_model.stl` | 3D 主动脉表面网格（平滑处理后） |
| `centerline_diameters.csv` | 中心线点坐标和直径数据（列：PointIndex, X, Y, Z, Radius, Diameter） |
| `log.txt` | 完整会话日志 |

### 快捷键

| 按键 | 功能 |
|------|------|
| `空格` | 在主动脉表面上选择起点/终点 |
| `C` | 计算中心线 |
| `R` | 重新选择点 |
| `1` | 同时显示表面和中心线 |
| `2` | 仅显示中心线 |
| `3` | 仅显示表面 |

## 技术细节

- **分割模型：** 3D 残差 U-Net（RUNet），6级编解码器架构
- **输入归一化：** CT窗宽 [-100, 400] HU
- **推理分辨率：** 224 × 224 × 300 体素
- **后处理：** 二值阈值（0.5）+ 小连通域去除
- **表面平滑：** 拉普拉斯平滑（50次迭代，松弛因子0.1）
- **中心线方法：** VMTK `vmtkCenterlines`，使用 pointlist 种子选择器和最大内切球半径

## 开源协议

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

本项目建立在以下优秀工作的基础之上：

- **[SEGA_MW_2023](https://github.com/MWod/SEGA_MW_2023)** — 由 [MWod](https://github.com/MWod) 开发的主动脉分割模型训练框架。感谢其开源训练代码和方法论。本项目使用的预训练 RUNet 模型即基于 SEGA 框架训练。
- **[VMTK](http://www.vmtk.org/)** — 血管建模工具包，用于中心线提取和血管分析。
- **[TorchIO](https://github.com/fepegar/torchio)** — 医学图像增强和预处理库，用于数据增强流程。
