# Daily Updates on 3D-Related Papers

This repository automatically fetches new or updated arXiv papers in the [cs.CV] category every day, checks if they are relevant to "3D reconstruction" or "3D generation" via ChatGPT, and lists them below.

## How It Works
1. A GitHub Actions workflow runs daily at 09:00 UTC.  
2. It uses the script [fetch_cv_3d_papers.py](fetch_cv_3d_papers.py) to:  
   - Retrieve the latest arXiv papers in cs.CV.  
   - Use ChatGPT to filter out those related to 3D reconstruction/generation.  
   - Update this README.md with the new findings.  
   - Send an email via 163 Mail if any relevant papers are found.  

# Paper List
## Arxiv 2025-01-30

Relavance | Title | Research Topic | Keywords | Pipeline
|------|---------------|----------------|----------|---------|
8.5 | [[8.5] 2501.18594v1 Foundational Models for 3D Point Clouds: A Survey and Outlook](http://arxiv.org/abs/2501.18594v1) | 3D reconstruction 3D重建 | 3D point clouds<br>foundational models<br>3D视觉理解<br>基础模型<br>3D点云 | input: 3D point clouds 3D点云<br>step1: review of foundational models FMs 基础模型的回顾<br>step2: categorize use of FMs in 3D tasks 分类基础模型在3D任务中的应用<br>step3: summarize state-of-the-art methods 总结最新的方法<br>output: comprehensive overview of FMs for 3D understanding 输出：基础模型在3D理解中的综合概述 |
8.5 | [[8.5] 2501.18162v1 IROAM: Improving Roadside Monocular 3D Object Detection Learning from Autonomous Vehicle Data Domain](http://arxiv.org/abs/2501.18162v1) | Autonomous Driving 自动驾驶 | 3D object detection<br>autonomous driving<br>3D对象检测<br>自动驾驶 | input: roadside data and vehicle-side data<br>In-Domain Query Interaction module learns content and depth information<br>Cross-Domain Query Enhancement decouples queries into semantic and geometry parts<br>outputs enhanced object queries |
8.5 | [[8.5] 2501.18110v1 Lifelong 3D Mapping Framework for Hand-held & Robot-mounted LiDAR Mapping Systems](http://arxiv.org/abs/2501.18110v1) | 3D reconstruction 三维重建 | 3D Mapping<br>3D Reconstruction<br>Lifelong Mapping<br>激光雷达<br>三维映射<br>三维重建<br>终身映射 | Input: Hand-held and robot-mounted LiDAR maps 输入：手持和机器人安装的激光雷达地图<br>Dynamic point removal algorithm 动态点去除算法<br>Multi-session map alignment using feature descriptor matching and fine registration 多会话地图对齐，使用特征描述符匹配和精细配准<br>Map change detection to identify changes between aligned maps 地图变化检测以识别对齐地图之间的变化<br>Map version control for maintaining current environmental state and querying changes 地图版本控制，用于维护当前环境状态和查询变化 |
8.0 | [[8.0] 2501.18595v1 ROSA: Reconstructing Object Shape and Appearance Textures by Adaptive Detail Transfer](http://arxiv.org/abs/2501.18595v1) | Mesh Reconstruction 网格重建 | Mesh Reconstruction<br>3D reconstruction<br>网格重建<br>三维重建 | input: limited set of images 限制的图像集<br>step1: optimize mesh geometry 优化网格几何形状<br>step2: refine mesh with spatially adaptive resolution 使用空间自适应分辨率细化网格<br>step3: reconstruct high-resolution textures 重新构建高分辨率纹理<br>output: textured mesh with detailed appearance 带有详细外观的纹理网格 |
7.5 | [[7.5] 2501.18590v1 DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models](http://arxiv.org/abs/2501.18590v1) | Rendering Techniques 渲染技术 | Inverse Rendering<br>Forward Rendering<br>Video Diffusion Models<br>Inverse渲染<br>正向渲染<br>视频扩散模型 | input: real-world videos, 真实世界视频<br>step1: estimate G-buffers using inverse rendering model, 使用逆向渲染模型估计G-buffer<br>step2: generate photorealistic images from G-buffers, 从G-buffer生成照片级真实图像<br>output: relit images, material edited images, realistic object insertions, 重新照明图像，材料编辑图像，逼真的物体插入 |
7.5 | [[7.5] 2501.18315v1 Surface Defect Identification using Bayesian Filtering on a 3D Mesh](http://arxiv.org/abs/2501.18315v1) | Mesh Reconstruction 网格重建 | 3D Mesh<br>Mesh Reconstruction<br>3D网格<br>网格重建 | input: CAD model and point cloud data 输入：CAD模型和点云数据<br>transform CAD model into polygonal mesh 将CAD模型转换为多边形网格<br>apply weighted least squares algorithm 应用加权最小二乘算法<br>estimate state based on point cloud measurements 根据点云测量估计状态<br>output: high-precision defect identification 输出：高精度缺陷识别 |
7.5 | [[7.5] 2501.17636v2 Efficient Interactive 3D Multi-Object Removal](http://arxiv.org/abs/2501.17636v2) | 3D reconstruction 三维重建 | 3D scene understanding<br>multi-object removal<br>3D场景理解<br>多对象移除 | input: selected areas and objects for removal 选定的移除区域和对象<br>step1: mask matching and refinement mask 匹配和细化掩码步骤<br>step2: homography-based warping 同伦变换基础的扭曲<br>step3: inpainting process 修复过程<br>output: modified 3D scene 修改后的3D场景 |
7.0 | [[7.0] 2501.18246v1 Ground Awareness in Deep Learning for Large Outdoor Point Cloud Segmentation](http://arxiv.org/abs/2501.18246v1) | 3D reconstruction  三维重建 | point cloud segmentation<br>outdoor point clouds<br>semantic segmentation<br>point cloud<br>关键点云分割<br>户外点云<br>语义分割<br>点云 | input: outdoor point clouds 户外点云<br>compute Digital Terrain Models (DTMs) 计算数字地形模型<br>employ RandLA-Net for segmentation 使用 RandLA-Net 进行分割<br>evaluate performance on datasets 评估在数据集上的表现<br>integrate relative elevation features 集成相对高程特征 |
6.5 | [[6.5] 2501.18494v1 Runway vs. Taxiway: Challenges in Automated Line Identification and Notation Approaches](http://arxiv.org/abs/2501.18494v1) | Autonomous Driving 自动驾驶 | Automated line identification 自动化线识别<br>Convolutional Neural Network 卷积神经网络<br>runway markings 跑道标记<br>autonomous systems 自动化系统<br>labeling algorithms 标记算法 | input: runway and taxiway images 跑道和滑行道图像<br>Step 1: color threshold adjustment 颜色阈值调整<br>Step 2: refine region of interest selection 精细化感兴趣区域选择<br>Step 3: integrate CNN classification 集成CNN分类<br>output: improved marking identification 改进的标记识别 |


## Newly Found Papers on ...
(Older entries get replaced automatically when the script runs again.)