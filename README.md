# AI Driven Low Dose Image Enhancement

## Overview
<br>
*This project focuses on enhancing low-dose CT images using advanced reconstruction techniques on the LoDoPaB-CT dataset.
<br>
*Total Variation  Regularization
<br>
*Learned Primal-Dual  Reconstruction

## Methods used
### **1)1. Filtered Back Projection:**
*This is the classical, non-AI standard. When a CT scanner takes X-rays from different angles, it creates a "sinogram." FBP is the mathematical algorithm used to "smear" those X-rays back across a grid to reconstruct the 2D image.
<br>
*Signal Processing / Medical Physics.
<br>
*The Math: It uses the Radon Transform. It’s fast and predictable but creates "noisy" images if the X-ray dose is low. 
<br>
### **2)Total Variation Regularization:**
*This belongs to Iterative Reconstruction. Instead of a single mathematical formula (like FBP), this treats reconstruction as an optimization problem.
<br>
*It is used for Mathematical Optimization / Computer Vision.
<br>
*The Logic: It assumes that real-world images are "piecewise constant" (smooth areas with sharp edges). It tries to find an image that matches the scan data while 
minimizing "Total Variation" (noise).
<br>
*Role in AI: It was the precursor to AI; it’s a "hand-crafted" rule rather than a learned one.
<br>
*Optimization-based reconstruction method
<br>
*Solves inverse problem: **y = Ax + ε**
   <br>
   {*y: The noisy sinogram (raw data).
   <br>
   *A: The Radon transform (forward projection).
   <br>
   *ε: Noise}
<br>
*Promotes smooth and noise free images while preserving edges
<br>
*Implemented using iterative optimization

### **3)Learned Primal Dual:** 
*This is firmly in the AIML domain.It is a **"Deep Physcis"** approach
<br>
*How it works: It takes the classical mathematical optimization (Primal-Dual Hybrid Gradient) and "unrolls" it into a Neural Network. Instead of using hand-crafted math to remove noise, the network learns the best way to reconstruct the image from training data.
<br>
*Why it matters: It combines the reliability of physics with the power of AI, allowinguch lower radiation doses.
<br>
**Note:**
<br>
*Deep learning-based iterative reconstruction method
<br>
*Combines physics-based forward model with CNN updates
<br>
*Alternates between primal (image) and dual (data) updates
<br>
*Achieves superior reconstruction quality

## Hardware & Path Setup Tips
**GPU Acceleration:**
<br>
Your scripts use impl="astra_cuda". Ensure you have an NVIDIA GPU and that astra-toolbox is properly recognizing your drivers. You can check this by running import astra; print(astra.test()) in Python.
<br>
**Data Paths:**
<br>
Ensure the directories in your scripts (e.g., /DATA/biomedical/... or /DATA/Nith/...) exist on your machine. If you are on Windows, remember to use double backslashes \\ or raw strings r"C:\DATA\..." for paths.
<br>
**Memory Management: **
<br>
Primal-Dual networks and TV-reconstruction can be memory-intensive. If you encounter "Out of Memory" (OOM) errors, try reducing the batch_size to 1 (which it appears to be already) or reducing the iterations in the TV script for testing.
<br>
Note: If you encounter an error regarding dival during installation, ensure your pip is up to date within the environment by running python -m pip install --upgrade pip.

## Results
| Method | PSNR | SSIM |
| :--- | :---: | :---: |
| TV Regularization | ~32 dB | ~0.82 |
| Learned Primal-Dual | ~36 dB | ~0.88 |

## Dataset
This project uses the LoDoPaB-CT dataset.
<br>
**Dataset link:** [https://zenodo.org/records/3384092](https://zenodo.org/records/3384092)
<br>
Note: Dataset is not included due to large size.

## Team:
**Team Leader:** Anem GnanaGanesh
<br>
**Team Member 1:** Annam Yogitha
<br>
**Team Member 2:** Chintamani Manoj Ram Sai


