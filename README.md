# AI Driven Low Dose Image Enhancement

## Overview
<br>
This project focuses on enhancing low-dose CT images using advanced reconstruction techniques on the LoDoPaB-CT dataset.
<br>
Total Variation  Regularization
<br>
Learned Primal-Dual  Reconstruction

## Methods used
### **1)Total Variation Regularization:**
Optimization-based reconstruction method
<br>
Solves inverse problem: **y = Ax + ε**
   <br>
   {*y: The noisy sinogram (raw data).
   <br>
   *A: The Radon transform (forward projection).
   <br>
   *ε: Noise}
<br>
Promotes smooth and noise free images while preserving edges
<br>
Implemented using iterative optimization

### **2)Learned Primal Dual:** 
Deep learning-based iterative reconstruction method
<br>
Combines physics-based forward model with CNN updates
<br>
Alternates between primal (image) and dual (data) updates
<br>
Achieves superior reconstruction quality

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

