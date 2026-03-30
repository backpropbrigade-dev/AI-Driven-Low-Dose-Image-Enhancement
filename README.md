# AI Driven Low Dose Image Enhancement
<br>
## Overview
This project focuses on enhancing low-dose CT images using advanced reconstruction techniques on the LoDoPaB-CT dataset.

## Methods used
### 1)Total Variation Regularization:
* Optimization-based reconstruction method
*Solves inverse problem: **y = Ax + ε**
*Promotes smooth and noise-free images while preserving edges
*Implemented using iterative optimization

### 2)Learned Primal Dual 
*Deep learning-based iterative reconstruction method
*Combines physics-based forward model with CNN updates
*Alternates between primal (image) and dual (data) updates
*Achieves superior reconstruction quality

## Results
| Method | PSNR | SSIM |
| :--- | :---: | :---: |
| TV Regularization | ~32 dB | ~0.82 |
| Learned Primal-Dual | ~36 dB | ~0.88 |

## Dataset
*This project uses the LoDoPaB-CT dataset.
**Dataset link:** [https://zenodo.org/records/3384092](https://zenodo.org/records/3384092)
*Note: Dataset is not included due to large size.

