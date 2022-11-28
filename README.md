# Joint Cross-Attention Network with Deep Modality Prior for Fast MRI Reconstruction 

Abstract:

Current deep learning-based reconstruction models for accelerated multi-coil magnetic resonance imaging (MRI) mainly focus on subsampled k-space data of single modality using convolutional neural network (CNN). Although dual-domain information and data consistency constraint are usually adopted to improve image reconstruction, the performance of existing models is still limited mainly by three factors: inaccurate estimation of the coil sensitivity, inductive bias of CNN, and inadequate utilization of structural prior. To tackle these challenges, we propose an unrolling-based joint Cross-Attention Network (jCAN) under deep guidance of pre-acquired intra-subject data. Particularly, to improve the performance of coil sensitivity estimation, we simultaneously optimize the latent MR image and sensitivity map (SM). Besides, we introduce Gating layer and Gaussian layers into SM estimation to alleviate the “defocus” and “over-coupling” issues and further improve the quality of the estimated SM. To enhance the representation ability of the proposed model, we deploy Vision Transformer (ViT) and CNN in the image and k-space domains, respectively. Moreover, we exploit pre-acquired intra-subject MRI data as prior information to guide the reconstruction of subsampled target modality by resorting to the introduced self- and cross-attention scheme. Experimental results demonstrate that the proposed jCAN outperforms the state-of-the-art methods by a large margin in terms of SSIM and PSNR for different sampling patterns and acceleration rates.

#### To train your model ####
python3 train.py  --force_gpu --protocals T2 T1Flair --logdir /PATH_to_SAVE  --shape 320  --sparsity 0.125 --mask equispaced --coils 24 

#### To test your model ####
python3 eval.py  --protocals T2 T1Flair --model_path /PATH_to_Pretrained_Model  --save_path /PATH_to_SAVE
