# Sat-Img-Vis-Tra
Satellite Images Visual Transformer, attempt to classify with a ViT, comparison with pretrained vs in house trained and CNN method, this repository contains other people's code too as a reference point, individual licenses may vary, will be updated soon


TESTED ON:
    Processor: AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD
    Machine: AMD64
    Physical cores: 8
    Logical processors: 16
    CPU frequency per core (MHz): [2516.0]
    Average CPU frequency (MHz): 2516.0
    RAM_GB: 15.31
    OS_System: Windows
    OS_Version: Windows 11
    OS_Node: Nitro-V16-4060
    OS_Release: 10
    OS_Machine: AMD64
    OS_Processor: AMD64 Family 25 Model 117 Stepping 2, AuthenticAMD
    Python_Version: 3.10.0
    PyTorch_Version: 2.8.0+cu126
    Timm_Version: 1.0.22
    Safetensors_Version: 0.7.0
    Scikit-learn_Version: 1.7.2
    NumPy_Version: 1.23.0
    CUDA_Available: True
    CUDA_Version: 12.6
    GPU_Name: NVIDIA GeForce RTX 4060 Laptop GPU



HOW TO RUN:
This repo DOES NOT contain the dataset, open Kaggle Dataset page and download the it from there, put it in a folder in this folder and run all the ipynb

Dataset used:
https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification/data


Overview:
Problem Statement:
The problem at hand is simple, developing a model to classify the images in these 4 classes with the aid of a Visual Transformer.
Pretrained model chosen: vit_base_patch16_224

Dataset Overview:
The dataset used is "Satellite image Classification Dataset-RSI-CB256", it has 4 different classes mixed from Sensors and Google Maps snapshots.
Labels and quantities, confirmed through dataset looping:
Class 'cloudy':  Resolution 256x256 : 1500 images
Class 'desert':  Resolution 256x256 : 1131 images
Class 'green_area':  Resolution 64x64 : 1500 images
Class 'water':  Resolution 64x64 : 1500 images

Data Augmentation:
Attempted, failed partially due to generally low number of input data, in my pipeline I tried:
Resize, which is obviously needed for 224x224 resolution to feed to the ViT model.
RandomHorizontalFlip and RandomVerticalFlip, no effect
RandomRotation, ended up hurting performance, probably because resolution is too low and rotating means reducing quality too much, introducing potential artifacts in the image
ColorJitter, attempted, turned off because of reduction in accuracy
Normalize(mean=[0.485, 0.456, 0.406],                         std=[0.229, 0.224, 0.225]), seems to be standard and not something worth fiddling with
CenterCrop, has been attempted, but possibly due to the low resolution of the input image it was just not a good addition to the pipeline

More details about ColorJitter:
As a mistake I forgot to turn off Jitter from old code copy-paste and run the same pretrained model with and without it:
With jitter it achieves barely an 86% accuracy, with a big confusion between Green_Area and Water, probably because they look very similar and not having the chance to rely on color makes them easier to mistake with one another (300 confusion which is 1/5 cases of Water mistakenly attributed to Green_Area, perhaps the name itself should imply that jitter wasn't going to work), totaling runs that average at around 75s.
Without Jitter we have better accuracy and faster training speed: 89% accuracy and train time slightly above 53s on average, while the confusion matrix shows a slight improvement (261 vs 300 as earlier).


Training:
In order to more easily follow the development and testing there is a folder called "Results" containing PLT graphs and confusion matrixes for the testing, each folder is named in a way to better represent the state in which the input is handled and with which model.
Once established a baseline I tried augmenting, more details are available above, but it became quickly apparent that it was not giving much room for improvement, actually it was hurting performance.

Locally Trained:
    Architecture:
        Images → PatchEmbedding (16x16 patches via Conv2d) → CLS token + PositionalEncoding → 6 Transformer blocks (MultiHeadAttention + MLP + LayerNorm residuals) → Classification head

    Training Process:
        224x224 images, batch_size=16, AdamW (lr=3e-4), CrossEntropyLoss
        20 epochs, tracks loss/accuracy/precision/recall/F1 per epoch
        Mixed precision (torch.amp.autocast) on RTX 4060 for speed

    Key Components:
        Custom VisionTransformer class (196 patches + 1 CLS token = 197 seq len)
        Satellite images resized/normalized, num_workers=0 (Windows stability)
        Achieves ~92-96% accuracy after 10-20 epochs
        Trained from random weights (not pretrained), data-hungry ViT on small dataset (5.6k images) with mixed resolutions (256x256/64x64).

Pretrained:
    Architecture:
        Images → timm 'vit_base_patch16_224' (pretrained ImageNet weights) → replaced classification head (Linear layer for 4 satellite classes)

    Training Process:
        224x224 images, batch_size=16, AdamW (lr=3e-4), CrossEntropyLoss
        20 epochs, tracks loss/accuracy/precision/recall/F1 per epoch
        Mixed precision (torch.amp.autocast) on RTX 4060 for speed

    Key Components:
        Pretrained timm ViT-Base (86M params, 12 transformer blocks)
        Satellite images resized/normalized (ImageNet stats), num_workers=0 (Windows stability)
        Transfer learning → fast convergence from strong ImageNet features
        Same RSI-CB256 dataset (5.6k images, mixed 256x256/64x64 resolutions)

Key differences (Local vs Pretrained):
    Custom-built (6 blocks, ~22M params?)    vs     timm pretrained (12 blocks, 86M params)
    Random weights → slow start     vs      ImageNet weights → instant strong features
    Data-hungry (needs 10+ epochs)      vs      Transfer learning (peaks in 5-10 epochs)
    Architecture experimentation        vs      Production-ready baseline




Final verdict:
The pretrained version is certainly a solid model, but suffers from disadvantages against the locally trained model such as:
Forced to run at 224x224, while the locally trained also uses mixed (and then resized to 72x128) resolutions in the input, which slows down considerably the epoch time (pretrained needs 53s per epoch with accuracy at 89%, locally trained can go as low as 15s per epoch, see sample 6 and 7 in results folder, with accuracy at 95%).
Pretrained version needs more fiddling and required more precise tuning for better adapt to my GPU in terms of num_workers and other parameters that required debugging, and obviously requires to download an external file.
Given the amount of work needed for an inferior result, even with a very simple pretrained model, it is not very wise to install a heavier ViT model to be compared with a pretrained model that would need several times the training time to have maybe minimal improvement.

When compared to a simple locally trained CNN with an accuracy score of 93% it's certainly promising, but after leaving the CNN running for the first 20 epoch runs it had 80s epoch time on start vs 22s on average for next cycle, rerunning the local model again for 20 new epochs wielded better results:
Epoch [20/20] Loss: 0.0621 Accuracy: 0.9791 Precision: 0.9792 Recall: 0.9791 F1-score: 0.9791 Epoch Time: 22.36 seconds
But running for 50 epochs wielded even better results:
Epoch [45/50] Loss: 0.0548 Accuracy: 0.9825 Precision: 0.9825 Recall: 0.9825 F1-score: 0.9825 Epoch Time: 23.71 seconds
Epoch [46/50] Loss: 0.0502 Accuracy: 0.9840 Precision: 0.9840 Recall: 0.9840 F1-score: 0.9840 Epoch Time: 24.08 seconds
Epoch [47/50] Loss: 0.0656 Accuracy: 0.9785 Precision: 0.9785 Recall: 0.9785 F1-score: 0.9785 Epoch Time: 23.02 seconds
Epoch [48/50] Loss: 0.0618 Accuracy: 0.9807 Precision: 0.9807 Recall: 0.9807 F1-score: 0.9807 Epoch Time: 23.65 seconds
Epoch [49/50] Loss: 0.0486 Accuracy: 0.9853 Precision: 0.9854 Recall: 0.9853 F1-score: 0.9853 Epoch Time: 23.41 seconds
Epoch [50/50] Loss: 0.0657 Accuracy: 0.9816 Precision: 0.9816 Recall: 0.9816 F1-score: 0.9816 Epoch Time: 23.26 seconds

In short: A locally trained ViT is quite promising, even beating a pretrained solution finetuned for the new dataset, but a CNN is more robust and better tailored to handle the task

Personal notes:
Many things are janky and could have just been rewritten better as classes to be reused, like graphs and metrics object at the very least, but the main focus of the project is research and not to output the best and cleanest code


