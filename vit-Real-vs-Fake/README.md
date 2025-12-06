# VISION TRANSFORMER - REAL VS AI IMAGE CLASSIFICATION

## Project Overview

This project aims to develop a binary classification model to distinguish between AI-generated images (FAKE) and real images (REAL). The pipeline involves fine-tuning a Vision Transformer (ViT), training it on the CIFAKE dataset, and performing a comprehensive evaluation of its performance.

## Requirements

To run this project, the following libraries are required, as specified in the dependencies:
* torch
* torchvision
* transformers
* datasets
* scikit-learn
* matplotlib
* seaborn
* pillow
* tqdm

## Model Configuration and Fine-Tuning

The model used is a Fine-Tuned Vision Transformer

| Parameter                | Detail                        | 
| -------------------------|:-----------------------------:|
| **Base Model**           | *google/vit-base-patch16-224* |
| **Strategy**             | *Backbone Freezed - Only the classifier head is trained*|
| **Trainable Parameters** |*1,538 out of 85,800,194 total parameters*|
| **Classes**              | 2 (REAL, FAKE)                     |

The ViTFineTuner uses the pre-trained ViTForImageClassification model and optionally freezes the backbone weights to train only the last classification layer

## Dataset and Preprocessing

The project utilizes the CIFAKE dataset, located in the *./data/cifake* directory

#### Dataset Structure

The directory structure is:

* data/cifake/
    * train/
       * REAL/
       * FAKE/
    * test/
       * REAL/
       * FAKE/

The dataset can be obtained using the Kaggle command: 

`kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images`

#### Data Loading and Split

* Total Training Images: 100,000 images loaded from *./data/cifake/train.*
    * **REAL**: 50,000 images.
    * **FAKE**: 50,000 images.

* Training/Validation Split: A 20% split was used for validation.
    * **Training Set Size**: 80,000 images.
    * **Validation Set Size**: 20,000 images.

* Preprocessing: The process used custom transforms instead of the ViTImageProcessor.

    * **Base Transforms** (Validation/Test): Include *resize* and *normalization*.
    * **Training Transforms** (Augmentation): Additionally include *RandomHorizontalFlip*, *RandomRotation*, and *ColorJitter*.
* Test Set: 20,000 images were loaded from *./data/cifake/test* for final evaluation (10,000 REAL, 10,000 FAKE).

## Training Details

The training was executed for 10 epochs.

| Parameter                   | Value      | 
| ----------------------------|:----------:|
| **Epochs**                  | *10*       |
| **Learning Rate**           | *2e−05*    |
| **Weight Decay**            | *0,01*     |
| **Early Stopping Patience** | *5*        |
| **Batch Size**              | *128*      |
| **Device**                  | *cuda*     |


## Training Results
The training process was completed in a total time of 161 minutes.

| Metric           | Final Training  | Best Validation  |
| -----------------|:---------------:|:----------------:|
| **Loss**         | *0.3312*        | *0.3265*         |
| **Accuracy**     | *0.8628*        | *0.8659*         |
| **Precision**    | *0.8628*        | *0.8660*         |
| **F1-Score**     | *0.8628*        | *0.8659*         |

During training, the validation metrics generally showed consistent improvement. For example, the loss decreased steadily from 0.5329 (Epoch 1) to 0.3265 (Epoch 10)


## Evaluation Results

The final evaluation was performed on the separate test set of 20,000 images.

#### Final Metrics

The key metrics achieved by the model on the test set are:

| Metric                | Value           |
| ----------------------|:---------------:|
| **Overall Accuracy**  | *0.8181*        |
| **ROC AUC Score**     | *0.9020*        |
| **PR AUC Score**      | *0.8971*        |

#### Per-Class Metrics

| Class     | Precision  | Recall   | F-1 Score| Class Accuracy  |
| ----------|:----------:|:--------:|:--------:|:---------------:|
| **REAL**  | *0.8248*   | *0.8076* | *0.8161* | *0.8076*        |
| **FAKE**  | *0.8115*   | *0.8285* | *0.8199* | *0.8285*        |


#### Confusion Matrix (Test Set)

The Confusion Matrix based on the test set evaluation is:

| True Label (Actual)  | Predicted REAL  | Predicted FAKE   |
| ---------------------|:---------------:|:----------------:|
| **REAL**             | *8076*          | *1924*           |
| **FAKE**             | *1715*          | *8285*           |




## Results Artifacts

All resulting files and visualizations are saved in the configured directory *./results*

| Filename                    | Content      | 
| ----------------------------|:----------:|
| **best_model.pth**          | *Trained model based on best Validation Loss* |
| **epoch_training_curves.png** | *Training and Validation metrics over epochs* |
| **batch_training_curves.png** | *Metrics recorded batch-wise* |
| **confusion_matrix.png**      | *Confusion Matrix (Test Set)* |
| **roc_curve.png**             | *Precision-Recall Curve*      |
| **metrics.json**              | *Final evaluation metrics in JSON format* |


## Usage and Launch

The project pipeline is executed via a main script.
The script supports command-line arguments to configure the pipeline:

| Argument                   | Default Value      | Description|
| ---------------------------|:------------------:|:----------------:|
| **--data_dir**             | *./data/cifake*    | *Path to dataset directory*|
| **--batch_size**           | *128*              | Batch size for training |
| **--epochs**               | *2*                | Number of training epochs  |
| **--lr**                   | *2e−5*             | Learning rate     |
| **--model**                | *google/vit-base-patch16-224*| Pre-trained model name  |





