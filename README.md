# Investigation of custom CNN architecture for Schizophrenia detection using generalized sMRI data

This project aims to predict schizophrenia from MRI scan data using deep learning techniques. The MRI scans are preprocessed and augmented to increase the robustness of the model. The final model is a 3D CNN that is trained on a binary classification task (Schizophrenia vs. Healthy). This repository includes code for data preprocessing, augmentation, model training, evaluation, and cross-validation.

## Project Structure

- **nd/**: Original directories containing MRI scan files for healthy and schizophrenia data.
  - `healthy/`: Contains MRI scans for healthy subjects.
  - `schizophrenia/`: Contains MRI scans for schizophrenia subjects.
  
- **resized/**: Directory to store resized MRI data.
  - `healthy/`: Contains resized MRI scans for healthy subjects.
  - `schizophrenia/`: Contains resized MRI scans for schizophrenia subjects.

- **code.py**: Contains the full implementation of preprocessing, augmentation, model training, and evaluation.
- **code_for_5cross_validation.py**: Contains the full implementation of preprocessing, augmentation, model training, and evaluation.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn
- nibabel
- albumentations

You can install the necessary dependencies by running:

```bash
pip install tensorflow numpy matplotlib scikit-learn nibabel albumentations
```

## Data Preprocessing

The MRI data undergoes several preprocessing steps before being fed into the model:

1. **Resizing**: MRI slices are resized to a target shape (128x128) for uniformity.
2. **Slice Selection**: Middle slices (from slice 86 to 105) are extracted to focus on the region of interest in the brain.
3. **Augmentation**: Various augmentation techniques (such as random rotation, flips, brightness contrast, and gamma adjustments) are applied to improve model generalization.

## Model

The model used for schizophrenia classification is a **3D Convolutional Neural Network (CNN)**. The architecture consists of:

- **3D Convolution Layers**: Used to extract spatial features from the 3D MRI slices.
- **Max Pooling Layers**: To reduce the spatial dimensions and retain important features.
- **Fully Connected Layers**: To make the final classification based on the learned features.
- **Dropout**: To prevent overfitting.

The model is trained with **binary cross-entropy loss** and optimized using the **Adam optimizer**. The model is saved in both `.h5` and `.keras` formats for future use.

## Training and Evaluation

The data is split into training and testing sets (80% for training and 20% for testing). 

The model is evaluated using:
- **Accuracy**: The proportion of correct predictions.
- **Confusion Matrix**: To visualize the model’s performance in classifying both classes.
- **ROC Curve and AUC**: To evaluate the trade-off between sensitivity and specificity.

## Cross-validation

The project also supports **5-fold cross-validation**. This allows the model to be trained and evaluated on different subsets of the data, providing a more robust estimate of performance.

## Instructions

1. **Preprocessing**: Run the `code.py` script to preprocess and resize the MRI data.
2. **Model Training**: The model is trained on the processed data and saved in `s1.h5` and `s2.keras`.
3. **Evaluation**: After training, the model is evaluated on the test set and the accuracy, confusion matrix, and ROC curve are displayed.

## Usage

To run the code:

1. Ensure you have all the necessary dependencies installed.
2. Run the `code.py` script to preprocess the data and train the model.

```bash
python code.py
```

This will save the trained models as `s1.h5` and `s2.keras`, which can be used for further predictions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
