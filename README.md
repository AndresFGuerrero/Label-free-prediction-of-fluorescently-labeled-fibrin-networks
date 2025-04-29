# 🧠 Predicting Fluorescently Labeled Fibrin Networks From Transmission and Reflection Images

This repository contains the code associated with our deep learning method for predicting fluorescently labeled fibrin networks from reflection confocal microscopy (RCM)  and transmission images. The approach eliminates the need for fluorescent labeling by learning a mapping from label-free endogenous contrast data to synthetic fluorescence.

## 📌 Overview

Fluorescent dyes are commonly used to visualize fibrin fibers, but they can compromise cell viability and introduce photobleaching during time-lapse imaging. Our method bypasses this limitation by training a convolutional neural network (CNN) to infer the fluorescence signal directly from unstained stacks.

Key features:
- Input: RCM images captured at **three laser wavelengths** and **one transmission channel**
- Output: Predicted **fluorescent label image stacks**
- Architecture: Fully-convolutional 3D image-to-image model
- Loss: Custom hybrid loss combining Lp norm and structural similarity
- Performance: Recovers 3D fibrous architectures with submicron precision  

## 🧪 Applications

- Label-free visualization of ECM in biomaterials like **fibrin**  fibers
- Live-cell compatible imaging without photobleaching
- Automated generation of virtual fluorescence for downstream analysis

  
## 🗂 Repository Structure

```plaintext
root/
├── src/                    # Core model preprocessing, patching, architecture, training, and inference code
│   ├── z_score_normalization.py
│   ├── patching.py
│   ├── architecture_and_training.py
│   └── inference.py
├── data/                   # Sample raw tiff files, normalized data, patched data, and input, ground_truth, output data.
├── notebooks/              # Jupyter notebooks with example workflows
├── requirements.txt        # Python dependencies
├── .gitignore              # Files and directories to exclude from Git
├── LICENSE                 # Open source license
└── README.md               # This file

