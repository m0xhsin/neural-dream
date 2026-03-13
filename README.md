# neural-dream-
DeepDreamer is a Python project that applies the Deep Dream algorithm to images using TensorFlow and MobileNetV2. Users can enhance patterns and textures by specifying the target layer and number of iterations. The project saves output images, demonstrating neural network feature visualization and creative image transformations.
# 🌌 DeepDreamer

![Python](https://img.shields.io/badge/Python-3-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Transform your images with DeepDreamer: Creative neural network visualizations using TensorFlow & MobileNetV2**

DeepDreamer applies the Deep Dream algorithm to images using **TensorFlow** and **MobileNetV2**. It enhances patterns and textures by maximizing activations in chosen layers of a pre-trained neural network. This project demonstrates **neural network feature visualization, deep learning creativity, and image transformation techniques**.

---

## 🚀 Features
- 🖼️ Apply Deep Dream to any input image  
- 🎨 Choose target layer and number of steps for custom transformations  
- 💻 Uses MobileNetV2 pre-trained model  
- 🔁 Multi-octave processing for richer details  
- 📂 Saves output images in the root folder  
- ⚡ Temporary images saved during processing for step visualization  

---

## 🛠️ Tech Stack
- Python 3  
- TensorFlow 2.x  
- NumPy  
- PIL (Python Imaging Library)  

---

## ▶️ How to Run

```bash
python3 dream.py <image_path> <layer_index> <steps>


Example:
```bash
python3 dream.py images/input.jpg 50 40
