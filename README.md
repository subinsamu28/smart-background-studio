# smart-background-studio
🧠 Smart Background Studio: AI-based image &amp; video background remover with real-time webcam support, batch mode, and a sleek PyQt5 GUI. Powered by U²-Net, OpenCV &amp; PyTorch.

# 🧠 Smart Background Remover

> A beginner-friendly AI desktop app that removes and replaces image and video backgrounds using deep learning.  
> No green screen. No complex editing. Just Python + AI.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![GUI](https://img.shields.io/badge/GUI-Tkinter-informational)
![Model](https://img.shields.io/badge/AI-U²--Net-success)
![Status](https://img.shields.io/badge/Status-Final-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)

---

## 🖼️ What is Smart Background Remover?

Smart Background Remover is a desktop application that uses U²-Net, an AI model for salient object detection, to intelligently remove and replace backgrounds in images and webcam streams.

It allows users to:

- 🧍 Remove backgrounds from images or webcam in real-time  
- 🎨 Replace the background with a solid color, image, or transparent output  
- 🗂️ Batch process images from a folder  
- 🎥 Record webcam video with background effects  
- 🖱️ Use an interactive GUI with live preview and control sliders

---

## ✨ Key Features

✅ Foreground Segmentation (U²-Net)  
✅ Real-Time Webcam Support  
✅ Transparent PNG Export  
✅ Solid Color / Image Background Replacement  
✅ Background Blur Effect  
✅ Snapshot & Recording  
✅ Batch Image Processing  
✅ Modern, Responsive GUI

---

## 🧰 Tech Stack

| Component          | Technology              |
|-------------------|--------------------------|
| Language           | Python 3.10+             |
| AI Engine          | PyTorch + U²-Net         |
| GUI                | Tkinter                  |
| Image Processing   | OpenCV, Pillow, NumPy    |
| Video Handling     | OpenCV + VLC             |
| Performance Boost  | GPU / CUDA (if available) |

---

## 📁 Project Structure

smart-background-remover/
├── model/               # Pretrained u2net.pth  
├── src/                 # Core logic (GUI, AI, image utils)  
│   ├── main.py  
│   ├── u2net.py  
│   └── data_loader.py  
├── images/              # Input images  
├── outputs/             # Processed outputs (PNG/MP4)  
├── requirements.txt  
└── README.md

---

## 🖥️ Screenshots

<img width="1122" height="596" alt="image" src="https://github.com/user-attachments/assets/cafe955d-a7b7-4540-aac8-802a45a83d9b" />
<img width="1122" height="591" alt="image" src="https://github.com/user-attachments/assets/bdbc3f30-62d6-4fec-a6e1-380ff1b622a0" />
<img width="1122" height="593" alt="image" src="https://github.com/user-attachments/assets/f2369c56-e905-4706-b7a3-cbcbcc281bde" />
<img width="1122" height="595" alt="image" src="https://github.com/user-attachments/assets/6530fa2d-3de2-4236-9022-6bb998254f04" />

Demo video available upon request.
---

## 🚀 How to Run

1. Clone the repository:
   git clone https://github.com/your-username/smart-background-remover.git
   cd smart-background-remover

2. Install requirements:
   pip install -r requirements.txt

3. Place u2net.pth in the model/ folder  
   Download: https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view

4. Launch the app:
      python ui/app.py

---

## 🧑‍🎓 Authors & Contributors

| Name                                       | Role                    |
|--------------------------------------------|-------------------------|
| Subin Samu                                 | Developer & Presenter   |
| Abhishek Sadasivan Karumpumkal             | Developer & Testing     |
| Bharath Sankar Sankaramangalam             | Data Handling & Design  |
| Vimal Amarnath Palassery Muruganandan      | GUI & Documentation     |

---

## 🎓 Institution

Submitted to:  
Deggendorf Institute of Technology  
Prof. Dr. Patrick Glauner  
Master of Applied Computer Science, 2025  
Semester Project

---

## 📜 License

This is an academic project submitted in partial fulfillment of university requirements.  
All rights reserved © 2025 by the authors.

---

“Built to combine AI research with real-world usability — fast, friendly, and functional.”
