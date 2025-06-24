# OCT-Line-Masking-Dot-Analysis-Pipeline

This repository contains a suite of Python scripts designed for advanced image processing and analysis, particularly focusing on segmentation, enhancement, and feature extraction from grayscale images. The pipeline is built using popular libraries like OpenCV, NumPy, scikit-image, and TensorFlow, offering robust functionalities for various image analysis tasks.

## Project Structure and Overview
The project is composed of several modular scripts, each responsible for a specific stage in the image processing workflow:

- **image_enhancer.py: Handles image loading, enhancement, and preparation.

- **predict.py: Performs patch-wise semantic segmentation using a pre-trained TensorFlow/Keras model.

- **createmask.py: Generates binary masks from enhanced grayscale images.

- **mask_cleaner.py: Cleans and refines generated binary masks by removing spurious components.

- **line_dot_marker.py: Detects and marks linear components within binary masks, outputting data and visualization

![oct](https://github.com/user-attachments/assets/f17775c8-4728-4138-9f65-0e8d5fd4b134)

## Setup and Installation
- **Python 3.10.8**  
- `git`, `pip`, `WSL`

## 🔧 Installation
```bash
# 1) Clone this repo and enter it
git clone https://github.com/Araf01/OCT-Line-Masking-Dot-Analysis-Pipeline.git
cd OCT-Line-Masking-Dot-Analysis-Pipeline

# 2) Create & activate a Python 3.10.8 virtual environment
# macOS/Linux:
python3.10 -m venv .venv
#Windows:
py -3.10 -m venv .venv (or python -m venv .venv)

# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\Activate.ps1

# 3) Upgrade pip & install dependencies
pip install --upgrade pip 
pip install -r requirements.txt

# 4)Download Microsoft Visual C++ Redistributable if needed
 #https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

#5) Run demo.py (click yes if it asks to install ipykernel)
```
