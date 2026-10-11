# Smart Document Scanner

A Computer Vision application built with Python and OpenCV to transform document photographs into enhanced digital scans.

## Features

- Load document images in JPG and PNG formats
- Detect document boundaries using edge detection and contour analysis
- Correct perspective distortion
- Enhance readability using adaptive thresholding
- Save the processed document as an image

## Technologies

- Python
- OpenCV
- NumPy

## Project Structure

```text
smart-document-scanner-v2/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/Shambhavi-code/smart-document-scanner-v2.git
   cd smart-document-scanner-v2
   ```

2. Create and activate a virtual environment.

   Windows:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the program:

```bash
python main.py
```

To scan an image, import the function in Python and call:

```python
from main import scan_document

scan_document("document.jpg", "scanned_document.jpg")
```

## Computer Vision Concepts

- Grayscale conversion
- Gaussian smoothing
- Canny edge detection
- Contour approximation
- Perspective transformation
- Adaptive thresholding

## Current Limitations

Document boundary detection may fail on images with poor lighting, low contrast, complex backgrounds, or overlapping objects.

## Project Status

Initial implementation. Testing and improvements are ongoing.
