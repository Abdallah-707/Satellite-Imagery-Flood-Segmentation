# 🌊 Flood Area Segmentation using U-Net with ResNet34 Backbone

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Frameworks](https://img.shields.io/badge/framework-PyTorch%20%7C%20Flask-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This project performs **semantic segmentation of flooded areas** using **multi-spectral satellite imagery** with 12 input channels. The model is built using a **U-Net architecture** with a **ResNet34 encoder**, trained on TIFF images, and deployed using a **Flask web application** for interactive inference.

![Prediction Demo](https://i.ibb.co/Vp3Ms8vx/image.png)
---

## 🛰️ Input Data: 12 Satellite Channels

The model processes TIFF images with **12 specialized channels**, combining spectral bands, elevation, land cover, and water occurrence. Here's a breakdown:

| Index | Channel Name                | Wavelength | Type                  | Resolution |
|-------|-----------------------------|------------|------------------------|------------|
| 0     | Coastal aerosol             | 443nm      | Sentinel-2            | 60m        |
| 1     | Blue                        | 490nm      | Sentinel-2 / Landsat  | 30m        |
| 2     | Green                       | 560nm      | Sentinel-2 / Landsat  | 30m        |
| 3     | Red                         | 665nm      | Sentinel-2 / Landsat  | 30m        |
| 4     | NIR                         | 842nm      | Sentinel-2 / Landsat  | 30m        |
| 5     | SWIR1                       | 1610nm     | Sentinel-2 / Landsat  | 30m        |
| 6     | SWIR2                       | 2190nm     | Sentinel-2 / Landsat  | 30m        |
| 7     | QA Band                     | —          | Quality mask          | 30m        |
| 8     | MeritDEM                    | —          | Elevation             | 30m        |
| 9     | CopernicusDEM               | —          | Elevation             | 30m        |
| 10    | ESA World Cover Map         | —          | Land cover            | 30m        |
| 11    | Water Occurrence Probability| —          | Derived (JRC)         | 30m        |

📌 *You may include a visual sample here:*
> ![12 Channels Example](https://i.ibb.co/6RQzrcqy/image.png)

---

## 🧠 Model Architecture

- **U-Net** segmentation model from `segmentation_models_pytorch`
- **Encoder**: Pretrained ResNet34
- **Input**: 12 channels (converted to 3 via 1x1 conv layer for encoder compatibility)
- **Output**: Binary segmentation (flooded vs non-flooded)

### Loss Functions

- Combined **CrossEntropyLoss** and **Dice Loss**
- Evaluation metrics: **IoU (Intersection over Union)** per class

---

## 🏗️ Project Structure

```bash
.
├── model.py                 # Model definition, training, evaluation
├── app.py                   # Flask app for web inference
├── templates/
│   └── index.html           # HTML UI for uploading and viewing predictions
├── best_flood_model.pth     # Trained model weights
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🚀 Training

To train the model:

1. Organize your data:
    - `data/images/*.tif` (12-channel images)
    - `data/labels/*.png` (binary masks)
2. Set your paths in `main()` of `model.py`
3. Run:

```bash
python model.py
```

Training will:
- Split data into train/val sets
- Save best model (`best_flood_model.pth`)
- Plot training curves
- Visualize predictions

---

## 🌐 Web Deployment (Flask)

You can run a simple web app to test predictions on `.tif` files.

### Start the app:

```bash
python app.py
```

Visit `http://localhost:5000` and upload a TIFF image to see the predicted flood mask.

### Features:
- Upload `.tif` file (12-channel)
- Live prediction using trained model
- View mask overlay (white = flooded)

---

## 🧪 Evaluation & Visualization

- `IoU` per class (Flooded / Background)
- Visualization plots:
  - RGB composite
  - Ground truth mask
  - Prediction mask
  - Difference heatmap
- Training loss and IoU curves

> 📊 Example:
> 
> ![Ground Truth](https://i.ibb.co/zYc4vxR/68747470733a2f2f692e6962622e636f2f574e44685a3559342f696d6167652e706e67-2.png)
> ![Predection](https://i.ibb.co/rfFDPp1D/68747470733a2f2f692e6962622e636f2f396b734b575176432f696d6167652e706e67.png)

---

## 📦 Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

### Requirements include:
- torch, torchvision
- segmentation-models-pytorch
- rasterio
- pillow
- matplotlib, seaborn
- tqdm
- flask
- scikit-learn

---

## 🛠 Customization

- Use different encoders via `smp.Unet`
- Adjust number of classes for multi-class segmentation
- Extend normalization logic per new data type

---

## 📄 License

MIT License. Free to use, modify, and deploy.

---

## 🙋‍♂️ Author

Developed by [Your Name] — feel free to contribute or fork!

---

## 📬 Feedback

Have suggestions, issues, or want to collaborate? Open an issue or submit a PR.
