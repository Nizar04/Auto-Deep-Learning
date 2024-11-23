# AutoDL - Automated Deep Learning Platform

An interactive web application built with Streamlit that automates the deep learning model creation process. Simply upload your dataset, and the platform automatically handles preprocessing, model selection, training, and evaluation.

## Features

- Drag and drop dataset upload
- Automatic data preprocessing
- Automated model architecture selection
- Hyperparameter optimization
- Interactive training progress visualization
- Model performance evaluation
- Export trained models
- Support for:
  - Classification tasks
  - Regression tasks
  - Image classification
  - Time series prediction

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autodl.git
cd autodl

# Install requirements
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## Requirements

```
streamlit>=1.24.0
tensorflow>=2.9.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
plotly>=5.13.0
keras-tuner>=1.3.0
pillow>=9.4.0
```

## Usage

1. Launch the application
2. Upload your dataset (CSV or Excel for tabular data, folder of images for image classification)
3. Select the task type (classification/regression)
4. Configure basic settings (optional)
5. Start automated training
6. Download the trained model and predictions

## Supported Data Formats

- CSV files
- Excel files (.xlsx, .xls)
- Images (PNG, JPG, JPEG)
- JSON
- Parquet

## License

MIT License
