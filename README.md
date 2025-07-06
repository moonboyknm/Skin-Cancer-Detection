# Skin Cancer Detection

A deep learning-based web application for skin cancer classification, developed during an internship at NIT Rourkela. This project uses computer vision and machine learning techniques to assist in the early detection and classification of skin lesions.

## 🎯 Project Overview

This application leverages deep learning models to analyze skin lesion images and classify them into different categories, potentially helping in early skin cancer detection. The project features a user-friendly web interface built with Streamlit.

## ✨ Features

- **Image Classification**: Upload skin lesion images for automated analysis
- **Deep Learning Model**: Uses trained convolutional neural networks for accurate classification
- **Web Interface**: Interactive Streamlit-based frontend for easy use
- **Real-time Results**: Instant classification results with confidence scores

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow/Keras, PyTorch
- **Frontend**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Data Analysis**: NumPy, Pandas
- **Model Training**: Jupyter Notebooks (Kaggle)

## 📋 Prerequisites

Before running this application, ensure you have the following installed:

- Python 3.7 or higher
- pip (Python package installer)
- Git

## 🚀 Installation & Setup

### Step 1: Download the Trained Model
1. Visit the [Kaggle notebook](https://www.kaggle.com/code/kratikmudgal/skin-cancer-classification)
2. Download the trained model file
3. Save it for use in Step 3

### Step 2: Clone the Repository
```bash
git clone https://github.com/kratikmudgal/Skin-Cancer-Detection.git
cd Skin-Cancer-Detection
```

### Step 3: Set Up the Model
1. Place the downloaded model file directly in the `Frontend` directory:
   ```bash
   # Your model file should be in:
   # Skin-Cancer-Detection/Frontend/[your_model_file]
   ```

### Step 4: Install Dependencies
Navigate to the Frontend directory and install the required packages:
```bash
cd Frontend
pip install -r requirements.txt
```

### Step 5: Run the Application
From the Frontend directory, run:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
Skin-Cancer-Detection/
├── Backend/              # Backend implementation
├── Frontend/             # Frontend application
│   ├── app.py           # Main Streamlit application
│   ├── requirements.txt # Python dependencies
│   └── [model_file]     # Downloaded trained model
├── README.md            # Project documentation
└── LICENSE              # License file
```

## 🔬 Model Information

The deep learning model was trained on a comprehensive dataset of skin lesion images and can classify multiple types of skin conditions. For detailed information about the model architecture, training process, and performance metrics, please refer to the [Kaggle notebook](https://www.kaggle.com/code/kratikmudgal/skin-cancer-classification).

## 💡 Usage

1. Launch the application using the installation steps above
2. Upload a skin lesion image using the file uploader
3. Wait for the model to process the image
4. View the classification results and confidence scores
5. Consult with medical professionals for proper diagnosis

## ⚠️ Important Disclaimer

**This application is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult with qualified healthcare professionals for proper medical advice.**

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📧 Contact

For questions or collaboration opportunities, please reach out:

- **Developer**: Kratik Mudgal
- **Email**: mudgal.kratik@gmail.com
- **LinkedIn**: [www.linkedin.com/in/kratikmudgal](https://www.linkedin.com/in/kratikmudgal)
- **Kaggle**: [kratikmudgal](https://www.kaggle.com/kratikmudgal)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NIT Rourkela for providing the internship opportunity
- The open-source community for the tools and libraries used
- Contributors to the skin lesion datasets used for training

## 📊 Model Performance

For detailed performance metrics, validation results, and model comparison, please refer to the [Kaggle notebook](https://www.kaggle.com/code/kratikmudgal/skin-cancer-classification).

---

*Developed with ❤️ during internship at NIT Rourkela*
