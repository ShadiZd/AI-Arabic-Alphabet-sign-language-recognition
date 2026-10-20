# 🤟 AI Arabic Alphabet Sign Language Recognition

An AI-powered application that recognizes Arabic alphabet signs using computer vision and machine learning. The project provides tools for collecting sign samples, building a recognition model, and recognizing signs through a Python application.

## ✨ Features

- **Arabic Sign Recognition:** Recognize Arabic alphabet signs using a trained machine learning model.
- **Sign Data Collection:** Collect and save samples to build or expand your own dataset.
- **Model Training:** Train a recognition model using collected sign samples.
- **Computer Vision:** Use a camera to capture and process signs.
- **Speech Support:** Support for text-to-speech functionality where implemented.
- **Customizable Dataset:** Collect your own samples to adapt the system to your needs.

## 🛠️ Technologies Used

- Python
- Computer Vision
- Machine Learning
- NumPy and other dependencies listed in `req.txt`

## 📁 Project Structure

| File or folder | Description |
|---|---|
| `main.py` | Main application entry point |
| `simple_signrec.py` | Sign recognition implementation |
| `collect_signs.py` | Collects sign samples |
| `build_model.py` | Builds or trains the recognition model |
| `signs_model.joblib` | Saved trained model |
| `signs.jpg` | Project image or visual resource |
| `req.txt` | Python dependency list |
| `encodings/` | Directory for collected or generated sign data |
| `tts_cache/` | Directory for generated text-to-speech cache files |

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ShadiZd/AI-Arabic-Alphabet-sign-language-recognition.git
cd AI-Arabic-Alphabet-sign-language-recognition
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r req.txt
```

### 4. Run the application

```bash
python main.py
```

Make sure your camera is connected and that the application has permission to access it.

## 🧠 How It Works

The project follows a machine learning workflow:

1. **Collect data:** Capture samples of Arabic alphabet signs.
2. **Build the model:** Process the collected samples and train the recognition model.
3. **Recognize signs:** Use the trained model to predict signs from new input.
4. **Present the result:** Display the recognized sign and provide speech output where supported.

## 📊 Model and Dataset

The repository includes a saved model file, `signs_model.joblib`. You can also use the project's data collection and model-building scripts to develop your own dataset and model.

Recognition performance depends on the quality and variety of training samples, camera conditions, lighting, and how consistently signs are performed.

## 🔧 Customization

To adapt the project for your own use:

- Collect additional examples of the signs you want to recognize.
- Retrain the model when your dataset changes.
- Test recognition under different lighting and camera conditions.
- Keep generated data and cache files local to your machine.

The `encodings/` and `tts_cache/` directories are preserved in the repository using `.gitkeep` files. Generated files inside these directories are excluded from Git.

## 🔮 Future Improvements

- Expand the dataset with more sign variations.
- Improve recognition accuracy and robustness.
- Add real-time recognition performance metrics.
- Develop a user-friendly graphical interface.
- Support additional signs and sign-language applications.
- Evaluate the model using a separate test dataset.

## 👨‍💻 Author

**Shadi Zyoud**

AI and Robotics student interested in artificial intelligence, machine learning, and computer vision.

- GitHub: [@ShadiZd](https://github.com/ShadiZd)

## 🤝 Contributions

Suggestions, bug reports, and contributions are welcome. Feel free to fork the repository, improve the implementation, and submit a pull request.

## 📄 License

A license has not yet been specified. Please add a license file if you intend to define how others may use, modify, and distribute this project.
