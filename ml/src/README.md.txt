🛡️ Insurance Fraud Detection System

An end-to-end **Machine Learning web application** that predicts whether an insurance claim is **Fraudulent** or **Genuine**, built using **Streamlit** and deployed on **Hugging Face Spaces**.

🚀 Live Demo

Hugging Face App: https://huggingface.co/spaces/Addiyo16/insurance-fraud-detection-streamlit

📌 Project Overview

Insurance fraud causes significant financial losses each year.  
This project uses **machine learning classification models** to analyze insurance claim details and provide real-time fraud predictions through a user-friendly web interface.

The application is designed as a **production-ready ML system**, covering:
- Model training
- Inference pipeline
- User interface
- Cloud deployment

🧠 Key Features

- Real-time insurance fraud prediction
- Interactive **Streamlit-based UI**
- Multiple ML models evaluated (Random Forest, XGBoost, Logistic Regression)
- Consistent preprocessing using ML pipelines
- Cloud deployment on **Hugging Face Spaces**
- GitHub-integrated automatic redeployment


🏗️ System Architecture

The application follows a **single-service ML architecture**, integrating UI, inference logic, and model loading.

📐 **Detailed architecture documentation:**  
➡️ [`docs/architecture.md`](docs/architecture.md)

🛠️ Technology Stack

- **Frontend:** Streamlit  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Data Processing:** Pandas, NumPy  
- **Deployment:** Hugging Face Spaces  
- **Version Control:** GitHub  


## ⚙️ How It Works

1. User enters insurance claim details in the web UI  
2. Inputs are validated and preprocessed  
3. Trained ML model performs fraud classification  
4. Prediction result (Fraudulent / Genuine) is displayed instantly  


🧪 Model Development

- Trained and evaluated multiple classification models
- Selected the best-performing model based on evaluation metrics
- Serialized the final model for production inference
- Ensured feature consistency between training and deployment
