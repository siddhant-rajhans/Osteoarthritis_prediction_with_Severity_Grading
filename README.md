## Osteoarthritis Classification

This project utilizes a pre-trained deep learning model to predict the severity of osteoarthritis in an X-ray image. It leverages Streamlit to create a user-friendly web application for image upload and prediction visualization.

### Getting Started

**Prerequisites:**

- Python 3.6 or later [3.10.4 used by me]
- PyTorch
- torchvision
- OpenCV (cv2)
- Streamlit

**Installation:**

1. **Create a virtual environment (recommended):**
    
    Bash
    
    ```
    python -m venv env
    source env/bin/activate  # Windows: env\Scripts\activate
    ```
    
    
2. **Install required libraries from requirements.txt:**
    
    Bash
    
    ```
    pip install -r requirements.txt
    ```
    
    This command will automatically install all the necessary libraries listed in the `requirements.txt` file within your project directory.

    
3. **Clone this repository or download the project files.**
    

**Running the App:**

1. Navigate to the project directory in your terminal.
    
2. Run the following command:
    
    Bash
    
    ```
    streamlit run streamlit_app.py
    ```
    

This will launch the Streamlit app in your web browser, typically at http://localhost:8501.

### Using the App

1. Upload an X-ray image of a knee joint.
2. The app will display the uploaded image and predict the osteoarthritis grade (Healthy, Moderate, Severe).
3. It will also show the probability for each class.

**Note:**

- The model performance might vary depending on the training data and chosen architecture.
- Ensure the uploaded image is a valid X-ray of a knee joint for optimal results.

### Project Structure

The project consists of the following files:

- `streamlit_app.py`: The main script containing the Streamlit app logic, model loading, prediction function, and user interface elements.
- `requirements.txt`: A file listing the required Python libraries for easy installation.

### Model Details

The model used in this project is a pre-trained ResNet-18 architecture from the PyTorch library. It's fine-tuned for classifying knee X-ray images into three categories:

- Healthy (Grade 0)
- Moderate (Grade 1)
- Severe (Grade 2)

**Note:**

- The model performance depends on the training data and chosen hyperparameters.

### Model (Replace with yours to experiment)

I saved then used a pre-trained model named `saved_model.pt` is present in the `model` directory. You'll need to replace it with your own trained model if you want to experiment.

**Make sure the model architecture and output format (3 classes) are compatible with the prediction function in `streamlit_app.py`.**

### Future Enhancements

- Implement error handling for invalid image uploads.
- Visualize the predicted probabilities using a bar chart or heatmap.
- Integrate with a medical imaging library for advanced image processing.