import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="RecycleVision Dashboard", layout="wide")

# Load model
model = tf.keras.models.load_model("models/best_model.h5")

classes = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# Sidebar
st.sidebar.title("♻️ RecycleVision")

page = st.sidebar.radio(
    "Navigate",
    ["Home","Project Introduction","Objective","Garbage Classification","About Me","Conclusion"]
)

# ---------------- HOME ---------------- #

if page == "Home":

    st.title("♻️ RecycleVision - Garbage Image Classification")

    st.write(
    """
    Welcome to **RecycleVision Dashboard**.

    This project is based on **Artificial Intelligence and Deep Learning**.

    The system can automatically identify different types of garbage using images.

    Instead of manually sorting waste, our AI model helps detect the category of waste.

    This project can help improve **recycling systems and waste management**.
    """
    )

    st.subheader("Why this project is important")

    st.write(
    """
    • Waste management is a major environmental problem.  
    • Many recyclable materials are mixed with general waste.  
    • Manual waste sorting is slow and inefficient.  

    Our AI model helps **automatically classify waste images**.
    """
    )

# ---------------- INTRODUCTION ---------------- #

elif page == "Project Introduction":

    st.title("📘 Project Introduction")

    st.image(
        "https://cdn-icons-png.flaticon.com/512/2920/2920349.png",
        width=200
    )

    st.write(
    """
    Waste segregation is an important part of modern recycling systems.

    If waste is properly separated, it becomes easier to recycle materials
    like plastic, glass, metal, and paper.

    In this project we use **Deep Learning and Computer Vision**
    to classify garbage images into different categories.

    A trained AI model looks at an image and predicts the type of waste.

    This system can be used in:

    • Smart recycling bins  
    • Waste management systems  
    • Environmental monitoring
    """
    )

# ---------------- OBJECTIVE ---------------- #

elif page == "Objective":

    st.title("🎯 Project Objective")

    st.write(
    """
    The main goals of this project are:

    1️⃣ Build a deep learning model to classify garbage images.

    2️⃣ Automatically detect waste types such as:

    • Plastic  
    • Glass  
    • Metal  
    • Paper  
    • Cardboard  
    • Trash  

    3️⃣ Use **transfer learning (MobileNetV2)** to improve accuracy.

    4️⃣ Evaluate the model using:

    • Accuracy  
    • Precision  
    • Recall  
    • F1 Score  
    • Confusion Matrix

    5️⃣ Create a **user-friendly Streamlit dashboard**
    where users can upload images and see predictions.
    """
    )

# ---------------- CLASSIFICATION ---------------- #

elif page == "Garbage Classification":

    st.title("🧠 Garbage Image Classification")

    st.image(
        "https://cdn-icons-png.flaticon.com/512/728/728093.png",
        width=150
    )

    st.write(
    """
    Upload a garbage image and the AI model will identify the waste category.

    The model will also show a **confidence score** for the prediction.
    """
    )

    file = st.file_uploader("Upload Garbage Image", type=["jpg","png","jpeg"])

    if file is not None:

        img = Image.open(file)

        st.image(img, caption="Uploaded Image", use_column_width=True)

        img = img.resize((224,224))
        img = np.array(img)/255.0
        img = img.reshape(1,224,224,3)

        prediction = model.predict(img)

        result = classes[np.argmax(prediction)]
        confidence = np.max(prediction)*100

        st.success("Prediction: " + result)
        st.write("Confidence:", round(confidence,2), "%")

# ---------------- ABOUT ME ---------------- #

elif page == "About Me":

    st.title("👩‍💻 About Me")

    st.write(
    """
    Hello, my name is **Priya Sharma**.

    I am a student interested in **Artificial Intelligence,
    Machine Learning, and Data Science**.

    This project demonstrates how AI can be used to solve
    real-world environmental problems.

    Through this project I learned:

    • Image preprocessing  
    • Deep learning using CNN  
    • Transfer learning  
    • Model evaluation using confusion matrix  
    • Building dashboards using Streamlit  

    I am interested in learning more about
    **AI applications in real-world problems**.
    """
    )

# ---------------- CONCLUSION ---------------- #

elif page == "Conclusion":

    st.title("📊 Conclusion")

    st.write(
    """
    This project shows how **Artificial Intelligence can help in waste management**.

    The trained deep learning model can classify garbage images
    into different categories.

    By integrating the model with a **Streamlit dashboard**, users
    can easily upload images and get predictions.

    In the future, this system can be used in:

    • Smart recycling bins  
    • Automated waste sorting systems  
    • Municipal waste management  

    This project demonstrates the potential of
    **AI for environmental sustainability**.
    """
    )