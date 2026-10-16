# RecycleVision AI — Viva Examination & Technical Interview Guide

**Student Name**: Priya Sharma  
**Degree**: B.Tech, Computer Science and Engineering  
**Project**: RecycleVision AI: Smart Waste Classification & Recycling Analytics  

---

## 1. Project Overview & Fundamentals

### Q1: What is RecycleVision AI and what problem does it solve?
**Answer**:  
RecycleVision AI is an intelligent computer vision application designed to automate the classification of waste materials. Solid waste management is a major environmental crisis—municipal facilities often discard potentially recyclable items because manual sorting is slow, error-prone, hazardous to human workers, and expensive. RecycleVision AI classifies waste images into 6 standard categories (*cardboard, glass, metal, paper, plastic, trash*), presents confidence scores, and educates users on proper disposal rules to prevent contamination at the source.

### Q2: What is the tech stack used in your project?
**Answer**:  
- **Deep Learning / AI Core**: Python, TensorFlow 2.12.0, Keras, MobileNetV2 architecture, Pillow, and Scikit-learn.
- **Backend API Service**: FastAPI (Python) and Uvicorn ASGI server with Pydantic request/response validation.
- **Frontend User Interface**: Next.js (App Router), TypeScript, Tailwind CSS, Lucide icons, and Recharts.
- **Database & Authentication**: Supabase PostgreSQL (with Row-Level Security) with offline SQLite fallback.
- **Hosting Targets**: Vercel for frontend Next.js, and Render for the FastAPI backend container.

---

## 2. Machine Learning & Computer Vision

### Q3: What neural network architecture is used, and why did you choose it over training a CNN from scratch?
**Answer**:  
We used **MobileNetV2** with **Transfer Learning**.  
Training a deep Convolutional Neural Network (CNN) from scratch on small datasets (like TrashNet's 2,527 images) often leads to overfitting and requires millions of labeled samples. MobileNetV2 was pre-trained on ImageNet (1.4 million images across 1,000 classes), meaning its early layers already learned generalized visual features such as edges, textures, contours, and color transitions. We froze the base feature extractor and trained a custom classification head (`GlobalAveragePooling2D` -> `Dense(128, ReLU)` -> `Dense(6, Softmax)`), achieving 78.5%+ test accuracy with only 2.42 million parameters.

### Q4: What makes MobileNetV2 efficient for real-world deployment?
**Answer**:  
MobileNetV2 introduces two key innovations:
1. **Depthwise Separable Convolutions**: It splits standard convolution into a depthwise convolution (filtering per channel) followed by a pointwise $1 \times 1$ convolution (combining channels), reducing computation by 8 to 9 times.
2. **Inverted Residuals with Linear Bottlenecks**: High-dimensional representations are compressed into low-dimensional bottlenecks before expansion, drastically lowering RAM usage while preserving non-linear expressiveness.

### Q5: What image preprocessing steps are applied before feeding an image to the model?
**Answer**:  
1. **RGB Conversion**: Any uploaded image (including RGBA PNGs or grayscale) is converted to standard 3-channel RGB.
2. **Resizing**: Resized to $224 \times 224$ pixels using bilinear interpolation.
3. **Rescaling / Normalization**: Pixel values in the range $[0, 255]$ are divided by $255.0$ to scale into $[0.0, 1.0]$.
4. **Batch Dimension Expansion**: Transformed into shape $(1, 224, 224, 3)$.

### Q6: What activation function is used in the output layer and why?
**Answer**:  
We use the **Softmax** activation function because waste classification is a single-label, mutually exclusive multi-class problem (6 categories). Softmax converts raw logit outputs into a normalized probability distribution where each class score is between 0 and 1, and the sum of all 6 scores equals exactly 1.0:
$$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

### Q7: What loss function was used to train the model?
**Answer**:  
**Categorical Cross-Entropy** ($-\sum y_i \log(\hat{y}_i)$) was used because the targets were one-hot encoded vectors (shape $(N, 6)$).

---

## 3. Evaluation & Metrics

### Q8: What is the difference between Model Confidence and Test Accuracy?
**Answer**:  
- **Model Confidence**: The softmax score produced by the neural network for a single image prediction. A confidence of 96% simply means the model assigned high mathematical weight to that output neuron, but an uncalibrated network can still be confidently wrong.
- **Test Accuracy**: The empirical percentage of correct predictions made across an independent, unseen test dataset of 503 images ($78.53\%$).

### Q9: Why is Accuracy alone not enough to evaluate the model?
**Answer**:  
Because datasets often have class imbalance. For example, our dataset contains 594 paper images but only 137 trash images. A naive classifier could predict "paper" more frequently and achieve high accuracy while failing on rare classes. Therefore, we evaluate **Precision, Recall, F1-Score (Macro & Weighted)**, and inspect the **Confusion Matrix** to examine per-class misclassifications.

### Q10: What are Precision, Recall, and F1-Score?
**Answer**:  
- **Precision**: $\frac{TP}{TP + FP}$ — Out of all items the model predicted as Plastic, how many were actually Plastic? (Avoids false alarms).
- **Recall**: $\frac{TP}{TP + FN}$ — Out of all actual Plastic items in the test set, how many did the model identify? (Avoids missed detections).
- **F1-Score**: The harmonic mean of precision and recall:
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Q11: What is a Confusion Matrix?
**Answer**:  
A square grid comparing the ground truth labels against model predictions. Rows represent the actual classes, and columns represent the predicted classes. Diagonal cells represent correct predictions (True Positives), while off-diagonal cells highlight systematic confusion (e.g. glass items misclassified as plastic due to translucent reflections).

---

## 4. Full-Stack & System Design

### Q12: Why did you choose FastAPI over Flask or Django?
**Answer**:  
1. **Asynchronous Performance**: Built on Starlette and ASGI, FastAPI handles concurrent network I/O efficiently.
2. **Type Safety & Pydantic**: Automated request parsing, type conversion, and data validation.
3. **Automatic OpenAPI / Swagger Documentation**: Generates interactive documentation (`/docs`) directly from Python code schemas.

### Q13: What is CORS and why is it important in this project?
**Answer**:  
**Cross-Origin Resource Sharing (CORS)** is a browser security protocol that prevents a web application hosted on one origin (e.g., `http://localhost:3000` or `https://recyclevision-ai.vercel.app`) from querying an API on a different origin (e.g., `http://localhost:8000` or `https://recyclevision-backend.onrender.com`) unless the API explicitly permits it. In FastAPI, `CORSMiddleware` configures allowed origins and HTTP headers.

### Q14: How does the application prevent leaking internal server details or file paths?
**Answer**:  
We implemented a global exception handler in FastAPI that intercepts unhandled exceptions, logs the detailed stack trace to internal server logs, and returns a clean, sanitized HTTP 500 JSON error message to the client without exposing directory structures, environment variables, or database connection strings.

### Q15: How does database persistence work if Supabase credentials are missing?
**Answer**:  
We engineered an intelligent **dual-mode persistence service** (`DatabaseService`). If Supabase credentials are provided, it persists records to Supabase PostgreSQL over REST/WebSockets. If credentials are omitted or the connection is offline, it automatically falls back to an embedded local SQLite database (`backend/data/recyclevision.db`), ensuring that testing, history, filtering, CSV export, and analytics work out-of-the-box.

---

## 5. Limitations & Future Scope

### Q16: Can this model detect multiple waste items in a single photograph?
**Answer**:  
No, RecycleVision AI is currently an **Image Classification** model, which assigns a single label to the predominant object in the scene. To detect multiple waste items simultaneously, we would need an **Object Detection** model (such as YOLOv8 or Faster R-CNN) trained on bounding-box annotations (e.g., TACO dataset). This is documented as the top priority in our future scope.

### Q17: What are other known limitations of the system?
**Answer**:  
1. **Lighting and reflection**: Clear glass and translucent plastics can be confused under harsh lighting.
2. **Contamination & Deformation**: Crushed cans or soiled cardboard covered in grease exhibit higher prediction uncertainty.
3. **Regional Recycling Differences**: Recycling rules differ across municipal corporations; the application notes this clearly to prevent user misconceptions.
