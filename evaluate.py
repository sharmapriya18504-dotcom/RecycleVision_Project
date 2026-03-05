import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

IMG_SIZE = 224
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    "dataset",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

model = tf.keras.models.load_model("models/best_model.h5")

predictions = model.predict(test_data)

y_pred = np.argmax(predictions, axis=1)

print(classification_report(test_data.classes, y_pred))
print(confusion_matrix(test_data.classes, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(test_data.classes, y_pred)

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()