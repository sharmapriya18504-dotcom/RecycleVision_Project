import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def train():
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 15
    TRAIN_DIR = "dataset"

    if not os.path.exists(TRAIN_DIR):
        raise FileNotFoundError(f"Dataset folder '{TRAIN_DIR}' not found.")

    # Data Augmentation & Normalization
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_data = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    print(f"Detected {train_data.num_classes} classes: {list(train_data.class_indices.keys())}")

    # Transfer Learning Backbone: MobileNetV2 pre-trained on ImageNet
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base model weights
    for layer in base_model.layers:
        layer.trainable = False

    # Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = Dense(128, activation='relu', name="dense_features")(x)
    x = Dropout(0.2, name="dropout_regularization")(x)
    predictions = Dense(train_data.num_classes, activation='softmax', name="predictions")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("backend/models", exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            "models/best_model.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    print("Beginning training...")
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # Save final model
    model.save("models/best_model.h5")
    model.save("backend/models/best_model.h5")
    print("Training finished successfully. Models saved to models/best_model.h5 and backend/models/best_model.h5.")

if __name__ == "__main__":
    train()
