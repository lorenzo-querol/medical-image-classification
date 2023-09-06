import os
import tensorflow as tf


def create_data_augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
            tf.keras.layers.experimental.preprocessing.RandomContrast(factor=0.1),
            tf.keras.layers.RandomBrightness(factor=0.1),
            tf.keras.layers.experimental.preprocessing.RandomTranslation(
                height_factor=0.1, width_factor=0.1
            ),
        ]
    )


def create_metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.F1Score(threshold=0.5),
    ]


def create_model(base_model, config):
    data_augmentation = create_data_augmentation()

    # Preprocess and augment the data
    input_tensor = tf.keras.layers.Input(shape=(224, 224, 3))
    x = data_augmentation(input_tensor)
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)

    # # Pass through the feature extractor
    x = base_model(x)

    # Fully connected layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.8, seed=config["seed"])(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(
        x
    )  # Replace with sigmoid instead of softmax since binary classification

    # Compile the model
    model = tf.keras.models.Model(inputs=input_tensor, outputs=x)

    metrics = create_metrics()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            ema_momentum=0.9, learning_rate=config["hparams"]["learning_rate"]
        ),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics,
    )

    return model


def prepare_datasets(train_path, valid_path, config):
    train_set = tf.keras.utils.image_dataset_from_directory(
        train_path,
        seed=config["seed"],
        image_size=config["image_size"],
        batch_size=config["hparams"]["batch_size"],
        label_mode="binary",
    )

    valid_set = tf.keras.utils.image_dataset_from_directory(
        valid_path,
        seed=config["seed"],
        image_size=config["image_size"],
        batch_size=config["hparams"]["batch_size"],
        label_mode="binary",
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
    valid_set = valid_set.cache().prefetch(buffer_size=AUTOTUNE)

    return train_set, valid_set


def get_class_distribution(split):
    dataset_folder = f"Dataset/extracted/{split}"
    class_folders = os.listdir(dataset_folder)
    class_distribution = {}

    for class_folder in class_folders:
        if os.path.isdir(os.path.join(dataset_folder, class_folder)):
            num_samples = len(os.listdir(os.path.join(dataset_folder, class_folder)))
            class_distribution[class_folder] = num_samples

    print(f"Class Distribution of {split}")
    for class_label, num_samples in class_distribution.items():
        print(f"{class_label}: {num_samples} samples")
