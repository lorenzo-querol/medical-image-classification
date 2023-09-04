import os
import tensorflow as tf


def create_data_augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )


def create_metrics():
    return [
        tf.keras.metrics.Accuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.F1Score(threshold=0.5),
    ]


def create_model(config):
    base_model = tf.keras.applications.vgg16.VGG16(
        weights=None, include_top=False, input_shape=(224, 224, 3)
    )

    data_augmentation = create_data_augmentation()

    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            data_augmentation,
            tf.keras.layers.Lambda(tf.keras.applications.vgg16.preprocess_input),
            base_model,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dense(4096, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    metrics = create_metrics()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["hparams"]["learning_rate"]
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

    train_dataset = train_set.prefetch(buffer_size=tf.data.AUTOTUNE)
    valid_dataset = valid_set.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, valid_dataset


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
