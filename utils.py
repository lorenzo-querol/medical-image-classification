import os
import tensorflow as tf
import cv2
import numpy as np


def create_data_augmentation():
    return tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
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

    model = tf.keras.Sequential()

    # Preprocessing: Data augmentation and normalization
    model.add(tf.keras.layers.InputLayer(input_shape=(224, 224, 3)))
    model.add(data_augmentation)
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255))

    # Feature extraction: base model
    model.add(base_model)

    # Fully connected layers: Dropout and Batch Norm
    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dropout(
            config["hparams"]["dropout_rate_1"], seed=config["seed"]
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(
        tf.keras.layers.Dropout(
            config["hparams"]["dropout_rate_2"], seed=config["seed"]
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    # Change to Sigmoid activation for binary classification
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    # Compile model
    metrics = create_metrics()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["hparams"]["learning_rate"]
        ),
        loss="binary_crossentropy",
        metrics=metrics,
    )

    return model


def prepare_datasets(path, config):
    train_set = tf.keras.utils.image_dataset_from_directory(
        directory=path,
        validation_split=0.3,
        subset="training",
        color_mode="rgb",
        seed=config["seed"],
        image_size=config["image_size"],
        batch_size=config["hparams"]["batch_size"],
        label_mode="binary",
    )

    valid_set = tf.keras.utils.image_dataset_from_directory(
        directory=path,
        validation_split=0.2 / 0.3,
        subset="validation",
        color_mode="rgb",
        seed=config["seed"],
        image_size=config["image_size"],
        batch_size=config["hparams"]["batch_size"],
        label_mode="binary",
    )

    test_set = tf.keras.utils.image_dataset_from_directory(
        directory=path,
        validation_split=0.1 / 0.3,
        subset="validation",
        color_mode="rgb",
        seed=config["seed"]
        + 1,  # Use a different seed to ensure no overlap with validation set
        image_size=config["image_size"],
        batch_size=config["hparams"]["batch_size"],
        label_mode="binary",
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_set = train_set.cache().prefetch(buffer_size=AUTOTUNE)
    valid_set = valid_set.cache().prefetch(buffer_size=AUTOTUNE)
    test_set = test_set.cache().prefetch(buffer_size=AUTOTUNE)

    return train_set, valid_set, test_set


def gradCam(m, image, true_label, layer_conv_name):
    model_grad = tf.keras.models.Model(
        inputs=m.input, outputs=[m.get_layer(layer_conv_name).output, m.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = model_grad(image)
        tape.watch(conv_output)
        loss = tf.losses.binary_crossentropy(true_label, predictions)

    grad = tape.gradient(loss, conv_output)
    grad = tf.keras.backend.mean(tf.abs(grad), axis=(0, 1, 2))
    conv_output = np.squeeze(conv_output.numpy())

    for i in range(conv_output.shape[-1]):
        conv_output[:, :, i] = conv_output[:, :, i] * grad[i]

    heatmap = tf.reduce_mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / tf.reduce_max(heatmap)
    heatmap = cv2.resize(heatmap.numpy(), (224, 224))

    return np.squeeze(heatmap), np.squeeze(image)


def getHeatMap(m, images, labels, num_samples):
    heatmaps = []

    for index in range(num_samples):
        heatmap, _ = gradCam(
            m, images[index : index + 1], labels[index : index + 1], "relu"
        )
        heatmaps.append(heatmap)

    return np.array(heatmaps)
