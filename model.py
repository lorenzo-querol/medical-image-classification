import tensorflow as tf
from config import config
import datetime
import os
from utils import prepare_datasets, create_model


def build_config_paths(config, model_name):
    filename = f"{model_name}_{datetime.datetime.now()}_e{config['hparams']['epochs']}_lr{config['hparams']['learning_rate']}_bs{config['hparams']['batch_size']}"
    checkpoint_dir = f"checkpoints/{filename}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/cp-{{epoch:04d}}.ckpt"
    csv_log_path = f"runs/{filename}.csv"
    return checkpoint_path, csv_log_path


def create_callbacks(checkpoint_path, csv_log_path):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=8
    )
    csv_logger = tf.keras.callbacks.CSVLogger(csv_log_path, append=False)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        patience=6, monitor="val_loss", mode="min"
    )

    return [early_stopping, csv_logger, model_checkpoint, lr_scheduler]


def train_vgg16():
    train_dataset, valid_dataset = prepare_datasets(
        "Dataset/cropped/train", "Dataset/cropped/val", config
    )

    model_name = "vgg16"
    checkpoint_path, csv_log_path = build_config_paths(config, model_name)

    base_model = tf.keras.applications.vgg16.VGG16(
        weights=None,
        include_top=False,
        input_shape=(224, 224, 3),
    )

    model = create_model(base_model, config)
    callbacks = create_callbacks(checkpoint_path, csv_log_path)

    model.fit(
        train_dataset,
        epochs=config["hparams"]["epochs"],
        validation_data=valid_dataset,
        callbacks=callbacks,
    )


train_vgg16()
