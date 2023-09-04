import tensorflow as tf
from config import config
import datetime
import os
from utils import prepare_datasets, create_model


def build_config_paths(config):
    filename = f"vgg16_{datetime.datetime.now()}_e{config['hparams']['epochs']}_lr{config['hparams']['learning_rate']}_bs{config['hparams']['batch_size']}"
    checkpoint_dir = f"checkpoints/{filename}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/cp-{{epoch:04d}}.ckpt"
    csv_log_path = f"runs/{filename}.csv"
    return checkpoint_path, csv_log_path


def create_callbacks(checkpoint_path, csv_log_path):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10
    )
    csv_logger = tf.keras.callbacks.CSVLogger(csv_log_path, append=False)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True, save_best_only=True, verbose=1
    )
    return [early_stopping, csv_logger, model_checkpoint]


def train_vgg16():
    train_dataset, valid_dataset = prepare_datasets(
        "Dataset/extracted/train", "Dataset/extracted/val", config
    )
    checkpoint_path, csv_log_path = build_config_paths(config)
    model = create_model(config)
    callbacks = create_callbacks(checkpoint_path, csv_log_path)

    model.fit(
        train_dataset,
        epochs=config["hparams"]["epochs"],
        validation_data=valid_dataset,
        callbacks=callbacks,
    )


train_vgg16()
