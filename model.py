import tensorflow as tf
from configs import config
import datetime
import os
from utils import prepare_datasets, create_model


def train():
    train_dataset, valid_dataset = prepare_datasets(
        "Dataset/extracted/train", "Dataset/extracted/val", config
    )

    model = create_model(config)

    model_name = "vgg16"
    filename = "{}_{}_e{}_lr{}_bs{}_dr{}".format(
        datetime.datetime.now(),
        model_name,
        config["hparams"]["epochs"],
        config["hparams"]["learning_rate"],
        config["hparams"]["batch_size"],
        config["hparams"]["dropout_rate"],
    )

    if not os.path.exists(f"checkpoints/{filename}"):
        os.makedirs(f"checkpoints/{filename}", exist_ok=True)

    checkpoint_path = f"checkpoints/{filename}" + "/cp-{epoch:04d}.ckpt"

    model.fit(
        train_dataset,
        epochs=config["hparams"]["epochs"],
        validation_data=valid_dataset,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=10,
            ),
            tf.keras.callbacks.CSVLogger(f"runs/{filename}.csv", append=False),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                save_weights_only=True,
                save_best_only=True,
                verbose=1,
            ),
        ],
    )


train()
