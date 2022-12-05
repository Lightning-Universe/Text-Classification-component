import lightning as L


def default_callbacks():
    early_stopping = L.pytorch.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        verbose=True,
        mode="min",
    )
    checkpoints = L.pytorch.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )
    return [early_stopping, checkpoints]
