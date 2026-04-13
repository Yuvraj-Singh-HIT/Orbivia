import segmentation_models_pytorch as smp

def get_model(num_classes):
    model = smp.Segformer(
        encoder_name="mit_b0",
        encoder_weights="imagenet",
        classes=num_classes,
        activation=None
    )
    return model
