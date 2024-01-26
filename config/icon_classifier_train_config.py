import socket

hostname = socket.gethostname()
ICON_CLASSIFIER_TRAIN_CONFIG: dict = {
    "model":  "resnet18",
    "init_lr": 0.001,
    "epochs": 200,
    "batch_size": 32,
    "width": 224,
    "height": 224,
    "channels": 3,
    "black_out_item_amount": True,
    "resize_interpolation": "INTER_AREA",
    "host_name": str(hostname)
}