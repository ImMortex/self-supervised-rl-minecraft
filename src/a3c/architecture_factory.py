from src.a3c.a3c_dense import A3CDense
from src.a3c.a3c_vit_gru import A3CMcRlNet
from src.a3c.a3c_with_vision_encoder import A3CWithVisionEncoder
from src.a3c.a3c_vit import A3CViT
from src.a3c.a3c_vit_avg_pooling import A3CViTAvgPooling


def get_net_architecture(net_architecture: str, ssl_head_args, train_config: dict, device):
    if net_architecture.lower() == "a3c_dense":
        return A3CDense(ssl_head_args, train_config=train_config, device=device)  # only for POC

    elif net_architecture.lower() == "a3c_vit":
        return A3CViT(ssl_head_args, train_config=train_config, device=device)  # only for POC

    elif net_architecture.lower() == "a3c_vit_avg_pooling":
        return A3CViTAvgPooling(ssl_head_args, train_config=train_config, device=device)  # only for POC

    elif net_architecture.lower() == "a3c_vit_gru":
        return A3CMcRlNet(ssl_head_args, train_config=train_config, device=device)  # only for POC

    elif net_architecture.lower() == "resnet18":
        return A3CWithVisionEncoder(train_config=train_config, device=device, model_name="resnet18")

    elif net_architecture.lower() == "resnet50":
        return A3CWithVisionEncoder(train_config=train_config, device=device, model_name="resnet50")

    elif net_architecture.lower() == "naturecnn":
        return A3CWithVisionEncoder(train_config=train_config, device=device, model_name="naturecnn")



    return None
