from config.icon_classifier_train_config import ICON_CLASSIFIER_TRAIN_CONFIG
from src.agent.observation.trainers.simple_icon_classifier import IconClassifierTrainer

if __name__ == '__main__':
    """
    Train classifier that is used in src.agent.observation
    """
    config_dict: dict = ICON_CLASSIFIER_TRAIN_CONFIG

    config_dict["model"] = "resnet18"
    config_dict["channels"] = 3  # RGB or Grayscale
    trainer2 = IconClassifierTrainer(config_dict)
    trainer2.training(3)  # repeat training n times
