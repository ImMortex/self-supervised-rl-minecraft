import cv2
import numpy as np

from src.agent.observation.trainers.simple_icon_classifier_utils import custom_preprocess_data_x, \
    cv_img_list_to_torch_tensor, predict


def model_predict(img, model, config, int_2_label):

    if config["channels"] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    preprocessed = custom_preprocess_data_x([img], config_dict=config)
    # cv2.imwrite("tmp/classify/" + str(time.time()) +".png", preprocessed[0])  # debug
    input_tensor = cv_img_list_to_torch_tensor(preprocessed)
    predicted_labels: [] = predict(int_2_label, model, [input_tensor])

    return predicted_labels[0], preprocessed[0]


def classify_minecraft_item_icon(img: np.ndarray, model, train_config, int_2_label) -> str:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predicted_label, preprocessed = model_predict(img, model, config=train_config, int_2_label=int_2_label)
    # debug log
    # if predicted_label == "oak_log":
    #    cv2.imwrite(os.path.join(debug_dir, "prediction_" + str(predicted_label) + ".png"), preprocessed * 255)

    return predicted_label
