def icon_filename_to_display_name(class_label: str):
    """
    Extracts the display name of an item from its icon filename
    """
    if ".png" in class_label:
        class_label: str = class_label.replace(".png", "")
        if class_label.count("_") >= 1:
            splits = class_label.split("_", 1)
            label = item_display_name_to_class_label(splits[1])
            return label

    return item_display_name_to_class_label(class_label)


def item_display_name_to_class_label(item_display_name: str):
    """
    Normalizes a item name
    """
    return item_display_name.lower()
