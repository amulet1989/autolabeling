import os


def remove_empty_labels(image_dir, label_dir, remove_multiple=False):
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        lab_filename = label_file.replace(".txt", ".jpg")

        # Check if the label file is empty or if remove_multiple is True and it contains multiple BBoxes
        if os.stat(label_path).st_size == 0 or (
            remove_multiple and contains_multiple_bboxes(label_path)
        ):
            os.remove(label_path)
            os.remove(os.path.join(image_dir, lab_filename))


def contains_multiple_bboxes(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()
        num_bboxes = len(lines)
        return num_bboxes > 1


def remove_large_bboxes(label_dir, max_size):
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, "r") as f:
            annotations = f.readlines()
        updated_annotations = []
        for annotation in annotations:
            class_id, x, y, w, h = map(float, annotation.strip().split())
            if w < max_size and h < max_size:
                updated_annotations.append(annotation)
        with open(label_path, "w") as f:
            f.writelines(updated_annotations)


def remove_overlapping_bboxes(label_dir, iou_threshold=0.2):
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, "r") as f:
            annotations = f.readlines()
        updated_annotations = []
        for i, annotation in enumerate(annotations):
            bbox1 = list(map(float, annotation.strip().split()[1:]))
            skip = False
            for j in range(i + 1, len(annotations)):
                bbox2 = list(map(float, annotations[j].strip().split()[1:]))
                if iou(bbox1, bbox2) > iou_threshold:
                    skip = True
                    break
            if not skip:
                updated_annotations.append(annotation)
        with open(label_path, "w") as f:
            f.writelines(updated_annotations)


def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = x_overlap * y_overlap
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union


def run_processing_dataset(
    image_dir: str,
    label_dir: str,
    max_size: float = 0.2,
    iou_threshold: float = 0.1,
    remove_empty: bool = True,
    remove_large: bool = True,
    remove_overlapping: bool = True,
    remove_multiple: bool = True,
) -> None:
    """
    Process a dataset of images and labels.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing the images.
    label_dir : str
        Path to the directory containing the labels.
    max_size : float
        Maximum size of a bounding box.
    iou_threshold : float
        Minimum IoU between two bounding boxes.
    remove_empty : bool
        If True, remove images with no labels.
    remove_large : bool
        If True, remove images with bounding boxes larger than max_size.
    remove_overlapping : bool
        If True, remove bounding boxes that overlap with each other.
    remove_multiple : bool
        If True, remove images with multiple bounding boxes.

    Returns
    -------
    None.

    Notes
    -----
    This function is used to process the Merged_Dataset/train and Merged_Dataset/valid
    datasets. It removes empty labels, images with bounding boxes larger than
    max_size, images with bounding boxes that overlap with each other, and
    images with no labels.

    """
    if remove_large and max_size is not None:
        remove_large_bboxes(label_dir, max_size)
    if remove_overlapping:
        remove_overlapping_bboxes(label_dir, iou_threshold)
    if remove_empty:
        remove_empty_labels(image_dir, label_dir, remove_multiple=remove_multiple)
