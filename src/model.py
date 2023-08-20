from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from src import config


def autolabel_images(
    input_folder=config.IMAGE_DIR_PATH,
    ontology={"hand holding": "hand", "product held by": "product"},
    box_threshold=0.35,
    text_threshold=0.25,
    output_folder=config.DATASET_DIR_PATH,
    extension=".jpg",
):
    """
    Autolabel images in a folder.

    Args:
        input_folder (str, optional): Path to the folder with the images. Defaults to config.IMAGE_DIR_PATH.
        ontology (Dict[str, str], optional): Ontology of the captions. Defaults to {"hand holding": "hand", "product held by": "product"}.
        box_threshold (float, optional): Box threshold. Defaults to 0.35.
        text_threshold (float, optional): Text threshold. Defaults to 0.25.
        output_folder (str, optional): Path to the folder to save the labeled images. Defaults to config.DATASET_DIR_PATH.
        extension (str, optional): Extension of the images. Defaults to ".jpg".
    """
    # create the ontology
    ontology = CaptionOntology(ontology)

    base_model = GroundingDINO(
        ontology=ontology, box_threshold=box_threshold, text_threshold=text_threshold
    )

    # label all images in a folder called `context_images`
    base_model.label(
        input_folder=input_folder, extension=extension, output_folder=output_folder
    )
