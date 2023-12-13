import json
import random
import os

def create_dataset_json(description, training_images, validation_images, test_images, output_file_path):
    dataset_structure = {
        "description": "swinUNetr TB ds256",
        "labels": {
            "0": "background",
            "1": "bone",
            "2": "malleus",
            "3": "incus",
            "4": "stapes",
            "5": "bony_labyr",
            "6": "IAC",
            "7": "sup_vestib_n",
            "8": "inf_vestib_n",
            "9": "cochlear_n",
            "10": "facial_n",
            "11": "chorda_tymp_n",
            "12": "ICA",
            "13": "sinus_dura",
            "14": "vestib_aqueduct",
            "15": "mandible",
            "16": "EAC"
        },
        "licence": "yt",
        "modality": {
            "0": "CT"
        },
        "name": "tb_hop",
        "numTest": 20,
        "numTraining": 80,
        "reference": "Hopkins",
        "release": "1.0 12/10/23",
        "tensorImageSize": "3D",
        "test": [{"image": f"images/{img}", "label": f"labels/Segmentation_{img}"} for img in test_images],
        "training": [{"image": f"images/{img}", "label": f"labels/Segmentation_{img}"} for img in training_images],
        "validation": [{"image": f"images/{img}", "label": f"labelsTr/Segmentation_{img}"} for img in validation_images]
    }
    with open(output_file_path, 'w') as file:
        json.dump(dataset_structure, file, indent=4)

def get_filenames(directory):
    filenames = [] 
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            filenames.append(filename)
    return filenames

def create_folds(image_names, num_folds):
    random.shuffle(image_names)
    test_images = image_names[:2]
    remaining_images = image_names[2:]
    folds = [remaining_images[i::num_folds] for i in range(num_folds)]
    return test_images, folds

input_directory = '../data/images'
image_names = get_filenames(input_directory)

# Create 5 folds
test_images, folds = create_folds(image_names, 5)

# Create 5 JSON files, each with a different validation fold
random.seed(87992)
for i, validation_images in enumerate(folds):
    training_images = [img for fold in folds for img in fold if fold != validation_images]
    output_file_path = f'../data/dataset_fold_{i+1}.json'
    create_dataset_json("swinUNetr TB ds256", training_images, validation_images, test_images, output_file_path)
