"""Generate Train and Validation datasets from a dataset directory
that contains images stored in class folders.

The input data structure is:

<input_root_dir>
    <class A>
        image_A_1
        image_A_2
        . . .
    <class B>
        image_B_1
        image_B_2
        . . .
    , , ,

The output data structure starts with <output_root_dir>, which contains
folders called "train", "validation", and "test" (subsets). Inside these folders,
we have class directories like <class A>, <class B>, etc.

The class directories contain images assigned to each class and each subset.
"""

import os
import shutil
import random
import argparse

import constants as con

# Define constants for this script
TRN_RATIO = 75
VAL_RATIO = 25


class DirectoryNotFound(Exception):
    """Raise and exception when a directory does not exist."""
    pass

class InputData:
    def __init__(self, root_path):
        self.__root = root_path

        folders = os.listdir(self.__root)
        self.__folders = [f for f in folders if os.path.isdir(os.path.join(self.__root, f))]

    @property
    def root(self):
        return self.__root

    @property
    def classes(self):
        return self.__folders

    def get_class_path(self, class_name: str) -> str:
        if class_name not in self.classes:
            raise ValueError(f"Undefined class name: {class_name}")
        return os.path.join(self.__root, class_name)

    def get_class_image_paths(self, class_name: str) -> list[str]:
        if class_name not in self.classes:
            raise ValueError(f"Undefined class name: {class_name}")

        class_path = self.get_class_path(class_name)
        items = os.listdir(class_path)
        files = [item for item in items if os.path.isfile(os.path.join(class_path, item))]
        images = [f for f in files if f.split(".")[-1] in con.IMAGE_EXTENSIONS]
        paths = [os.path.join(class_path, img) for img in images]
        return paths

class OutputData:
    def __init__(self, root_path):
        self.__root = root_path

    def create(self, classes: list[str]):
        os.makedirs(self.__root, exist_ok=True)
        trn_dir_path = os.path.join(self.__root, con.TRN_DIR_NAME)
        os.makedirs(trn_dir_path, exist_ok=True)
        val_dir_path = os.path.join(self.__root, con.VAL_DIR_NAME)
        os.makedirs(val_dir_path, exist_ok=True)
        tst_dir_path = os.path.join(self.__root, con.TST_DIR_NAME)
        os.makedirs(tst_dir_path, exist_ok=True)

        for class_name in classes:
            os.makedirs(os.path.join(trn_dir_path, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir_path, class_name), exist_ok=True)
            os.makedirs(os.path.join(tst_dir_path, class_name), exist_ok=True)

    @property
    def root(self):
        return self.__root

    @property
    def classes(self):
        items = os.listdir(os.path.join(self.__root, con.TRN_DIR_NAME))
        folders = [item for item in items if os.path.isdir(os.path.join(self.__root, con.TRN_DIR_NAME, item))]
        return folders

    def get_class_image_paths(self, class_name: str, subset: str) -> list[str]:
        if class_name not in self.classes:
            raise ValueError(f"Undefined class name: {class_name}")
        if subset not in con.SUBSETS:
            raise ValueError(f"Undefined subset name: {subset}")

        class_path = os.path.join(self.__root, subset, class_name)

        items = os.listdir(class_path)
        files = [item for item in items if os.path.isfile(os.path.join(class_path, item))]
        images = [f for f in files if f.split(".")[-1] in con.IMAGE_EXTENSIONS]
        paths = [os.path.join(class_path, img) for img in images]
        return paths

    def copy_images(self, image_paths: list[str], subset: str, class_name: str):
        if class_name not in self.classes:
            raise ValueError(f"Undefined class name: {class_name}")
        if subset not in con.SUBSETS:
            raise ValueError(f"Undefined subset name: {subset}")

        class_path = os.path.join(self.__root, subset, class_name)

        for img_path in image_paths:
            shutil.copy(src=img_path, dst=class_path)


def parse_arguments():
    """Parse command-line arguments, validate paths, and return the arguments object."""
    describe = "Generate train, validation, and test subsets from a dataset."
    parser = argparse.ArgumentParser(description=describe)
    required = parser.add_argument_group("required arguments")

    required.add_argument("-i", "--input_root_dir", type=str, required=True,
                          help="Path to input root directory.")
    required.add_argument("-o", "--output_root_dir", type=str, required=True,
                          help="Path to output root directory")
    parser.add_argument("-trn", "--trn_ratio", type=int, default=TRN_RATIO,
                        help=f"Percentage of images per class for training. Default: {TRN_RATIO}%%")

    parser.add_argument("-val", "--val_ratio", type=int, default=VAL_RATIO,
                        help=f"Percentage of images per class for validation. Default: {VAL_RATIO}%%")

    args = parser.parse_args()

    validate_arguments(args)

    args.tst_ratio = 100 - (args.trn_ratio + args.val_ratio)

    return args

def validate_arguments(args):
    """Perform a quick validation of input arguments. Raise errors if there are any issues."""
    if not os.path.isdir(args.input_root_dir):
        raise DirectoryNotFound(f"Unable to find directory {args.input_root_dir}")

    if args.trn_ratio + args.val_ratio > 100:
        raise ValueError("The sum of Train and Validation percentages cannot exceed 100%")


def main():
    """Run the main sequence of procedures."""

    # parse command-line arguments
    args = parse_arguments()

    # create input and output dir handlers and folders
    input_h = InputData(root_path=args.input_root_dir)
    output_h = OutputData(root_path=args.output_root_dir)
    output_h.create(classes=input_h.classes)

    # read per-class data, shuffle, extract partitions, and copy files
    for class_name in input_h.classes:
        print(f"\nProcessing data for class {class_name}")
        image_path_list = input_h.get_class_image_paths(class_name=class_name)
        random.shuffle(image_path_list)

        num_img = len(image_path_list)
        num_trn = round(num_img * args.trn_ratio / 100)
        if args.tst_ratio == 0:
            num_val = num_img - num_trn
            num_tst = 0
        else:
            num_val = round(num_img * args.val_ratio / 100)
            num_tst = num_img - num_trn - num_val

        trn_paths = random.sample(image_path_list, num_trn)
        rem_paths = [p for p in image_path_list if p not in trn_paths]

        if args.tst_ratio == 0:
            val_paths = rem_paths
            tst_paths = []
        else:
            val_paths = random.sample(rem_paths, num_val)
            tst_paths = [p for p in rem_paths if p not in val_paths]

        print("   Number of samples in prepared lists:")
        print(f"      all images: {num_img}")
        print(f"      trn images: {len(trn_paths)}, val images: {len(val_paths)}, tst images: {len(tst_paths)}")

        print("   Creating TRN images")
        output_h.copy_images(image_paths=trn_paths, subset="TRN", class_name=class_name)
        print("   Creating VAL images")
        output_h.copy_images(image_paths=val_paths, subset="VAL", class_name=class_name)
        print("   Creating TST images")
        output_h.copy_images(image_paths=tst_paths, subset="TST", class_name=class_name)

    print("\nFinished setting up TRN, VAL, and TST datasets.")


if __name__ == "__main__":
    main()
