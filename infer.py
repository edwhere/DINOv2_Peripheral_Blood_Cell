""" Perform inference using a trained model and a directory of images."""
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as nnfun
from tqdm.auto import tqdm

import model as mod
import datahandler as dhl
import errordefs as err

INFER_BATCH_SIZE = 8


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Perform inference using a trained model.")
    required = parser.add_argument_group("requirements:")

    required.add_argument("--image_dir", type=str, required=True,
                          help="Path to a directory with images.")

    required.add_argument("--model_file", type=str, required=True,
                          help="Path to a trained model (.pth file).")

    parser.add_argument("--results_file", type=str,
                        help="Path for a file that will contain results in CSV format. If this argument is omitted, "
                             "the results are not saved.")

    parser.add_argument("--show_results", action="store_true",
                        help="Display the results on the screen.")

    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise err.DirectoryNotFoundError(f"Unable to access directory: {args.image_dir}")

    if not os.path.isfile(args.model_file):
        raise FileNotFoundError(f"Unable to find model file: {args.model_file}")

    return args


def get_device():
    """Determine if a machine runs on a gpu or cpu. Includes M1/M2/M3 gpus. Return the device
    identifier for PyTorch operations.
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device


def get_batch_results(probs_array: np.ndarray, labels: list[str], filenames: list[str]):
    """Create a table with key results. The probs array represents the model output, which is a
    matrix of size B x C, where B is batch size and C is number of classes. The labels are the
    class names. The list of file names includes the image files that were processed in this batch.
    Returns:
        A dataframe with columns: filename, one column per label, prediction.
        The label columns have the probabilities. The prediction column has the selected label (highest prob.).
        The filename column is the image file name.
    """
    df = pd.DataFrame(probs_array, columns=labels)
    df.insert(0, 'filename', pd.Series(filenames))

    max_indices = np.argmax(probs_array, axis=1)
    selected = [labels[i] for i in max_indices]

    df["prediction"] = selected

    return df


def infer(model_data, infer_data_loader, device):
    """Perform inference on the data loaded by infer_data_loader using the designated model and
    the selected target device.
    Returns:
        A pandas dataframe with the results.
    """
    model = model_data[0]
    labels = model_data[1]

    model.eval()
    model = model.to(device)

    print('Performing inference')
    counter = 0

    with torch.no_grad():
        batch_dataframes = []
        for i, image_data in tqdm(enumerate(infer_data_loader), total=len(infer_data_loader)):
            counter += 1

            img_tensors = image_data['image']
            fnames = image_data["filename"]

            img_tensors = img_tensors.to(device)
            outputs = model(img_tensors)

            # Calculate the predictions
            print("type of outputs:", type(outputs))
            print("shape of outputs:", outputs.shape)


            probs = nnfun.softmax(outputs.data, dim=1)
            probs_arr = probs.detach().cpu().numpy()

            batch_df = get_batch_results(probs_arr, labels, fnames)
            batch_dataframes.append(batch_df)

            # _, preds = torch.max(outputs.data, 1)
            # preds_arr = preds.detach().cpu().numpy()

    result_df = pd.concat(batch_dataframes, axis=0, ignore_index=True)
    return result_df



def main():
    """Perform main sequence of operations."""

    args = parse_arguments()

    # Get inference data loader
    inference_dataset = dhl.get_infer_dataset(args.image_dir)
    inference_loader = dhl.get_infer_loader(dataset_infer=inference_dataset, batch_size=INFER_BATCH_SIZE)

    # Get stored model
    model_data = mod.load_stored_data(file_path=args.model_file)

    # Get target device
    device = get_device()

    # Run inference method
    res_df = infer(model_data, inference_loader, device)

    if args.show_results:
        print("\n--- Results -------------------------------------------------------")
        print(res_df)
        print()

    if args.results_file:
        res_df.to_csv(args.results_file, index=False)
        print(f"\nSaved inference results to {args.results_file}")
    else:
        print("\nResults have not been saved since the user did not provide a path for saving.")


if __name__ == "__main__":
    main()
