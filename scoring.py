# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich, Andreas Schörgenhumer
Contact -- schoergenhumer@ml.jku.at
Date -- 01.07.2022

###############################################################################

The following copyright statement applies to all code within this file.

Copyright statement:
This  material,  no  matter  whether  in  printed  or  electronic  form,
may  be  used  for personal  and non-commercial educational use only.
Any reproduction of this manuscript, no matter whether as a whole or in parts,
no matter whether in printed or in electronic form, requires explicit prior
acceptance of the authors.

###############################################################################

"""
import argparse
import bz2
import gzip
import lzma
import os
import re
import zipfile

import dill as pkl
import numpy as np
import onnx
import onnxruntime
import torch.jit

TEST_DATA_PATH = r"/daten/challenge/django/data/datasets/image_inpainting_2022/test.zip"


def load_data(file: str):
    if file.endswith(".zip"):
        # "mode" cannot be "rb", so set it manually to "r" (still need the parameter or the function invocation fails)
        # noinspection PyUnusedLocal
        def zip_open(file_, mode):
            with zipfile.ZipFile(file_, "r") as myzip:
                return myzip.open(myzip.namelist()[0])

        open_fn = zip_open
    elif file.endswith(".bz2"):
        open_fn = bz2.open
    elif file.endswith(".xz"):
        open_fn = lzma.open
    elif file.endswith(".gz"):
        open_fn = gzip.open
    else:
        open_fn = open
    with open_fn(file, "rb") as pfh:
        return pkl.load(pfh)


def rmse(predictions: list, targets: list):
    def rmse_(prediction_array: np.ndarray, target_array: np.ndarray):
        if prediction_array.shape != target_array.shape:
            raise IndexError(f"Target shape is {target_array.shape} but prediction shape is {prediction_array.shape}")
        prediction_array, target_array = np.asarray(prediction_array, np.float64), np.asarray(target_array, np.float64)
        return np.sqrt(np.mean((prediction_array - target_array) ** 2))

    # Compute RMSE for each sample
    rmses = [rmse_(prediction, target) for prediction, target in zip(predictions, targets)]
    return np.mean(rmses)


def scoring_file(prediction_file: str, target_file: str):
    """Computes the mean RMSE loss on two lists of numpy arrays stored in pickle files prediction_file and targets_file

    Computation of mean RMSE loss, as used in the challenge for exercise 5. See files "example_testset.pkl" and
    "example_submission_random.pkl" for an example test set and example targets, respectively. The real test set
    (without targets) will be available as download (see assignment sheet 2).

    Parameters
    ----------
    prediction_file: str
        File path of prediction file. Has to be a pickle file (or dill file) and contain a list of numpy arrays of dtype
        uint8, as specified in assignment sheet 2. The file can optionally be compressed, which will be automatically
        determined based on its file extension, of which the following are supported:
        > ".zip": zip compression (https://docs.python.org/3/library/zipfile.html, including the requirement of the zlib
          module: https://docs.python.org/3/library/zlib.html)
        > ".gz": gzip compression (https://docs.python.org/3/library/gzip.html, also requires the zlib module)
        > ".bz2": bzip2 compression (https://docs.python.org/3/library/bz2.html)
        > ".xz": lzma compression (https://docs.python.org/3/library/lzma.html)
        If none of these file extensions match, it is assumed to be a raw pickle file.
    target_file: str
        File path of target file. Has to be a pickle file (or dill file) and contain a list of numpy arrays of dtype
        uint8, as specified in assignment sheet 2. The file can optionally be compressed (refer to "predictions_file"
        above for more details). This file will not be available for the challenge.
    """
    # Load predictions
    predictions = load_data(prediction_file)
    if not isinstance(predictions, list):
        raise TypeError(f"Expected a list of numpy arrays as pickle file. "
                        f"Got {type(predictions)} object in pickle file instead.")
    if not all([isinstance(prediction, np.ndarray) and np.uint8 == prediction.dtype
                for prediction in predictions]):
        raise TypeError("List of predictions contains elements which are not numpy arrays of dtype uint8")

    # Load targets
    targets = load_data(target_file)
    if len(targets) != len(predictions):
        raise IndexError(f"list of targets has {len(targets)} elements "
                         f"but list of submitted predictions has {len(predictions)} elements.")

    return rmse(predictions, targets)


class Normalizer:

    def __init__(self, shift, scale):
        self.shift = shift
        self.scale = scale

    def normalize(self, x):
        return (x - self.shift) / self.scale

    def denormalize(self, x):
        return x * self.scale + self.shift


def make_predictions(model_output_provider, test_data: np.ndarray, normalizer: Normalizer):
    n_samples = len(test_data["input_arrays"])

    # Create predictions for each sample (one by one)
    predictions = []
    for sample_i in range(n_samples):
        # Normalize input by maximal value
        input_array = normalizer.normalize(test_data["input_arrays"][sample_i].astype(np.float32))
        known_array = test_data["known_arrays"][sample_i].astype(np.float32)
        # Stack both inputs for the network
        input_array = np.concatenate([input_array, known_array], axis=0)
        # Pretend we have a minibatch dimension
        inputs = input_array[None]  # Add empty dimension

        # Get outputs for network, which is a list of predictions for each sample, and
        # since we only have one sample (pretended minibatch), extract its single prediction
        prediction = model_output_provider(inputs)[0]
        # De-normalize prediction
        prediction = normalizer.denormalize(prediction)
        # Clip the predictions to a valid range (we know our prediction values can only be in range 0-255 because of
        # uint8 datatype!)
        prediction = np.clip(prediction, a_min=0, a_max=255)
        # Challenge server wants uint8 datatype for predictions
        prediction = np.asarray(prediction, dtype=np.uint8)
        # Add prediction for sample to list
        predictions.append(prediction)

    return predictions


def scoring_model(model_file: str, test_file: str, target_file: str):
    """
    Computation of mean RMSE loss, as used in the challenge for exercise 5. The targets are loaded from the specified
    "target_file" (pickle file containing list of numpy arrays), whereas the predictions are created using the model
    stored at "model_file" using the original testset input data stored at "test_file".

    Parameters
    ----------
    model_file : str
        File path of the stored (trained) model. The model must be in ONNX or TorchScript format (which one is
        determined by the file extension: ".onnx" for ONNX or ".pt" for TorchScript), and the model output must be the
        list of predictions, where each prediction is a 1D torch.Tensor object (not a numpy array!) of an arbitrary
        numeric datatype, containing the predicted missing pixel values. These predictions will then automatically be
        converted to a numpy array, its values will be clipped to range [0, 255], and the datatype will be cast to
        numpy's uint8. The input to the model will be the concatenated image data and the known array data from the
        original testset input data (see parameter "test_file"), and the batch size is fixed to 1, i.e., the input
        shape is (N=1, C=6, H=100, W=100).

        This parameter also controls the normalization that can be applied to the input image data, before it is passed
        to the model. If this option is enabled, de-normalization is also automatically performed on the model output,
        i.e., the predictions, before they are clipped to range [0, 255] and converted to a numpy uint8 array. The
        normalization is based on two user-specifiable values: "shift" and "scale". Normalization is then done by
        calculating "(x - shift) / scale" and de-normalization via "y * scale + shift", where "x" is the image input
        data and "y" is the model output data (the predictions). "shift" and "scale" can be specified directly by the
        filename of "model_file": To do so, simply provide a filename that somewhere contains "shift=A" and "scale=B",
        where "A" and "B" can be arbitrary integer or floating point numbers. Example: "model_scale=255shift=1.5.pt",
        which would translate to a "scale" of 255 and a shift of "1.5" (for a TorchScript model, in this case). If the
        two parameters are not specified, no scaling/shifting will be done.

        In case of an ONNX model, GPU acceleration is performed automatically. In case of a TorchScript model, the
        filename again determines whether to run the model on the CPU (default) or on the GPU. If the TorchScript model
        should be run on the GPU, the filename must contain "gpu" somewhere. Example: "model_gpu.pt", which would then
        lead to running the model on the GPU. Note that the original model must have been moved to the GPU prior to
        exporting as a TorchScript model, or otherwise, errors or even unexpected behavior might occur (see FAQ:
        https://pytorch.org/docs/stable/jit.html#frequently-asked-questions).
    test_file: str
        File path of the original testset input data, which is a pickle file containing a dictionary with the following
        entries: "input_arrays" (list of numpy arrays), "known_arrays" (list of numpy arrays), "offsets" (list of
        integer 2-tuples), "spacings" (list of integer 2-tuples), "sample_ids" (list of strings). The file can
        optionally be compressed, which will be automatically determined based on its file extension, of which the
        following are supported:
        > ".zip": zip compression (https://docs.python.org/3/library/zipfile.html, including the requirement of the zlib
          module: https://docs.python.org/3/library/zlib.html)
        > ".gz": gzip compression (https://docs.python.org/3/library/gzip.html, also requires the zlib module)
        > ".bz2": bzip2 compression (https://docs.python.org/3/library/bz2.html)
        > ".xz": lzma compression (https://docs.python.org/3/library/lzma.html)
        If none of these file extensions match, it is assumed to be a raw pickle file.
    target_file: str
        File path of target file. Has to be a pickle file (or dill file) and contain a list of numpy arrays of dtype
        uint8, as specified in assignment sheet 2. The file can optionally be compressed (refer to "test_file" above
        for more details). This file will not be available for the challenge.
    """
    # TODO: provide config file via upload instead of extracting properties from the filename
    filename = os.path.basename(model_file)
    shift = re.search(r"shift=((\.|\d+)\.?\d*)", filename)
    scale = re.search(r"scale=((\.|\d+)\.?\d*)", filename)
    shift = float(shift.group(1)) if shift is not None else 0  # shift=0 equals no change
    scale = float(scale.group(1)) if scale is not None else 1  # scale=1 equals no change
    normalizer = Normalizer(shift, scale)

    if model_file.endswith(".onnx"):
        onnx_model = onnx.load_model(model_file)
        onnx.checker.check_model(onnx_model)
        session = onnxruntime.InferenceSession(model_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        def model_output_provider(inputs):
            inputs = {session.get_inputs()[0].name: inputs}
            outputs = session.run(None, inputs)[0]  # Get first return value
            return outputs
    elif model_file.endswith(".pt"):
        model = torch.jit.load(model_file)
        device = "cuda" if "gpu" in filename else "cpu"

        def model_output_provider(inputs):
            inputs = torch.from_numpy(inputs).to(device=device)
            return [o.detach().cpu().numpy() for o in model(inputs)]
    else:
        raise ValueError("'model_file' must either be an ONNX file ('.onnx') or a TorchScript file ('.pt')")

    targets = load_data(target_file)
    test_data = load_data(test_file)
    predictions = make_predictions(model_output_provider, test_data, normalizer)

    return rmse(predictions, targets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=str, help="Path to submission file")
    parser.add_argument("--target", type=str, default=None, help="Path to target file")
    args = parser.parse_args()

    # Infer the type of submission: 1) exported ONNX/TorchScript model or 2) predictions file
    if args.submission.endswith(".onnx") or args.submission.endswith(".pt"):
        mse_loss = scoring_model(model_file=args.submission, test_file=TEST_DATA_PATH, target_file=args.target)
    else:
        # Prediction files are too big to keep, so ensure that they are always deleted after use
        try:
            mse_loss = scoring_file(prediction_file=args.submission, target_file=args.target)
        finally:
            if os.path.exists(args.submission):
                os.remove(args.submission)

    print(mse_loss)