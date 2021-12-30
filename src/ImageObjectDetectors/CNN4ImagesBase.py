from enum import Enum


class KernelProgression(Enum):
    KERNEL_GETS_BIGGER = 1
    KERNEL_GETS_SMALLER = 2


class CNN4ImagesBase:
    DEFAULT_LEARNING_RATE = 0.003
    DEFAULT_SEED = 42
    DEFAULT_KERNEL_PROGRESSION = KernelProgression.KERNEL_GETS_BIGGER
    H5_FILE_TYPE = ".h5"

    @staticmethod
    def add_file_type(model_filename):
        if not model_filename.endswith(CNN4ImagesBase.H5_FILE_TYPE):
            model_filename += CNN4ImagesBase.H5_FILE_TYPE
        return model_filename
