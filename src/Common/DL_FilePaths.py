from pathlib import Path
from Common.Platforms import in_mac_os, in_linux
from enum import Enum


class FIGHTER_JET(Enum):
    F_22 = 1
    F_35 = 2
    F_18 = 3
    F_16 = 4
    F_15 = 5
    F_14 = 6


class IMG_SIZE(Enum):
    IS_1920 = 1920
    IS_960 = 960


HOME_DIR = str(Path.home())
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)

if in_linux():
    DEEP_LEARNING_DIR = HOME_DIR + '/Desktop/DeepLearning'
elif in_mac_os():
    DEEP_LEARNING_DIR = HOME_DIR + '/DeepLearning'
else:
    DEEP_LEARNING_DIR = HOME_DIR

KAGGLE_DIR = DEEP_LEARNING_DIR + '/Kaggle_Competitions'
KAGGLE_TITANIC_DIR = KAGGLE_DIR + '/Titanic'

F_22_DETECTOR_DIR = DEEP_LEARNING_DIR + '/F-22_Detector'
SIZE_1920_DIR = F_22_DETECTOR_DIR + '/1920'
SIZE_960_DIR = F_22_DETECTOR_DIR + '/960'

F_22_RELPATH = '/F-22s'
F_14_RELPATH = '/F-14s'
F_15_RELPATH = '/F-15s'
F_16_RELPATH = '/F-16s'
F_18_RELPATH = '/F-18s'
F_35_RELPATH = '/F-35s'


FIGHTER_JET_SIZE_DIRS = {
    IMG_SIZE.IS_1920: {
        FIGHTER_JET.F_14: SIZE_1920_DIR + F_14_RELPATH,
        FIGHTER_JET.F_15: SIZE_1920_DIR + F_15_RELPATH,
        FIGHTER_JET.F_16: SIZE_1920_DIR + F_16_RELPATH,
        FIGHTER_JET.F_18: SIZE_1920_DIR + F_18_RELPATH,
        FIGHTER_JET.F_22: SIZE_1920_DIR + F_22_RELPATH,
        FIGHTER_JET.F_35: SIZE_1920_DIR + F_35_RELPATH
    },
    IMG_SIZE.IS_960: {
        FIGHTER_JET.F_14: SIZE_960_DIR + F_14_RELPATH,
        FIGHTER_JET.F_15: SIZE_960_DIR + F_15_RELPATH,
        FIGHTER_JET.F_16: SIZE_960_DIR + F_16_RELPATH,
        FIGHTER_JET.F_18: SIZE_960_DIR + F_18_RELPATH,
        FIGHTER_JET.F_22: SIZE_960_DIR + F_22_RELPATH,
        FIGHTER_JET.F_35: SIZE_960_DIR + F_35_RELPATH
    }
}


def get_fighter_jet_dir(jet, size=IMG_SIZE.IS_1920):
    fdir = FIGHTER_JET_SIZE_DIRS.get(size).get(jet)
    if fdir is not None:
        return fdir
    else:
        print(f'fdir none for jet: {jet}, size: {size}')
        return None
