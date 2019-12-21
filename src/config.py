import torch
import os

# device
DEVICE = torch.device("cuda:0")

# paths
INPUT = '/root/user/car-disk'
TRAIN_CSV = os.path.join(INPUT, 'train.csv')
TRAIN_IMAGE = os.path.join(INPUT, 'train_images')
TRAIN_MASK = os.path.join(INPUT, 'train_masks')
TEST_IMAGE = os.path.join(INPUT, 'test_images')
TEST_MASK = os.path.join(INPUT, 'test_masks')

# train/valid/predict
EPOCHS = 2
BATCH_SIZE = 4
NUM_WORKERS = 8
PRINT_FREQ = 10
ACC_ITER = 8

# image
# aspect ratio(approx): 4: 5
IMG_SIZE = (2710, 3384)
INPUT_SIZE = (128 * 4, 128 * 5)
SCALE = IMG_SIZE[1] / INPUT_SIZE[1]

# object detection
MAX_OBJ = 50
MODEL_SCALE = 4
OUTPUT_WIDTH = INPUT_SIZE[1] // MODEL_SCALE
OUTPUT_HEIGHT = INPUT_SIZE[0] // MODEL_SCALE

# loss weights
HM_WEIGHT = 1
OFFSET_WEIGHT = 1
DEPTH_WEIGHT = 1
ROTATE_WEIGHT = 1

# optimizer
ADAM_LR = 1e-5

# clear duplicates
DISTANCE_THRESH_CLEAR = 2
