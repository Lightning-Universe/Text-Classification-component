import os
import shutil
import tempfile
import urllib

import torchtext.datasets
import csv

from lai_textclf.data import YelpReviewFull
#
# dataset = torchtext.datasets.YelpReviewFull("dataset/yelp-full", split="test")
# # next(iter(dataset))x
# from torch.utils.data import Dataset

with tempfile.TemporaryDirectory() as download_dir:
    filename = os.path.join(download_dir, "data.tar.gz")
    r = urllib.request.urlretrieve(url="https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbZlU4dXhHTFhZQU0",
                               filename=filename)
    shutil.unpack_archive(filename=filename, extract_dir=download_dir)
    train_dset = YelpReviewFull(csv_file=os.path.join(download_dir, "train.csv"))
    val_dset = YelpReviewFull(csv_file=os.path.join(download_dir, "test.csv"))
