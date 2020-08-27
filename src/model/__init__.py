# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from logging import getLogger
from torchvision import models

EMBEDDING_SIZE = {
    "resnet18": 512,
    "resnet50": 2048
}

logger = getLogger()

def check_model_params(params):
    if hasattr(params, "train_path") and params.train_path == "none":
        params.train_path = ""

    if hasattr(params, "from_ckpt") and params.from_ckpt == "none":
        params.from_ckpt = ""


def build_model(params):
    if params.architecture in models.__dict__ and callable(models.__dict__[params.architecture]):
        model = models.__dict__[params.architecture](num_classes=params.num_classes)
    else:
        assert False, "Architecture not recognized"

    return model
