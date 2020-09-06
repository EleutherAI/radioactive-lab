# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import argparse
import torch
from torch import nn

from src.slurm import init_signal_handler
from src.utils import bool_flag, init_distributed_mode, initialize_exp
from src.model import check_model_params
from src.model import build_model
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.dataset import get_data_loader, load_dataset_params
import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Metadata Warning, tag [0-9]+ had too many entries", UserWarning)


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description='Language transfer')

    # main parameters
    parser.add_argument("--dump_path", type=str, default="")
    parser.add_argument("--exp_name", type=str, default="bypass")
    parser.add_argument("--save_periodic", type=int, default=0)
    parser.add_argument("--exp_id", type=str, default="")
    parser.add_argument("--nb_workers", type=int, default=10)
    parser.add_argument("--fp16", type=bool_flag, default=False)

    # dataset
    parser.add_argument("--dataset", type=str, default="cifar10")
    # parser.add_argument("--crop_size", type=int, default=224,
    #                     help="Crop size")
    # parser.add_argument("--img_size", type=int, default=256,
    #                     help="Img resize")

    # model type
    parser.add_argument("--architecture", type=str, default="myresnet2")
    parser.add_argument("--non_linearity", type=str, default="relu",
                        help="Non linearity")
    parser.add_argument("--pretrained", type=bool_flag, default=False,
                        help="Use a pretrained model")
    parser.add_argument("--teacher_ckpt", type=str, default="")
    parser.add_argument("--load_linear", type=bool_flag, default=False)
    parser.add_argument("--train_path", type=str, default="vanilla_train")
    parser.add_argument("--num_classes", type=int, default=-1,
                        help="Number of subclasses to use")

    # training parameters
    parser.add_argument("--optimizer", type=str, default="sgd,lr=0.1-0.01-0.001,momentum=0.9,weight_decay=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Number of sentences per batch")
    parser.add_argument("--epochs", type=int, default=90,
                        help="Number of epochs")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--train_transform", choices=["random", "flip", "center"], default="random",
                        help="Transformation applied to training images")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--temperature", type=float, default=1,
                        help="Distillation temperature")

    # reload
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")

    # evaluation
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")

    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug from a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")


    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    return parser


def main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # initialize the experiment / load data
    logger = initialize_exp(params)

    # Seed
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)

    # initialize SLURM signal handler for time limit / pre-emption
    if params.is_slurm_job:
        init_signal_handler()

    # data loaders / samplers
    load_dataset_params(params)
    train_data_loader, train_sampler, _ = get_data_loader(
        img_size=params.img_size,
        crop_size=params.crop_size,
        shuffle=True,
        batch_size=params.batch_size,
        num_classes=params.num_classes,
        nb_workers=params.nb_workers,
        distributed_sampler=params.multi_gpu,
        dataset=params.dataset,
        data_path=params.train_path,
        transform=params.train_transform,
        split='valid' if params.debug_train else 'train',
        seed=params.seed
    )

    valid_data_loader, _, _ = get_data_loader(
        img_size=params.img_size,
        crop_size=params.crop_size,
        shuffle=False,
        batch_size=params.batch_size,
        num_classes=params.num_classes,
        nb_workers=params.nb_workers,
        distributed_sampler=False,
        dataset=params.dataset,
        transform='center',
        split='valid',
        seed=params.seed
    )

    # build model / cuda
    logger.info("Building %s model ..." % params.architecture)
    model = build_model(params)
    model.cuda()

    teacher_model = build_model(params)
    teacher_model.cuda()
    ckpt = torch.load(params.teacher_ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
    missing_keys, unexcepted_keys = teacher_model.load_state_dict(state_dict, strict=False)
    print("Missing keys: ", missing_keys)
    print("Unexcepted keys: ", unexcepted_keys)

    # distributed  # TODO: check this https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main.py#L142
    if params.multi_gpu:
        logger.info("Using nn.parallel.DistributedDataParallel ...")
        model = nn.parallel.DistributedDataParallel(model, device_ids=[params.local_rank], output_device=params.local_rank, broadcast_buffers=True)

    # build trainer / reload potential checkpoints / build evaluator
    trainer = Trainer(model=model, params=params, teacher_model=teacher_model)
    trainer.reload_checkpoint()
    evaluator = Evaluator(trainer, params)

    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals(trainer, evals=['classif', 'recognition'], data_loader=valid_data_loader)

        for k, v in scores.items():
            logger.info('%s -> %.6f' % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # training
    for epoch in range(trainer.epoch, params.epochs):

        # update epoch / sampler / learning rate
        trainer.epoch = epoch
        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)
        if params.multi_gpu:
            train_sampler.set_epoch(epoch)

        # update learning rate
        trainer.update_learning_rate()

        # train
        for i, (images, targets) in enumerate(train_data_loader):
            trainer.distillation_step(images, None)
            trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate classification accuracy
        scores = evaluator.run_all_evals(trainer, evals=['classif'], data_loader=valid_data_loader)

        for name, val in trainer.get_scores().items():
            scores[name] = val

        # print / JSON log
        for k, v in scores.items():
            logger.info('%s -> %.6f' % (k, v))
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # debug mode
    if params.debug is True:
        params.exp_name = 'debug'
        params.debug_slurm = True
        params.debug_train = True

    # check parameters
    check_model_params(params)

    # run experiment
    main(params)
