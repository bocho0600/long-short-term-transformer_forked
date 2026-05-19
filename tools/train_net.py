# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from rekognition_online_action_detection.utils.parser import load_cfg
from rekognition_online_action_detection.utils.env import setup_environment
from rekognition_online_action_detection.utils.checkpointer import setup_checkpointer
from rekognition_online_action_detection.utils.logger import setup_logger
from rekognition_online_action_detection.datasets import build_data_loader
from rekognition_online_action_detection.models import build_model
from rekognition_online_action_detection.criterions import build_criterion
from rekognition_online_action_detection.optimizers import build_optimizer
from rekognition_online_action_detection.optimizers import build_scheduler
from rekognition_online_action_detection.engines import do_train


def main(cfg):

    # Calls setup_environment(cfg) to choose GPU/CPU and seed randomness.
    device = setup_environment(cfg)

    # Calls setup_checkpointer(cfg, phase='train') to prepare resume/save behavior from previous checkpoints. Calls setup_logger(cfg, phase='train') to prepare logging behavior.
    checkpointer = setup_checkpointer(cfg, phase='train')

    # Calls setup_logger(cfg, phase='train') to log config and epoch results.
    logger = setup_logger(cfg, phase='train')

    # Builds one DataLoader per phase in cfg.SOLVER.PHASES, usually train and test.
    data_loaders = {
        phase: build_data_loader(cfg, phase)
        for phase in cfg.SOLVER.PHASES
    }

    # I DON'T UNDERSTAND FROM HERE!!!!!!!!!
    # Calls build_model(cfg, device) to construct LSTRStream. (what is LSTRStream???)
    model = build_model(cfg, device)

    # Calls build_criterion(cfg, device) to create losses.
    criterion = build_criterion(cfg, device)

    # Calls build_optimizer(cfg, model) to create SGD/Adam/AdamW (what is SGD/Adam/AdamW???).
    optimizer = build_optimizer(cfg, model)

    # Calls checkpointer.load(model, optimizer) to resume weights and optimizer state if a checkpoint exists (why does load checkpointer is called after the setup checkpointer???).
    checkpointer.load(model, optimizer)

    # Build scheduler (what is scheduler???)
    scheduler = build_scheduler(
        cfg, optimizer, len(data_loaders['train']))

    # Is it the most important function in this code? Calls do_train() to run the training loop.
    do_train(
        cfg,
        data_loaders,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        checkpointer,
        logger,
    )


if __name__ == '__main__':
    main(load_cfg())
