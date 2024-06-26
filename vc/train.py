#!/usr/bin/env python3
# coding:utf-8

from logging import StreamHandler
import logging
from vc.Utils.JDC.model import JDCNet
from vc.Utils.ASR.models import ASRCNN
from torch.utils.tensorboard import SummaryWriter
from vc.trainer import Trainer
from vc.models import build_model
from vc.optimizers import build_optimizer
from vc.meldataset import build_dataloader
from munch import Munch
from functools import reduce
import os
import os.path as osp
import re
import sys
import yaml
import shutil
import numpy as np
import torch
import click
import warnings
warnings.simplefilter('ignore')


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = StreamHandler()
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

torch.backends.cudnn.benchmark = True


#@click.command()
#@click.option('-p', '--config_path', default='Configs/config.yml', type=str)
class Train:
    def __init__(self, user_id):
        self.user_id = user_id
        self.config_path = "./vc/Configs/p" + str(user_id) + "/config.yml"

    def get_data_path_list(self, train_path=None, val_path=None):
        if train_path is None:
            train_path = "./vc/Data/p" + str(self.user_id) + "/train_list.txt"
        if val_path is None:
            val_path = "./vc/Data/p" + str(self.user_id) + "/val_list.txt"

        with open(train_path, 'r') as f:
            train_list = f.readlines()
        with open(val_path, 'r') as f:
            val_list = f.readlines()

        return train_list, val_list

    def train(self):
        config = yaml.safe_load(open(self.config_path))

        log_dir = "./vc/Models/p" + str(self.user_id)
        if not osp.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        shutil.copy(self.config_path, osp.join(
            log_dir, osp.basename(self.config_path)))
        writer = SummaryWriter(log_dir + "/tensorboard")

        # write logs
        file_handler = logging.FileHandler(osp.join(log_dir, 'train.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            '%(levelname)s:%(asctime)s: %(message)s'))
        logger.addHandler(file_handler)

        batch_size = config.get('batch_size', 10)
        device = config.get('device', 'cpu')
        epochs = config.get('epochs', 1000)
        save_freq = config.get('save_freq', 20)
        train_path = "./vc/Data/p" + str(self.user_id) + "/train_list.txt"
        val_path = "./vc/Data/p" + str(self.user_id) + "/val_list.txt"
        stage = config.get('stage', 'star')
        fp16_run = config.get('fp16_run', False)

        # load data
        train_list, val_list = self.get_data_path_list(train_path, val_path)
        train_dataloader = build_dataloader(train_list,
                                            batch_size=batch_size,
                                            num_workers=4,
                                            device=device)
        val_dataloader = build_dataloader(val_list,
                                          batch_size=batch_size,
                                          validation=True,
                                          num_workers=2,
                                          device=device)

        # load pretrained ASR model
        ASR_config = config.get('ASR_config', False)
        ASR_path = config.get('ASR_path', False)
        with open(ASR_config) as f:
            ASR_config = yaml.safe_load(f)
        ASR_model_config = ASR_config['model_params']
        ASR_model = ASRCNN(**ASR_model_config)
        params = torch.load(ASR_path, map_location='cpu')['model']
        ASR_model.load_state_dict(params)
        _ = ASR_model.eval()

        # load pretrained F0 model
        F0_path = config.get('F0_path', False)
        F0_model = JDCNet(num_class=1, seq_len=192)
        params = torch.load(F0_path, map_location='cpu')['net']
        F0_model.load_state_dict(params)

        # build model
        model, model_ema = build_model(
            Munch(config['model_params']), F0_model, ASR_model)

        scheduler_params = {
            "max_lr": float(config['optimizer_params'].get('lr', 2e-4)),
            "pct_start": float(config['optimizer_params'].get('pct_start', 0.0)),
            "epochs": epochs,
            "steps_per_epoch": len(train_dataloader),
        }

        _ = [model[key].to(device) for key in model]
        _ = [model_ema[key].to(device) for key in model_ema]
        scheduler_params_dict = {key: scheduler_params.copy() for key in model}
        scheduler_params_dict['mapping_network']['max_lr'] = 2e-6
        optimizer = build_optimizer({key: model[key].parameters() for key in model},
                                    scheduler_params_dict=scheduler_params_dict)

        trainer = Trainer(args=Munch(config['loss_params']), model=model,
                          model_ema=model_ema,
                          optimizer=optimizer,
                          device=device,
                          train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          logger=logger,
                          fp16_run=fp16_run)

        # Config/p{user_id}/config.yml에 pretrained model 추가
        if config.get('pretrained_model', '') != '':
            trainer.load_checkpoint(config['pretrained_model'],
                                    load_only_params=config.get('load_only_params', True))

        for _ in range(1, epochs+1):
            epoch = trainer.epochs
            train_results = trainer._train_epoch()
            eval_results = trainer._eval_epoch()
            results = train_results.copy()
            results.update(eval_results)
            logger.info('--- epoch %d ---' % epoch)
            for key, value in results.items():
                if isinstance(value, float):
                    logger.info('%-15s: %.4f' % (key, value))
                    writer.add_scalar(key, value, epoch)
                else:
                    for v in value:
                        writer.add_figure('eval_spec', v, epoch)
            if (epoch % save_freq) == 0:  # 동작 확인 후 save_freq 148로 수정
                trainer.save_checkpoint(
                    osp.join(log_dir, 'epoch_%05d.pth' % epoch))

        return 0
