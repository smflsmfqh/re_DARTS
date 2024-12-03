import os
import sys
import time
import glob
import numpy as np
import torch
import json
import codecs
import yaml
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, '../RobustDARTS')

from src import utils
from src.utils import Genotype
from src.evaluation.model import Network
from src.evaluation.args import Helper

TORCH_VERSION = torch.__version__


helper = Helper()
args = helper.config

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save,
                                      'log_{}_{}.txt'.format(args.search_task_id, args.task_id)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# 파라미터 크기 측정
def count_parameters(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = pytorch_total_params - trainable_params
    return {
        "Total Parameters": pytorch_total_params,
        "Trainable Parameters": trainable_params,
        "Non-trainable Parameters": non_trainable_params
    }

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  # load search configuration file holding the found architectures
  configuration = '_'.join([args.space, args.dataset])
  settings = '_'.join([str(args.search_dp), str(args.search_wd)])
  with open(args.archs_config_file, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.CLoader)
    arch = dict(cfg)[configuration][settings][args.search_task_id]

  print(arch)
  genotype = eval(arch)
  model = Network(args.init_channels, args.n_classes, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  if args.model_path is not None:
    utils.load(model, args.model_path, genotype)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  # 파라미터 상세 정보 출력
  param_info = count_parameters(model)
  logging.info(f"Total Parameters: {param_info['Total Parameters']}")
  logging.info(f"Trainable Parameters: {param_info['Trainable Parameters']}")
  logging.info(f"Non-trainable Parameters: {param_info['Non-trainable Parameters']}")


  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  scheduler = CosineAnnealingLR(
      optimizer, float(args.epochs))

  train_queue, valid_queue, _, _ = helper.get_train_val_loaders()

  errors_dict = {'train_acc': [], 'train_loss': [], 'valid_acc': [],
                 'valid_loss': []}

  for epoch in range(args.epochs):
    logging.info('epoch %d lr %e', epoch, scheduler.get_last_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    # training
    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)

    # evaluation
    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    # update the errors dictionary
    errors_dict['train_acc'].append(100 - train_acc)
    errors_dict['train_loss'].append(train_obj)
    errors_dict['valid_acc'].append(100 - valid_acc)
    errors_dict['valid_loss'].append(valid_obj)

    scheduler.step()

  with codecs.open(os.path.join(args.save,
                                'errors_{}_{}.json'.format(args.search_task_id, args.task_id)),
                   'w', encoding='utf-8') as file:
    json.dump(errors_dict, file, separators=(',', ':'))

  utils.write_yaml_results_eval(args, args.results_test, 100-valid_acc)

def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda(non_blocking=True)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      if args.debug:
        break

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  batch_times = []  # 각 배치 시간을 저장할 리스트

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input).cuda()
      target = Variable(target).cuda(non_blocking=True)

      step_start_time = time.time()
      logits, _ = model(input)
      step_end_time = time.time()

      step_time = step_end_time - step_start_time
      batch_times.append(step_time)  # 배치 시간 기록

      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        if args.debug:
          break
  
  avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0.0  # 평균 배치 시간 계산
  total_time = sum(batch_times)

  logging.info(f"Total time: {total_time:.2f} seconds")
  logging.info(f"Average batch time: {avg_batch_time:.4f} seconds")

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

