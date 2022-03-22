import os
import sys
import time
import torch
import logging
import argparse
import torch.nn as nn
from torch import optim
import utils
import metrics
from config import tag_dict
from models.lstm import LSTM
from models.transformer import Transformer
from dataset.dataloader import DataLoader

parser = argparse.ArgumentParser("CLUE NER")
parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.00001, help='min learning rate')
parser.add_argument('--seed', default=1234, type=int, help='random seed')
parser.add_argument('--epochs', default=5, type=int, help='training epoch')
parser.add_argument('--batch_size', default=32, type=int, help='training batch size')
parser.add_argument('--arch', default='transformer', type=str, help='')
parser.add_argument('--embedding_size', default=128, type=int, help='embedding size')
parser.add_argument('--hidden_size', default=200, type=int, help='hidden size')
parser.add_argument('--model_dim', default=128, type=int, help='model dimension in transformer')
parser.add_argument('--num_blocks', default=2, type=int, help='block size in transformer')
parser.add_argument('--num_heads', default=4, type=int, help='attention heads in transformer')
parser.add_argument('--feedforward_dim', default=512, type=int, help='ff dimension in transformer')
parser.add_argument('--gpu', default='', type=str, help='gpu device')
parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
parser.add_argument('--grad_norm', default=5.0, type=float, help='Max gradient norm.')
parser.add_argument('--save', type=str, default='./outputs', help='save directory')
parser.add_argument('--data_dir', type=str, default='./cluener', help='data directory')

args = parser.parse_args()
if args.gpu != '':
    args.device = torch.device(f"cuda:{args.gpu}")
else:
    args.device = torch.device("cpu")
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    word_dict = utils.build_vocab(args.data_dir, min_freq=10)
    train_dataset = utils.get_train_examples(args.data_dir)
    train_loader = DataLoader(train_dataset, args.batch_size,
                              shuffle=False, word_dict=word_dict,
                              tag_dict=tag_dict, seed=args.seed, sort=True)
    eval_dataset = utils.get_dev_examples(args.data_dir)
    eval_loader = DataLoader(eval_dataset, args.batch_size,
                             shuffle=False, word_dict=word_dict,
                             tag_dict=tag_dict, seed=args.seed, sort=False)

    if args.arch == 'lstm':
        model = LSTM(vocab_size=len(word_dict), embedding_size=args.embedding_size,
                     hidden_size=args.hidden_size, tag_dict=tag_dict).to(args.device)
    elif args.arch == 'transformer':
        model = Transformer(vocab_size=len(word_dict), tag_dict=tag_dict, num_blocks=args.num_blocks,
                            model_dim=args.model_dim, num_heads=args.num_heads,
                            feedforward_dim=args.feedforward_dim).to(args.device)
    else:
        raise ValueError('Not supported architecture!')

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(parameters, lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

    for epoch in range(args.epochs):
        # scheduler.step()
        # lr = scheduler.get_lr()[0]
        # logging.info('Epoch %d lr %e', epoch, lr)

        # training
        train_obj = train(model, train_loader, optimizer, epoch)

        scheduler.step()

        # validation
        eval_info = evaluate(model, eval_loader, epoch)

        utils.save(model, os.path.join(args.save, 'model.pt'))


def train(model, train_loader, optimizer, epoch):
    objs = metrics.AverageMeter()
    model.train()
    for step, batch in enumerate(train_loader):
        input_ids, input_mask, input_tags, input_lens = batch
        input_ids = input_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        input_tags = input_tags.to(args.device)
        features, loss = model.forward_loss(input_ids, input_mask, input_lens, input_tags)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        objs.update(loss.item(), n=input_ids.size(0))

        if step % args.print_freq == 0 or step == len(train_loader) - 1:
            logging.info(
                "Train: [{:2d}/{}] Step {:03d}/{:03d} Loss {objs.avg:.3f}".format(
                    epoch + 1, args.epochs, step, len(train_loader) - 1, objs=objs))

    logging.info(
        "Train: [{:2d}/{}] Final Loss {:.3f}".format(epoch + 1, args.epochs, objs.avg))

    return objs.avg


def evaluate(model, eval_loader, epoch):
    # objs = metrics.AverageMeter()
    evaluator = metrics.SeqEntityScore(tag_dict)
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            input_ids, input_mask, input_tags, input_lens = batch
            input_ids = input_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            input_tags = input_tags.to(args.device)
            scores, tags = model(input_ids, input_mask)
            input_tags = input_tags.cpu().numpy()
            target = [input_[:len_] for input_, len_ in zip(input_tags, input_lens)]
            evaluator.update(pred_paths=tags, label_paths=target)

    eval_info, class_info = evaluator.result()
    logging.info("Valid: [{:2d}/{}] Final Accuracy {:.3f} Recall {:.3f} F1 {:.3f}".format(epoch + 1, args.epochs,
                                                                                          eval_info['acc'],
                                                                                          eval_info['recall'],
                                                                                          eval_info['f1']))
    logging.info("Valid Score For Each Entity Category: ")
    for key, value in class_info.items():
        logging.info("Category: {} Accuracy {:.3f} Recall {:.3f} F1 {:.3f}".format(key, value['acc'],
                                                                                   value['recall'], value['f1']))
    logging.info("\n")
    return eval_info


if __name__ == '__main__':
    main()
