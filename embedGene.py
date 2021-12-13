#!/usr/bin/env python3
# The original source code of Poincare Embedding can be found in  https://github.com/facebookresearch/poincare-embeddings
# This source code is partially modified for the application to HiG2Vec.

import torch as th
import numpy as np
import logging
import argparse
from hype.sn import Embedding, initialize
from hype import trainGene
from hype.graph import load_edge_list, eval_reconstruction
from hype.checkpoint import LocalCheckpoint
from hype.rsgd import RiemannianSGD
from hype.lorentz import LorentzManifold
from hype.euclidean import EuclideanManifold
from hype.poincare import PoincareManifold
import sys
import json
import torch.multiprocessing as mp
import shutil
import time

th.manual_seed(42)
np.random.seed(42)


MANIFOLDS = {
    'lorentz': LorentzManifold,
    'euclidean': EuclideanManifold,
    'poincare': PoincareManifold
}


def async_eval(adj, q, logQ, opt):
    manifold = MANIFOLDS[opt.manifold]()
    while True:
        temp = q.get()
        if temp is None:
            return

        if not q.empty():
            continue

        epoch, elapsed, loss, pth = temp
        chkpnt = th.load(pth, map_location='cpu')
        lt = chkpnt['embeddings']

        sqnorms = manifold.pnorm(lt)
        # if manifold.pnorm() doesn't work,
        # sqnorms = np.sqrt(np.sum(lt.numpy()*lt.numpy(),axis=1))

        lmsg = {
            'epoch': epoch,
            'elapsed': elapsed,
            'loss': loss,
            'sqnorm_min': sqnorms.min().item(),
            'sqnorm_avg': sqnorms.mean().item(),
            'sqnorm_max': sqnorms.max().item()
        }
        logQ.put((lmsg, pth))

# Adapated from:
# https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/
class Unsettable(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(Unsettable, self).__init__(option_strings, dest, nargs='?', **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        val = None if option_string.startswith('-no') else values
        setattr(namespace, self.dest, val)

def pretrain(model, data, pre_objects, pre_embeddings):
    trained_ix = []
    untrained_ix = []    
    for i, obj in enumerate(data.objects):
        if obj in pre_objects:
            j = pre_objects.index(obj)
            model.lt.state_dict()['weight'][i] = th.Tensor(pre_embeddings[j])
            trained_ix.append(i)
        else:
            untrained_ix.append(i)
    return model, trained_ix, untrained_ix

def main():
    parser = argparse.ArgumentParser(description='Train Hyperbolic Embeddings')
    parser.add_argument('-checkpoint', default='/result/HiG2Vec.pth',
                        help='Where to store the model checkpoint')
    parser.add_argument('-dset', type=str, required=True,
                        help='Dataset identifier')
    parser.add_argument('-dim', type=int, default=1000,
                        help='Embedding dimension')
    parser.add_argument('-manifold', type=str, default='poincare',
                        choices=MANIFOLDS.keys(), help='Embedding manifold')
    parser.add_argument('-lr', type=float, default=0.3,
                        help='Learning rate')
    parser.add_argument('-epochs', type=int, default=1000,
                        help='Number of epochs')
    parser.add_argument('-batchsize', type=int, default=50,
                        help='Batchsize')
    parser.add_argument('-negs', type=int, default=50,
                        help='Number of negatives')
    parser.add_argument('-burnin', type=int, default=20,
                        help='Epochs of burn in')
    parser.add_argument('-dampening', type=float, default=0.75,
                        help='Sample dampening during burnin')
    parser.add_argument('-ndproc', type=int, default=8,
                        help='Number of data loading processes')
    parser.add_argument('-eval_each', type=int, default=1,
                        help='Run evaluation every n-th epoch')
    parser.add_argument('-fresh', action='store_true', default=False,
                        help='Override checkpoint')
    parser.add_argument('-debug', action='store_true', default=False,
                        help='Print debuggin output')
    parser.add_argument('-gpu', default=0, type=int,
                        help='Which GPU to run on (-1 for no gpu)')
    parser.add_argument('-sym', action='store_true', default=False,
                        help='Symmetrize dataset')
    parser.add_argument('-maxnorm', '-no-maxnorm', default='1',
                        action=Unsettable, type=int)
    parser.add_argument('-sparse', default=False, action='store_true',
                        help='Use sparse gradients for embedding table')
    parser.add_argument('-burnin_multiplier', default=0.01, type=float)
    parser.add_argument('-neg_multiplier', default=1.0, type=float)
    parser.add_argument('-quiet', action='store_true', default=False)
    parser.add_argument('-lr_type', choices=['scale', 'constant'], default='constant')
    parser.add_argument('-train_threads', type=int, default=1,
                        help='Number of threads to use in training when using CPU')
    parser.add_argument('-pretrain', help='Pretrained embedding', type=str)
    parser.add_argument('-finetune', default=False, action='store_true', help='Finetune the pretrained parameters')
    opt = parser.parse_args()

    # setup debugging and logigng
    log_level = logging.DEBUG if opt.debug else logging.INFO
    log = logging.getLogger('lorentz')
    logging.basicConfig(level=log_level, format='%(message)s', stream=sys.stdout)

    # set default tensor type
    th.set_default_tensor_type('torch.DoubleTensor')
    
    # set device
    device = th.device(f'cuda:{opt.gpu}' if th.cuda.is_available() and opt.gpu >=0 else 'cpu')

    # select manifold to optimize on
    manifold = MANIFOLDS[opt.manifold](debug=opt.debug, max_norm=opt.maxnorm)
    opt.dim = manifold.dim(opt.dim)

    if 'tsv' in opt.dset:
        log.info('Using edge list dataloader')
        idx, objects, weights = load_edge_list(opt.dset, opt.sym)
        model, data, model_name, conf = initialize(
            manifold, opt, idx, objects, weights, sparse=opt.sparse
        )
    else:
        print("Not adaptive format")

    # initialitze with pretrain model
    if opt.pretrain != None:
        pre_model = th.load(opt.pretrain, map_location="cpu")
        pre_objects = pre_model['objects']
        pre_embeddings = pre_model['embeddings']
        model, trained_ix, untrained_ix = pretrain(model, data, pre_objects, pre_embeddings)
    else:
        trained_ix = []

    # set burnin parameters
    data.neg_multiplier = opt.neg_multiplier
    trainGene._lr_multiplier = opt.burnin_multiplier

    # Build config string for log
    log.info(f'json_conf: {json.dumps(vars(opt))}')

    if opt.lr_type == 'scale':
        opt.lr = opt.lr * opt.batchsize

    # setup optimizer
    optimizer = RiemannianSGD(model.optim_params(manifold), lr=opt.lr)

    # setup checkpoint
    checkpoint = LocalCheckpoint(
        opt.checkpoint,
        include_in_all={'conf' : vars(opt), 'objects' : objects},
        start_fresh=opt.fresh
    )

    # get state from checkpoint
    state = checkpoint.initialize({'epoch': 0, 'model': model.state_dict()})
    model.load_state_dict(state['model'])
    opt.epoch_start = state['epoch']

    adj = {}
    for inputs, _ in data:
        for row in inputs:
            x = row[0].item()
            y = row[1].item()
            if x in adj:
                adj[x].add(y)
            else:
                adj[x] = {y}

    controlQ, logQ = mp.Queue(), mp.Queue()
    control_thread = mp.Process(target=async_eval, args=(adj, controlQ, logQ, opt))
    control_thread.start()

    # control closure
    def control(model, epoch, elapsed, loss):
        """
        Control thread to evaluate embedding
        """
        lt = model.w_avg if hasattr(model, 'w_avg') else model.lt.weight.data
        manifold.normalize(lt)

        checkpoint.path = f'{opt.checkpoint}.{epoch}'
        checkpoint.save({
            'model': model.state_dict(),
            'embeddings': lt,
            'epoch': epoch,
            'manifold': opt.manifold,
        })

        controlQ.put((epoch, elapsed, loss, checkpoint.path))

        while not logQ.empty():
            lmsg, pth = logQ.get()
            shutil.move(pth, opt.checkpoint)
            log.info(f'json_stats: {json.dumps(lmsg)}')
    
    control.checkpoint = True
    model = model.to(device)
    if hasattr(model, 'w_avg'):
        model.w_avg = model.w_avg.to(device)
    if opt.train_threads > 1:
        threads = []
        model = model.share_memory()
        args = (device, model, data, optimizer, opt, log)
        kwargs = {'ctrl': control, 'progress' : not opt.quiet}
        for i in range(opt.train_threads):
            kwargs['rank'] = i
            threads.append(mp.Process(target=trainGene.train, args=args, kwargs=kwargs))
            threads[-1].start()
        [t.join() for t in threads]
    else:        
        trainGene.train(device, model, data, optimizer, trained_ix, opt, log, ctrl=control,
                progress=not opt.quiet)
    
    controlQ.put(None)
    control_thread.join()
    while not logQ.empty():
        lmsg, pth = logQ.get()
        shutil.move(pth, opt.checkpoint)
        log.info(f'json_stats: {json.dumps(lmsg)}')
                
if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
