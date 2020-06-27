import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from model.model import sim_matrix
from parse_config import ConfigParser
from utils.visualisation import batch_path_vis
from logger import TensorboardWriter
import ipdb
import numpy as np
import os
import json
import pandas as pd


def main(config):
    res_exp = str(config.resume).replace('model_best.pth', 'test_res.json')

    logger = config.get_logger('test')
    writer = TensorboardWriter(config.log_dir, logger, config['trainer']['tensorboard'])

    # setup data_loader instances
    config._config['data_loader']['args']['split'] = 'test'
    config._config['data_loader']['args']['batch_size'] = 6581
    data_loader = config.initialize('data_loader', module_data)

    experts_used = data_loader.dataset.experts_used
    config._config['arch']['args'][
        'experts_used'] = experts_used  # improve this, how to safely clone args across classes?
    # improve this, how to safely clone args across classes?
    config._config['arch']['args']['label'] = data_loader.dataset.label
    config._config['arch']['args']['expert_dims'] = data_loader.dataset.expert_dims

    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])

    # get assignment function handles
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    if config.resume is not None:
        logger.info('Loading checkpoint: {} ...'.format(config.resume))
        checkpoint = torch.load(config.resume)
        state_dict = checkpoint['state_dict']
        if config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model)
        model.load_state_dict(state_dict)
    else:
        print('Using untrained model...')

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))

    label_embeddings = []
    content_embeddings = []
    moe_weights = []
    imdbids = []
    videoids = []
    with torch.no_grad():
        for batch_idx, (minibatch, id) in enumerate(data_loader):
            for expert, subdict in minibatch.items():
                for key, val in subdict.items():
                    minibatch[expert][key] = val.to(device)
            imdbids += id['imdbid']
            videoids += id['videoid']
            output, res, target, moe = model(minibatch, evaluation=True)
            label_embeddings.append(target)
            content_embeddings.append(res)
            moe_weights.append(moe)
            # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

    label_embeddings = torch.cat(label_embeddings, dim=0).detach().cpu()
    content_embeddings = torch.cat(content_embeddings, dim=0).detach().cpu()
    moe_weights = torch.cat(moe_weights, dim=0).detach().cpu()
    sims = sim_matrix(label_embeddings, content_embeddings, weights=moe_weights).numpy()


    all_res = {'inter': {}}
    print('#### INTER-MOVIE ####')
    for metric in metrics:
        metric_name = metric.__name__
        res = metric(sims)  # query_masks=meta["query_masks"]) # TODO: Query mask
        verbose(epoch=0, metrics=res, name='MovieClips', mode=metric_name)  # TODO: refactor dataset name
        all_res['inter'][metric_name] = res

    # TODO: Print intra/inter metrics depending on training regime
    #print('\n#### INTRA-MOVIE ####')
    #all_res['intra'] = intra_movie_metrics(sims, imdbids, metrics)

    # n_samples = len(data_loader.sampler)
    # log = {'loss': total_loss / n_samples}
    # log.update({
    #    met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    # logger.info(log)
    # logger.info(log)
    with open(res_exp, 'w') as fid:
        json.dump(all_res, fid)

    all_res['n_params'] = model.tot_params()

    save_results = True
    if save_results:
        results = sims2ids(sims, videoids)
        results_fp = res_exp.replace('test_res.json', 'results.csv')
        results.to_csv(results_fp)


    return all_res


def verbose(epoch, metrics, mode, name="TEST"):
    r1, r5, r10, r50 = metrics["R1"], metrics["R5"], metrics["R10"], metrics["R50"]
    msg = f"[{mode}]{name:s} epoch {epoch}, R@1: {r1:.1f}"
    msg += f", R@5: {r5:.1f}, R@10 {r10:.1f}, R@50 {r50:.1f}"
    msg += f"MedR: {metrics['MedR']:g}, MeanR: {metrics['MeanR']:.1f}"
    print(msg)


def intra_movie_metrics(sims, imdbids, metrics):
    unique_ids = set(imdbids)
    imdbids = np.array(imdbids)

    sim_stack = []
    for id in unique_ids:
        target_idx = np.where(imdbids == id)[0]
        assert len(target_idx) > 0  # sanity check
        sim_stack.append(sims[target_idx][:, target_idx])

    nested_metrics = {}
    for metric in metrics:
        metric_name = metric.__name__
        r1 = []
        medr = []
        meanr = []
        n_clips = []
        for sim in sim_stack:
            res = metric(sim)
            r1.append(res['R1'])
            medr.append(res['MedR'])
            meanr.append(res['MeanR'])
            n_clips.append(sim.shape[0])

        # r1_n = np.array(r1) / np.array(n_clips)
        # medr_n = np.array(medr) / np.array(n_clips)
        # meanr_n = np.array(meanr) / np.array(n_clips)
        r1, medr, meanr = np.mean(r1), np.mean(medr), np.mean(meanr)
        res_dict = {'R1': r1, 'MedR': medr, 'MeanR': meanr}
        nested_metrics[metric_name] = res_dict
        print(metric_name, ':    ', str(res_dict))
    return nested_metrics

def sims2ids(sims, videoids):
    sims = torch.from_numpy(sims)
    values, indices = torch.topk(sims, 5, dim=-1)
    preds = {}
    for videoid, inds in zip(videoids, indices):
        pred = []
        for ind in inds:
            pred.append(videoids[ind])
        preds[videoid] = pred

    data = pd.DataFrame.from_dict(preds, orient='index')
    data['R1'] = (data[0] == data.index)
    data['2corr'] = (data[1] == data.index)
    data['3corr'] = (data[2] == data.index)
    data['4corr'] = (data[3] == data.index)
    data['5corr'] = (data[4] == data.index)
    data['R5'] = data[['R1', '2corr', '3corr', '4corr', '5corr']].any(axis=1)

    del data['2corr']
    del data['3corr']
    del data['4corr']
    del data['5corr']

    return data

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser(args)
    main(config)
