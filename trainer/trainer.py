import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop
from model.model import sim_matrix
import ipdb
from torch import autograd

class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

    def _eval_metrics(self, output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output)
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        res = self.data_loader.dataset.__getitem__(1)
        for batch_idx, (minibatch, id) in enumerate(self.data_loader):
            for expert, subdict in minibatch.items():
                for key, val in subdict.items():
                    minibatch[expert][key] = val.to(self.device)

            self.optimizer.zero_grad()
            with autograd.detect_anomaly():
                output = self.model(minibatch)
                loss = self.loss(output)
                loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            # total_metrics += self._eval_metrics(output.cpu().detach().numpy())

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True)) TODO: add clip - sent prediction?

            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist()
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def log_metrics(self, metric_store, metric_name, mode):
        if self.tensorboard:
            print(f"logging metrics: {metric_name}")
            self.writer.set_step(step=self.seen[mode], mode=mode)
            for key, value in metric_store.items():
                self.writer.add_scalar(f"{metric_name}/{key}", value)

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        label_embeddings = []
        content_embeddings = []
        moe_weights = []
        imdbids = []
        with torch.no_grad():
            for batch_idx, (minibatch, id) in enumerate(self.valid_data_loader):
                for expert, subdict in minibatch.items():
                    for key, val in subdict.items():
                        minibatch[expert][key] = val.to(self.device)
                imdbids += id['imdbid']
                self.optimizer.zero_grad()
                output, res, target, moe = self.model(minibatch, evaluation=True, debug=True)
                label_embeddings.append(target)
                content_embeddings.append(res)
                moe_weights.append(moe)
                loss = self.loss(output)
                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        label_embeddings = torch.cat(label_embeddings, dim=0).detach().cpu()
        content_embeddings = torch.cat(content_embeddings, dim=0).detach().cpu()
        moe_weights = torch.cat(moe_weights, dim=0).detach().cpu()
        sims = sim_matrix(label_embeddings, content_embeddings, weights=moe_weights).numpy()
        nested_metrics = {}

        if self.config['retrieval'] == 'intra':
            nested_metrics = intra_movie_metrics(sims, imdbids, self.metrics)
        else:
            for metric in self.metrics:
                metric_name = metric.__name__
                res = metric(sims)  # query_masks=meta["query_masks"]) # TODO: Query mask
                if metric_name == "mean_average_precision":
                    print(f"Epoch: {epoch}, mean AP: {res['mAP']}")
                else:
                    verbose(epoch=epoch, metrics=res, name='MovieClips', mode=metric_name) # TODO: refactor dataset name
                self.log_metrics(res, metric_name=metric_name, mode="val")
                nested_metrics[metric_name] = res

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'nested_val_metrics': nested_metrics
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

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
        assert len(target_idx) > 0 # sanity check
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

        #r1_n = np.array(r1) / np.array(n_clips)
        #medr_n = np.array(medr) / np.array(n_clips)
        #meanr_n = np.array(meanr) / np.array(n_clips)
        r1, medr, meanr = np.mean(r1), np.mean(medr), np.mean(meanr)
        nested_metrics[metric_name] = {'R1': r1, 'MedR': medr, 'MeanR': meanr}

    return nested_metrics
