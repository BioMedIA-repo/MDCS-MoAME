import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sksurv.metrics import concordance_index_censored
import pandas as pd
import numpy as np

import torch.optim
import torch.nn.parallel

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        # tensorboard
        if args.log_data:
            from tensorboardX import SummaryWriter
            writer_dir = os.path.join(results_dir, 'fold_' + str(fold))
            if not os.path.isdir(writer_dir):
                os.mkdir(writer_dir)
            self.writer = SummaryWriter(writer_dir, flush_secs=15)
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None
        self.filename_result =None
        self.filename_pt = None
        #loss
        self.cosine = nn.CosineEmbeddingLoss()
        self.MSE = MSE()

    def learning(self, model, train_loader, val_loader, criterion, optimizer, scheduler,device):
        if torch.cuda.is_available():
            model = model.cuda()
        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        if self.args.evaluate:
            # self.validate(val_loader, model, criterion, device)
            self.epoch = 1
            c_index, all_risk_scores, all_event_times, all_censorships = self.validate(val_loader, model, criterion, device)
            print(c_index)
            return

        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # train for one epoch
            _,all_risk_scores_t,all_event_times_t,all_censorships_t = self.train(train_loader, model, criterion, optimizer, device)
            # evaluate on validation set
            c_index,all_risk_scores,all_event_times,all_censorships = self.validate(val_loader, model, criterion, device)
            # remember best c-index and save checkpoint
            is_best = c_index > self.best_score
            if is_best:
                self.best_score = c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'best_score': self.best_score})
            print(' *** best c-index={:.4f} at epoch {}'.format(self.best_score, self.best_epoch))
            if scheduler is not None:
                scheduler.step()
            print('>')
        return self.best_score, self.best_epoch

    def train(self, data_loader, model, criterion, optimizer, device):
        model.train()
        train_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))
        dataloader = tqdm(data_loader, desc='Train Epoch: {}'.format(self.epoch))
        # for batch_idx, (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(dataloader):
        for batch_idx, (data_WSI1, data_WSI2, data_WSI3, data_WSI4, data_WSI5, data_WSI6, data_WSI7, data_WSI8, data_WSI9, data_WSI10, coord1, coord2, coord3, coord4, coord5, index_num, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI1 = data_WSI1.cuda()
                data_WSI2 = data_WSI2.cuda()
                data_WSI3 = data_WSI3.cuda()
                data_WSI4 = data_WSI4.cuda()
                data_WSI5 = data_WSI5.cuda()
                data_WSI6 = data_WSI6.cuda()
                data_WSI7 = data_WSI7.cuda()
                data_WSI8 = data_WSI8.cuda()
                data_WSI9 = data_WSI9.cuda()
                data_WSI10 = data_WSI10.cuda()
                coord1 = coord1.cuda()
                coord2 = coord2.cuda()
                coord3 = coord3.cuda()
                coord4 = coord4.cuda()
                coord5 = coord5.cuda()
                # index_num = index_num.cuda()
                # data_WSI_vec = data_WSI_vec.cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()
            if self.args.model == "MDCS_MoAME":
                hazards, S, I_p, I_p_g, I_r, I_r_g, G_g, G_g_p, G_g_r = model(x_path1=data_WSI1, x_path2=data_WSI2, x_path3=data_WSI3, x_path4=data_WSI4, x_path5=data_WSI5, x_path6=data_WSI6, x_path7=data_WSI7, x_path8=data_WSI8, x_path9=data_WSI9, x_path10=data_WSI10, coord1=coord1, coord2=coord2, coord3=coord3, coord4=coord4, coord5=coord5, index_num=index_num, x_omic1=data_omic1, x_omic2=data_omic2,x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5,x_omic6=data_omic6)

            # survival loss + sim loss + sim loss
            sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)

            if self.args.model == "MDCS_MoAME":
                sim_loss_r = criterion[1]( I_r_g, I_r.detach())
                sim_loss_p = criterion[1](I_p_g, I_p.detach())
                sim_loss_g = criterion[1]((G_g_p + G_g_r) / 2, G_g.detach())
                sim_loss_intra = -criterion[1](I_p, I_r)
                loss = sur_loss + self.args.alpha * (sim_loss_r + sim_loss_p + sim_loss_g) + self.args.beta * sim_loss_intra
            else:
                loss = sur_loss

            risk = -torch.sum(S, dim=1).detach().cpu().numpy()

            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.item()
            all_event_times[batch_idx] = event_time

            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # calculate loss and error for epoch
        train_loss /= len(dataloader)
        # print(all_risk_scores)

        c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]

        if self.writer:
            self.writer.add_scalar('train/loss', train_loss, self.epoch)
            self.writer.add_scalar('train/c_index', c_index, self.epoch)

        return c_index, all_risk_scores, all_event_times, all_censorships

    def validate(self, data_loader, model, criterion, device):
        model.eval()
        val_loss = 0.0
        all_risk_scores = np.zeros((len(data_loader)))
        all_censorships = np.zeros((len(data_loader)))
        all_event_times = np.zeros((len(data_loader)))

        dataloader = tqdm(data_loader, desc='Test Epoch: {}'.format(self.epoch))
        for batch_idx, (data_WSI1, data_WSI2, data_WSI3, data_WSI4, data_WSI5, data_WSI6, data_WSI7, data_WSI8, data_WSI9, data_WSI10,  coord1, coord2, coord3, coord4, coord5, index_num, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c) in enumerate(dataloader):
            if torch.cuda.is_available():
                data_WSI1 = data_WSI1.cuda()
                # data_WSI_vec = data_WSI_vec.cuda()
                data_WSI2 = data_WSI2.cuda()
                data_WSI3 = data_WSI3.cuda()
                data_WSI4 = data_WSI4.cuda()
                data_WSI5 = data_WSI5.cuda()
                data_WSI6 = data_WSI6.cuda()
                data_WSI7 = data_WSI7.cuda()
                data_WSI8 = data_WSI8.cuda()
                data_WSI9 = data_WSI9.cuda()
                data_WSI10 = data_WSI10.cuda()
                coord1 = coord1.cuda()
                coord2 = coord2.cuda()
                coord3 = coord3.cuda()
                coord4 = coord4.cuda()
                coord5 = coord5.cuda()
                # index_num = index_num.cuda()
                data_omic1 = data_omic1.type(torch.FloatTensor).cuda()
                data_omic2 = data_omic2.type(torch.FloatTensor).cuda()
                data_omic3 = data_omic3.type(torch.FloatTensor).cuda()
                data_omic4 = data_omic4.type(torch.FloatTensor).cuda()
                data_omic5 = data_omic5.type(torch.FloatTensor).cuda()
                data_omic6 = data_omic6.type(torch.FloatTensor).cuda()
                label = label.type(torch.LongTensor).cuda()
                c = c.type(torch.FloatTensor).cuda()

            with torch.no_grad():
                if self.args.model == "MDCS_MoAME":
                    hazards, S, I_p, I_p_g, I_r, I_r_g, G_g, G_g_p, G_g_r = model(x_path1=data_WSI1, x_path2=data_WSI2, x_path3=data_WSI3,
                                                           x_path4=data_WSI4, x_path5=data_WSI5, x_path6=data_WSI6,
                                                           x_path7=data_WSI7, x_path8=data_WSI8, x_path9=data_WSI9,
                                                           x_path10=data_WSI10, coord1=coord1, coord2=coord2,
                                                           coord3=coord3, coord4=coord4, coord5=coord5, index_num=index_num,
                                                           x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3,
                                                           x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)

            # survival loss + sim loss + sim loss
            sur_loss = criterion[0](hazards=hazards, S=S, Y=label, c=c)

            if self.args.model == "MDCS_MoAME":
                sim_loss_r = criterion[1]( I_r_g, I_r.detach())
                sim_loss_p = criterion[1](I_p_g, I_p.detach())
                sim_loss_g = criterion[1]((G_g_p + G_g_r) / 2, G_g.detach())
                sim_loss_intra = -criterion[1](I_p, I_r)
                loss = sur_loss + self.args.alpha * (sim_loss_r + sim_loss_p + sim_loss_g) + self.args.beta * sim_loss_intra
            
            else:
                loss = sur_loss

            risk = -torch.sum(S, dim=1).cpu().numpy()
            all_risk_scores[batch_idx] = risk
            all_censorships[batch_idx] = c.cpu().numpy()
            all_event_times[batch_idx] =  event_time


            val_loss += loss.item()

        val_loss /= len(dataloader)
        c_index = concordance_index_censored((1-all_censorships).astype(bool),
                                             all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        print('loss: {:.4f}, c_index: {:.4f}'.format(val_loss, c_index))
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, self.epoch)
            self.writer.add_scalar('val/c-index', c_index, self.epoch)
        return c_index, all_risk_scores, all_event_times, all_censorships

    def save_checkpoint(self, state):
        if self.filename_best is not None:
            os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          'fold_' + str(self.fold),
                                          'model_best_{score:.4f}_{epoch}.pth.tar'.format(score=state['best_score'], epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
