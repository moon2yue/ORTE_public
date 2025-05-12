import torch
import torch.nn.functional as F
import numpy as np

import argparse, os
import sys

from txai.utils.predictors.loss_smoother_stats import exp_criterion_eval_smoothers
from txai.utils.predictors.eval import eval_mv4
from txai.utils.cl import in_batch_triplet_sampling
from txai.models.run_model_utils import batch_forwards, batch_forwards_TransformerMVTS
from txai.utils.cl import basic_negative_sampling

from txai.utils.functional import js_divergence

default_scheduler_args = {
    'mode': 'max', 
    'factor': 0.1, 
    'patience': 5,
    'threshold': 0.00001, 
    'threshold_mode': 'rel',
    'cooldown': 0, 
    'min_lr': 1e-8, 
    'eps': 1e-08, 
    'verbose': True
}


def kl_divergence_gaussian(mean1, logvar1, mean2, logvar2):  
    kl_div = 0.5 * torch.mean((logvar2 - logvar1 - 1 + (logvar1.exp() + (mean1 - mean2).pow(2)) / logvar2.exp()))  
    return kl_div  

def contrast_loss(target, positive, negative, temperature_1, temperature_2, use_cosine_similarity=False):
    batch_size = target.shape[0]

    if use_cosine_similarity:
        consine_sim = torch.nn.CosineSimilarity(dim=1)
        positive_sim = consine_sim(target.reshape(batch_size,-1), positive.reshape(batch_size,-1))
        negative_sim = consine_sim(target.reshape(batch_size,-1), negative.reshape(batch_size,-1))
    else:
        positive_sim = torch.mul(target.reshape(batch_size,-1), positive.reshape(batch_size,-1)).mean(dim=1)
        negative_sim = torch.mul(target.reshape(batch_size,-1), negative.reshape(batch_size,-1)).mean(dim=1)

    positive_sim = torch.exp(positive_sim/temperature_1)
    negative_sim = torch.exp(negative_sim/temperature_2)

    loss = -torch.log(positive_sim / (positive_sim + negative_sim))
    return loss.mean()

def train_mv6_consistency(
        model,
        optimizer,
        train_loader,
        val_tuple,
        num_epochs,
        # Criterions:
        clf_criterion,
        sim_criterion,
        beta_exp,
        beta_sim,
        train_tuple,
        lam_label = 1.0,
        clip_norm = True,
        use_scheduler = False,
        wait_for_scheduler = 20,
        scheduler_args = default_scheduler_args,
        selection_criterion = None,
        save_path = None,
        early_stopping = True,
        label_matching = False,
        embedding_matching = True,
        opt_pred_mask = False, # If true, optimizes based on clf_criterion
        opt_pred_mask_to_full_pred = False,
        batch_forward_size = None,
        simclr_training = False,
        num_negatives_simclr = 64,
        max_batch_size_simclr_negs = None,
        bias_weight = 0.1
    ):
    '''
    Args:
        selection_criterion: function w signature f(out, val_tuple)

        if both label_matching and embedding_matching are true, then sim_criterion must be a list of length 2 
            with [embedding_sim, label_sim] functions

    '''
    # TODO: Add weights and biases logging

    best_epoch = 0
    best_val_metric = -1e9

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)

    dataX, dataT, dataY = train_tuple # Unpack training variables

    for epoch in range(num_epochs):

        if epoch == 8:
            print(epoch)
        model.train()
        cum_sparse, cum_exp_loss, cum_clf_loss, cum_sim_loss, cum_cont_loss = [], [], [], [], []
        label_sim_list, emb_sim_list = [], []
        for X, times, y, ids in train_loader: # Need negative sampling here


            optimizer.zero_grad()

            # if detect_irreg:
            #     src_mask = (X < 1e-7)
            #     out_dict = model(X, times, captum_input = True)

            out_dict = model(X, times, captum_input = True)
            out = out_dict['pred']
            ste_mask = out_dict['ste_mask']
            if out.isnan().sum() > 0:
                # Exits if nan's are found
                print('out', out.isnan().sum())
                exit()

            clf_loss = js_divergence(out_dict['pred_mask'].softmax(dim=-1), out_dict['pred'].softmax(dim=-1))

            org_div, emb_div = out_dict['all_z'] #
            reference_src, exp_src, _ = out_dict['reference_z']

            sim_loss = kl_divergence_gaussian(emb_div[0], emb_div[1], org_div[0].mean(0, keepdim=True), org_div[1].mean(0, keepdim=True)) * bias_weight
            sim_loss_1 = sim_loss

            sim_loss += F.mse_loss(reference_src, exp_src)
            sim_loss_2 = F.mse_loss(reference_src, exp_src)

            if bias_weight==0:
                sim_loss = F.mse_loss(reference_src, exp_src)

            sim_loss = beta_sim * sim_loss

            exp_loss = beta_exp * model.compute_loss(out_dict)
            positive_x_reference, positive_x_predict, negative_x_reference = out_dict['reference_z']
            cont_loss = 10 * contrast_loss(target=X[:,:,:], positive=positive_x_predict[:,:,:].transpose(0, 1),
                                           negative=negative_x_reference[:,:,:].transpose(0, 1), temperature_1=0.5, temperature_2=0.5,
                                           use_cosine_similarity=True)


            sparse_loss = 0.02 * torch.mean(out_dict['ste_mask'].sum(1))

            loss = clf_loss + exp_loss + cont_loss + sparse_loss

            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            loss.backward()
            optimizer.step()

            cum_sparse.append(((ste_mask).sum() / ste_mask.flatten().shape[0]).item())
            cum_clf_loss.append(clf_loss.detach().item())
            cum_exp_loss.append([exp_loss.detach().clone().item()])
            cum_sim_loss.append(sim_loss.detach().item())
            cum_cont_loss.append(cont_loss.detach().item())

        # Print all stats:
        # Convert to np:
        sparse = np.array(cum_sparse) # Should be size (B, M)
        sparse = sparse.mean()
        clf = sum(cum_clf_loss) / len(cum_clf_loss)
        exp = np.array(cum_exp_loss) # Size (B, M, L)
        exp = exp.mean(axis=0).flatten()
        sim = np.mean(cum_sim_loss)
        cont = np.mean(cum_cont_loss)

        sim_s = f'{sim:.4f}'

        print(f'Epoch: {epoch}: Sparsity = {sparse:.4f} \t Exp Loss = {exp} \t Clf Loss = {clf:.4f} \t CL Loss = {sim_s} \t Cont Loss = {cont}')

        # Eval after every epoch
        # Call evaluation function:
        model.eval()
        
        if batch_forward_size is None:
            f1, out = eval_mv4(val_tuple, model)
        else:
            out = batch_forwards(model, val_tuple[0], val_tuple[1], batch_size = 64)
            f1 = 0


        ste_mask = out['ste_mask']
        sparse = ste_mask.mean().item()

        met = -1.0 * loss
        cond = not early_stopping
        if early_stopping:
            cond = (met > best_val_metric)
        if cond:
            best_val_metric = met
            if save_path is not None:
                model.save_state(save_path)
            best_epoch = epoch
            print('Save at epoch {}: Metric={:.4f}'.format(epoch, met))

        if use_scheduler and (epoch > wait_for_scheduler):
            scheduler.step(met)

        if (epoch + 1) % 10 == 0:
            valsparse = '{:.4f}'.format(sparse)
            print(f'Epoch {epoch + 1}, Val F1 = {f1:.4f}, Val Sparsity = {valsparse}')

    print(f'Best Epoch: {best_epoch + 1} \t Val F1 = {best_val_metric:.4f}')