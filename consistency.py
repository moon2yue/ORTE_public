import torch
import torch.nn.functional as F
import numpy as np

import argparse, os
import sys
# Add the relative path to '../../' so that 'txai' can be found
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/data/Yuejinghang/TimeXplusplus/txai')))
# sys.path.append('/data/Yuejinghang/TimeXplusplus')

from txai.utils.predictors.loss_smoother_stats import exp_criterion_eval_smoothers
from txai.utils.predictors.eval import eval_mv4, eval_mv4_loader
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

from sklearn import metrics
def normalize_exp(exps):
    norm_exps = torch.empty_like(exps)
    for i in range(exps.shape[1]):
        norm_exps[:,i,:] = (exps[:,i,:] - exps[:,i,:].min()) / (exps[:,i,:].max() - exps[:,i,:].min() + 1e-9)
    return norm_exps


def print_results(mask_labelss, true_labelss):
    mask_labelss = normalize_exp(mask_labelss)

    if torch.is_tensor(mask_labelss):
        mask_labelss = mask_labelss.cpu().numpy()
    if torch.is_tensor(true_labelss):
        true_labelss = true_labelss.cpu().numpy()

    all_aupoc, all_auprc, all_aup, all_aur = [], [], [], []

    for i in range(mask_labelss.shape[1]):
        mask_label = mask_labelss[:, i, :]
        true_label = true_labelss[:, i, :]

        mask_prec, mask_rec, mask_thres = metrics.precision_recall_curve(
            true_label.flatten().astype(int), mask_label.flatten())
        try:
            AUROC = metrics.roc_auc_score(true_label.flatten(), mask_label.flatten())
        except ValueError:
            pass
        all_aupoc.append(AUROC)
        AUPRC = metrics.auc(mask_rec, mask_prec)
        all_auprc.append(AUPRC)
        AUP = metrics.auc(mask_thres, mask_prec[:-1])
        all_aup.append(AUP)
        AUR = metrics.auc(mask_thres, mask_rec[:-1])
        all_aur.append(AUR)
    print('Saliency AUROC: = {:.4f} +- {:.4f}'.format(np.mean(all_aupoc), np.std(all_aupoc) / np.sqrt(len(all_aupoc))))
    print('Saliency AUPRC: = {:.4f} +- {:.4f}'.format(np.mean(all_auprc), np.std(all_auprc) / np.sqrt(len(all_auprc))))
    print('Saliency AUP: = {:.4f} +- {:.4f}'.format(np.mean(all_aup), np.std(all_aup) / np.sqrt(len(all_aup))))
    print('Saliency AUR: = {:.4f} +- {:.4f}'.format(np.mean(all_aur), np.std(all_aur) / np.sqrt(len(all_aur))))

    resutlt_cur = {"AUROC": all_aupoc,
                   "AUPRC": all_auprc,
                   "AUP": all_aup,
                   "AUR": all_aur}
    return resutlt_cur




def kl_divergence_gaussian(mean1, logvar1, mean2, logvar2):
    kl_div = 0.5 * torch.mean((logvar2 - logvar1 - 1 + (logvar1.exp() + (mean1 - mean2).pow(2)) / logvar2.exp()))
    return kl_div


def contrast_loss(target, positive, negative, temperature=0.5, use_cosine_similarity=False):
    if use_cosine_similarity:
        consine_sim = torch.nn.CosineSimilarity(dim=1)
        positive_sim = consine_sim(target, positive)
        negative_sim = consine_sim(target, negative)
    else:
        positive_sim = torch.mul(target, positive).mean(dim=1)
        negative_sim = torch.mul(target, negative).mean(dim=1)

    positive_sim = torch.exp(positive_sim / temperature)
    negative_sim = torch.exp(negative_sim / temperature)

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
        lam_label=1.0,
        clip_norm=True,
        use_scheduler=False,
        wait_for_scheduler=20,
        scheduler_args=default_scheduler_args,
        selection_criterion=None,
        save_path=None,
        early_stopping=True,
        label_matching=False,
        embedding_matching=True,
        opt_pred_mask=False,  # If true, optimizes based on clf_criterion
        opt_pred_mask_to_full_pred=False,
        batch_forward_size=None,
        simclr_training=False,
        num_negatives_simclr=64,
        max_batch_size_simclr_negs=None,
        bias_weight=0.1,

        cont_layer=1,
        cont_loss_weight_first=10,
        cont_loss_weight_last=10,
        sparse_loss_weight=10,
        sim_loss_weight=1,
        kl_loss_weight=1,
        use_decoupled_ste=True,
        forward_ste_temperature=0.5,
        backward_ste_temperature=0.5,
        ste_learning_rate=0.01,
        test=None,
        gt_exps=None,
        # scheduler=None,
):
    '''
    Args:
        selection_criterion: function w signature f(out, val_tuple)

        if both label_matching and embedding_matching are true, then sim_criterion must be a list of length 2
            with [embedding_sim, label_sim] functions

    '''
    # TODO: Add weights and biases logging
    # print(cont_loss_weight)
    # print(sparse_loss_weight)
    # print(exp_loss_weight)
    # print(clf_loss_weight)

    best_epoch = 0
    if test != None:
        best_val_metric = 0.5#-1e9
    else:
        best_val_metric = -1e9

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_args)

    dataX, dataT, dataY = train_tuple  # Unpack training variables
    i=0
    for epoch in range(num_epochs):
        i = i + 1
        if i == 5:
            print(i)
        model.train()
        cum_sparse, cum_exp_loss, cum_clf_loss, cum_sim_loss, cum_cont_loss, cum_sparse_loss,cum_sim_loss,cum_kl_loss = [], [], [], [], [], [], [], []

        label_sim_list, emb_sim_list = [], []
        for step, (X, times, y, ids) in enumerate(train_loader):  # Need negative sampling here
            # i=i+1
            # if i==31:
            #     print(i)
            optimizer.zero_grad()

            # if detect_irreg:
            #     src_mask = (X < 1e-7)
            #     out_dict = model(X, times, captum_input = True)

            out_dict = model(X, times, captum_input=True)
            out = out_dict['pred']
            ste_mask = out_dict['ste_mask']
            if out.isnan().sum() > 0:
                # Exits if nan's are found
                print('out', out.isnan().sum())
                exit()
            ## Loss-LC——原模型预测和代理模型预测的KL散度损失函数
            clf_loss = js_divergence(out_dict['pred_mask'].softmax(dim=-1), out_dict['pred'].softmax(dim=-1))

            # org_div, emb_div = out_dict['all_z']  #
            # reference_src, exp_src, _ = out_dict['reference_z']

            # ## Loss-KL——(X, X~)KL散度损失函数
            # sim_loss = kl_divergence_gaussian(emb_div[0], emb_div[1], org_div[0].mean(0, keepdim=True), org_div[1].mean(0, keepdim=True)) * bias_weight
            kl_loss = kl_loss_weight*kl_divergence_gaussian(out_dict['all_z'][1][0], out_dict['all_z'][1][1], out_dict['all_z'][0][0].mean(0, keepdim=True),
                                              out_dict['all_z'][0][1].mean(0, keepdim=True))

            # ## Loss-dr——(X~^r, X~)的距离损失函数
            # sim_loss += F.mse_loss(reference_src, exp_src)## 这里加上了上一个KL散度损失函数

            # if bias_weight==0:
            #     sim_loss = F.mse_loss(reference_src, exp_src)

            # sim_loss = beta_sim * sim_loss

            ## Loss-M——STE分布拟合 + con 损失函数(beta_exp就是文中的alpha)
            exp_loss = beta_exp * model.compute_loss(out_dict)
            # conc_emb_loss = conc_embeddings.abs().mean()

            ## ①将原始输入作为锚样本，x点乘M作为正样本，x点乘（1-M）作为负样本，计算对比损失函数
            if cont_layer == 1:
                # positive_x_reference, positive_x_predict, negative_x_reference = out_dict['reference_z']
                cont_loss = cont_loss_weight_first*contrast_loss(target=X[:, :, :],
                                          positive=out_dict['reference_z'][1][:, :, :].transpose(0, 1),
                                          negative=out_dict['reference_z'][2][:, :, :].transpose(0, 1),
                                          temperature=0.5,
                                          use_cosine_similarity=True)
            elif cont_layer == 2:
                # positive_x_reference_z, positive_x_predict_z, negative_x_reference_z = out_dict['vis']
                cont_loss = cont_loss_weight_last*contrast_loss(target=out_dict['vis'][0][:,:],
                                          positive=out_dict['vis'][1][:,:],
                                          negative=out_dict['vis'][2][:,:],
                                          temperature=0.5,
                                          use_cosine_similarity=True)
            elif cont_layer == 3:
                cont_loss_first_layer = cont_loss_weight_first*contrast_loss(target=X[:, :, :],
                                          positive=out_dict['reference_z'][1][:, :, :].transpose(0, 1),
                                          negative=out_dict['reference_z'][2][:, :, :].transpose(0, 1),
                                          temperature=0.5,
                                          use_cosine_similarity=True)
                cont_loss_last_layer = cont_loss_weight_last*contrast_loss(target=out_dict['vis'][0][:,:],
                                          positive=out_dict['vis'][1][:,:],
                                          negative=out_dict['vis'][2][:,:],
                                          temperature=0.5,
                                          use_cosine_similarity=True)
                cont_loss = cont_loss_first_layer + cont_loss_last_layer
            ## ②将原始输入，x点乘M，x点乘（1-M）三者进入模型的特征编码分别作为锚样本、正样本和负样本，计算损失函数
            # target_z, positive_z, negative_z = out_dict['vis']
            # cont_loss = 10 * contrast_loss(target=target_z, positive=positive_z, negative=negative_z, temperature = 0.5, use_cosine_similarity = True)
            ## ③将训练集各个类别的特征编码均值作为特征锚点，参考《ICICLE: Interpretable Class Incremental Continual Learning》，
            ##

            ## ④计算掩码矩阵1的个数，保证|M|作为损失函数，保证稀疏性
            sparse_loss = sparse_loss_weight * torch.mean(out_dict['ste_mask'].sum(1))

            ## ⑤计算生成的正样本与噪声填充的正样本间的距离，避免过拟合
            sim_loss = sim_loss_weight*F.mse_loss(out_dict['reference_z'][0], out_dict['reference_z'][1])

            loss = clf_loss + exp_loss + cont_loss + sparse_loss + sim_loss + kl_loss# sim_loss # cont_loss + sparse_loss # #sim_loss # # #
            # +sim_loss
            # + conc_emb_loss
            # print('---------')
            # print('clf', clf_loss)
            # print('exp', exp_loss)
            # print('sim', sim_loss)
            # print('loss', loss)

            # import ipdb; ipdb.set_trace()

            if clip_norm:
                # print('Clip')
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # print('loss', loss.item())
            # exit()

            loss.backward()
            optimizer.step()
            # scheduler.step()

            cum_sparse.append(((ste_mask).sum() / ste_mask.flatten().shape[0]).item())
            cum_clf_loss.append(clf_loss.detach().item())
            cum_exp_loss.append([exp_loss.detach().clone().item()])
            # cum_sim_loss.append(sim_loss.detach().item())
            cum_cont_loss.append(cont_loss.detach().item())
            cum_sparse_loss.append(sparse_loss.detach().item())
            cum_sim_loss.append(sim_loss.detach().item())
            cum_kl_loss.append(kl_loss.detach().item())

        # Print all stats:
        # Convert to np:
        sparse = np.array(cum_sparse)  # Should be size (B, M)
        sparse = sparse.mean()
        clf = sum(cum_clf_loss) / len(cum_clf_loss)
        exp = np.array(cum_exp_loss)  # Size (B, M, L)
        exp = exp.mean(axis=0).flatten()
        # sim = np.mean(cum_sim_loss)
        cont = np.mean(cum_cont_loss)
        sparse_loss = np.mean(cum_sparse_loss)
        sim = np.mean(cum_sim_loss)
        kl = np.mean(cum_kl_loss)

        # sim_s = f'{sim:.4f}'

        print(
            f'Epoch: {epoch}: Sparsity = {sparse:.4f} \t Exp Loss = {exp} \t Clf Loss = {clf:.4f} \t Cont Loss = {cont}\t sparse Loss = {cont_loss}\t Sim Loss = {sim}\t kl Loss = {kl}')

        # Eval after every epoch
        # Call evaluation function:
        model.eval()

        if batch_forward_size is None:
            f1, out = eval_mv4(val_tuple, model)
        else:
            out = batch_forwards(model, val_tuple[0], val_tuple[1], batch_size=64)
            f1 = 0

        ste_mask = out['ste_mask']
        sparse = ste_mask.mean().item()

        # met = -1.0 * loss
        if test != None:
            if len(test) !=3:
                f1, _, results_dict = eval_mv4_loader(test, model, gt_exps=gt_exps)
            else:
                f1, _, results_dict = eval_mv4(test, model, gt_exps=gt_exps)
            generated_exps = results_dict["generated_exps"]
            gt_exps = results_dict["gt_exps"]
            resutlt_cur = print_results(generated_exps, gt_exps)
            met = np.mean(resutlt_cur['AUPRC'])
        else:
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