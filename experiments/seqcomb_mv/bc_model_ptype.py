import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
import torch
import sys


from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
from sklearn import metrics
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import *
from txai.utils.predictors.select_models import *
import random
import numpy as np

is_timex = False
is_timexplusplus = False

if is_timex:
    from txai.models.bc_model4 import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv4_consistency import train_mv6_consistency
elif is_timexplusplus:
    from txai.models.bc_model import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv6_consistency import train_mv6_consistency
else:
    from txai.models.bc_model8 import TimeXModel, AblationParameters, transformer_default_args
    from txai.trainers.train_mv8_consistency import train_mv6_consistency

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def naming_convention(args,time):
    if args.eq_ge:
        name = "bc_eqge_split={}.pt"
    elif args.eq_pret:
        name = "bc_eqpret_split={}.pt"
    elif args.ge_rand_init:
        name = "bc_gerand_split={}.pt"
    elif args.no_ste:
        name = "bc_noste_split={}.pt"
    elif args.simclr:
        name = "bc_simclr_split={}.pt"
    elif args.no_la:
        name = "bc_nola_split={}.pt"
    elif args.no_con:
        name = "bc_nocon_split={}.pt"
    elif args.cnn:
        name = "bc_cnn_split={}.pt"
    elif args.lstm:
        name = "bc_lstm_split={}.pt"
    else:
        name = 'bc_full_LC_split={}.pt'
    if not is_timex:
        name = 'our_'+ time + '_' + name
    if args.lam != 1.0:
        name = name[:-3] + '_lam={}'.format(args.lam) + '.pt'
    
    return name

def main(args, formstyleTime):



    if args.lstm:
        arch = 'lstm'
        tencoder_path = "./models/ScombMV_lstm_split={}.pt"
    elif args.cnn:
        arch = 'cnn'
        tencoder_path = "./models/ScombMV_cnn_split={}.pt"
    else:
        arch = 'transformer'
        tencoder_path = "./models/transformer_split={}.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf_criterion = Poly1CrossEntropyLoss(
        num_classes = 4,
        epsilon = 1.0,
        weight = None,
        reduction = 'mean'
    )

    sim_criterion_label = LabelConsistencyLoss()
    if args.simclr:
        sim_criterion_cons = SimCLRLoss()
        sc_expand_args = {'simclr_training':True, 'num_negatives_simclr':32}
    else:
        sim_criterion_cons = EmbedConsistencyLoss(normalize_distance = False)
        sc_expand_args = {'simclr_training':False, 'num_negatives_simclr':32}


    if args.no_la:
        sim_criterion = sim_criterion_cons
        selection_criterion = simloss_on_val_cononly(sim_criterion)
        label_matching = False
        embedding_matching = True
    elif args.no_con:
        sim_criterion = sim_criterion_label
        selection_criterion = simloss_on_val_laonly(sim_criterion)
        label_matching = True
        embedding_matching = False
    else: # Regular
        sim_criterion = [sim_criterion_cons, sim_criterion_label]
        if args.simclr:
            selection_criterion = simloss_on_val_wboth([cosine_sim_for_simclr, sim_criterion_label], lam = 1.0)
        else:
            selection_criterion = simloss_on_val_wboth(sim_criterion, lam = 1.0)
        label_matching = True
        embedding_matching = True

    targs = transformer_default_args
    all_results = {"AUROC": [],
                   "AUPRC": [],
                   "AUP": [],
                   "AUR": [],
                             }

    for i in range(1, 6):
        set_seed(args.seed)
        print('seed:{}'.format(args.seed))
        D = process_Synth(split_no = i, device = device, base_path = '')
        dset = DatasetwInds(D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device))
        train_loader = torch.utils.data.DataLoader(dset, batch_size = 64, shuffle = True)

        val, test = D['val'], D['test']
        gt_exps = D['gt_exps']

        # Calc statistics for baseline:
        mu = D['train_loader'].X.mean(dim=1)
        std = D['train_loader'].X.std(unbiased = True, dim = 1)

        # Change transformer args:
        targs['trans_dim_feedforward'] = 128
        targs['trans_dropout'] = 0.25
        targs['nlayers'] = 2
        targs['norm_embedding'] = False

        abl_params = AblationParameters(
            equal_g_gt = args.eq_ge,
            g_pret_equals_g = args.eq_pret, 
            label_based_on_mask = True,
            ptype_assimilation = True, 
            side_assimilation = True,
            use_ste = (not args.no_ste),
            archtype = arch,
        )

        loss_weight_dict = {
            'gsat': 1.0,
            'connect': 2.0
        }

        model = TimeXModel(
            d_inp = 4,
            max_len = 200,
            n_classes = 4,
            n_prototypes = 50,
            gsat_r = 0.5,
            transformer_args = targs,
            ablation_parameters = abl_params,
            loss_weight_dict = loss_weight_dict,
            masktoken_stats = (mu, std),

            use_decoupled_ste=args.decoupled_ste,
            forward_ste_temperature=args.f_ste,
            backward_ste_temperature=args.b_ste,
            ste_learning_rate=args.lr_ste,
            noise_level=args.noise_level,
        )

        model.encoder_main.load_state_dict(torch.load(tencoder_path.format(i)))
        model.to(device)

        if is_timex:
            model.init_prototypes(train = (D['train_loader'].X.to(device), D['train_loader'].times.to(device), D['train_loader'].y.to(device)))

            if not args.ge_rand_init: # Copies if not running this ablation
                model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

        for param in model.encoder_main.parameters():
            param.requires_grad = False
        if is_timex:
            optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.001) #For regular
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = 0.001) #For regular

        
        model_suffix = naming_convention(args, formstyleTime)
        spath = os.path.join('models', model_suffix)
        spath = spath.format(i)
        print('saving at', spath)

        y = test[2]
        X = test[0][:,(y != 0),:]
        times = test[1][:,y != 0]
        gt_exps = D['gt_exps'][:,(y != 0).detach().cpu(),:]
        y = y[y != 0]
        test = (X, times, y)

        best_model = train_mv6_consistency(
            model,
            optimizer = optimizer,
            train_loader = train_loader,
            clf_criterion = clf_criterion,
            sim_criterion = sim_criterion,
            beta_exp = 2.0,
            beta_sim = 1.0,
            lam_label = 1.0,
            val_tuple = val,
            num_epochs = args.epoch,
            save_path = spath,
            train_tuple = (D['train_loader'].X, D['train_loader'].times, D['train_loader'].y),
            early_stopping = True,
            selection_criterion = selection_criterion,
            label_matching = label_matching,
            embedding_matching = embedding_matching,
            use_scheduler = False,
            clip_norm = False,
            **sc_expand_args,

            cont_layer=args.cont_layer,
            cont_loss_weight_first=args.cont_lw_first,
            cont_loss_weight_last=args.cont_lw_last,
            sparse_loss_weight=args.sparse_lw,
            sim_loss_weight=args.sim_lw,
            kl_loss_weight=args.kl_lw,
            use_decoupled_ste=args.decoupled_ste,
            forward_ste_temperature=args.f_ste,
            backward_ste_temperature=args.b_ste,
            ste_learning_rate=args.lr_ste,
            test=test,
            gt_exps=gt_exps,
        )

        sdict, config = torch.load(spath)

        model.load_state_dict(sdict)

        f1, _, results_dict = eval_mv4(test, model, gt_exps=gt_exps)

        for k, v in results_dict.items():
            print(k)
            if k not in "generated_exps" and k not in "gt_exps":
                print('\t{} \t = {:.4f} +- {:.4f}'.format(k, np.mean(v), np.std(v) / np.sqrt(len(v))))
        print('Test F1: {:.4f}'.format(f1))


        generated_exps = results_dict["generated_exps"]
        gt_exps = results_dict["gt_exps"]
        resutlt_cur = print_results(generated_exps, gt_exps)
        all_results["AUROC"] += resutlt_cur["AUROC"]
        all_results["AUPRC"] += resutlt_cur["AUPRC"]
        all_results["AUP"] += resutlt_cur["AUP"]
        all_results["AUR"] += resutlt_cur["AUR"]
    all_aupoc, all_auprc, all_aup, all_aur = all_results["AUROC"], all_results["AUPRC"], all_results["AUP"], all_results["AUR"]
    print('============================================================================================')
    print('Saliency AUROC: = {:.4f} +- {:.4f}'.format(np.mean(all_aupoc), np.std(all_aupoc) / np.sqrt(len(all_aupoc))))
    print('Saliency AUPRC: = {:.4f} +- {:.4f}'.format(np.mean(all_auprc), np.std(all_auprc) / np.sqrt(len(all_auprc))))
    print('Saliency AUP: = {:.4f} +- {:.4f}'.format(np.mean(all_aup), np.std(all_aup) / np.sqrt(len(all_aup))))
    print('Saliency AUR: = {:.4f} +- {:.4f}'.format(np.mean(all_aur), np.std(all_aur) / np.sqrt(len(all_aur))))


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
        AUROC = metrics.roc_auc_score(true_label.flatten(), mask_label.flatten())
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

def normalize_exp(exps):
    norm_exps = torch.empty_like(exps)
    for i in range(exps.shape[1]):
        norm_exps[:,i,:] = (exps[:,i,:] - exps[:,i,:].min()) / (exps[:,i,:].max() - exps[:,i,:].min() + 1e-9)
    return norm_exps


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ablations = parser.add_mutually_exclusive_group()
    ablations.add_argument('--eq_ge', action = 'store_true', help = 'G = G_E')
    ablations.add_argument('--eq_pret', action = 'store_true', help = 'G_pret = G')
    ablations.add_argument('--ge_rand_init', action = 'store_true', help = "Randomly initialized G_E, i.e. don't copy")
    ablations.add_argument('--no_ste', action = 'store_true', default= False, help = 'Does not use STE')
    ablations.add_argument('--simclr', action = 'store_true', help = 'Uses SimCLR loss instead of consistency loss')
    ablations.add_argument('--no_la', action = 'store_true', help = 'No label alignment - just consistency loss')
    ablations.add_argument('--no_con', action = 'store_true', help = 'No consistency loss - just label')

    ablations.add_argument('--lstm', action = 'store_true', default=False)
    ablations.add_argument('--cnn', action = 'store_true', default=True)

    parser.add_argument('--r', type = float, default = 0.5, help = 'r for GSAT loss')
    parser.add_argument('--lam', type = float, default = 1.0, help = 'lambda between label alignment and consistency loss')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--lr', type = float, default = 0.0005)
    parser.add_argument('--epoch', type = int, default = 100)
    parser.add_argument('--cont_layer', type = float, default = 1.0,
                        help = 'contrastive layer: 1--first layer; 2--last layer; 3--first and last layer')
    parser.add_argument('--cont_lw_first', type = float, default = 10, help = 'first_layer: cont_loss_weight')
    parser.add_argument('--cont_lw_last', type = float, default = 10.0, help = 'last_layer: cont_loss_weight')
    parser.add_argument('--sparse_lw', type = float, default = 0.035, help = 'sparse_loss_weight')
    parser.add_argument('--sim_lw', type = float, default = 1, help = 'similar_loss_mse_weight')
    parser.add_argument('--kl_lw', type=float, default=0, help='kl_loss_mse_weight')
    parser.add_argument('--decoupled_ste', type = bool, default = True, help = 'use decoupled forward and backward weight')
    parser.add_argument('--f_ste', type = float, default = 1, help = 'ste: forward weight')
    parser.add_argument('--b_ste', type = float, default = 3, help = 'ste: backward weight')
    parser.add_argument('--lr_ste', type = float, default = 0.001, help = 'ste: learning rate')
    parser.add_argument('--noise_level', type=float, default=2.5)
    args = parser.parse_args()

    formstyleTime = set_time()

    main(args, formstyleTime)