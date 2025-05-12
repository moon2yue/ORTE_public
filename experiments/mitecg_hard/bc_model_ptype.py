import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import argparse, os
import sys
import numpy as np
from sklearn import metrics


from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data import process_Synth
from txai.utils.predictors.eval import eval_mv4, eval_mv4_loader
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import *
from txai.utils.predictors.select_models import simloss_on_val_wboth, simloss_on_val_laonly, simloss_on_val_cononly, cosine_sim_for_simclr
from txai.utils.data.preprocess import process_MITECG

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
    elif args.runtime_exp:
        name = None
        return name
    else:
        name = 'bc_full_retry_split={}.pt'
    if not is_timex:
        name = 'our_'+ time + '_' + name
    if args.lam != 1.0:
        # Not included in ablation parameters or other, so name it;
        name = name[:-3] + '_lam={}'.format(args.lam) + '.pt'
    
    return name

def main(args, formstyleTime):

    if args.lstm:
        arch = 'lstm'
        tencoder_path = "models/MITECG-Hard_lstm_split={}.pt"
    elif args.cnn:
        arch = 'cnn'
        tencoder_path = "models/MITECG-Hard_cnn_split={}.pt"
    else:
        arch = 'transformer'
        tencoder_path = "models/transformer_exc_split={}.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf_criterion = Poly1CrossEntropyLoss(
        num_classes = 2,
        epsilon = 1.0,
        weight = None,
        reduction = 'mean'
    )

    sim_criterion_label = LabelConsistencyLoss_LS()

    if args.simclr:
        sim_criterion_cons = SimCLRLoss()
        sc_expand_args = {'simclr_training':True, 'num_negatives_simclr':32, 'max_batch_size_simclr_negs': 32}
    else:
        sim_criterion_cons = EmbedConsistencyLoss(normalize_distance = True)

        sc_expand_args = {'simclr_training':False, 'num_negatives_simclr':64, 'max_batch_size_simclr_negs': None}

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
    for i in range(2, 6):
        set_seed(args.seed)
        print('seed:{}'.format(args.seed))
        trainEpi, val, test, gt_exps = process_MITECG(split_no = i, device = device, hard_split = True, need_binarize = True,
            base_path = '')
        train_dataset = DatasetwInds(trainEpi.X, trainEpi.time, trainEpi.y)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
        test_dataset = DatasetwInds(test.X[:,0:1000,:], test.time[:,0:1000], test.y[0:1000])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 256, shuffle = False)
        val = (val.X, val.time, val.y)
        test = (test.X, test.time, test.y)

        mu = trainEpi.X.mean(dim=1)
        std = trainEpi.X.std(unbiased = True, dim = 1)
        targs['trans_dim_feedforward'] = 64
        targs['trans_dropout'] = 0.1
        targs['nlayers'] = 1
        targs['stronger_clf_head'] = False
        targs['norm_embedding'] = True

        abl_params = AblationParameters(
            equal_g_gt = args.eq_ge,
            g_pret_equals_g = args.eq_pret, 
            label_based_on_mask = True,
            ptype_assimilation = True, 
            side_assimilation = True,
            use_ste = (not args.no_ste),
            archtype = arch
        )

        loss_weight_dict = {
            'gsat': 1.0,
            'connect': 2.0
        }

        model = TimeXModel(
            d_inp = val[0].shape[-1],
            max_len = val[0].shape[0],
            n_classes = 2,
            n_prototypes = 50,
            gsat_r = 0.5,
            transformer_args = targs,
            ablation_parameters = abl_params,
            loss_weight_dict = loss_weight_dict,
            tau = 1.0,
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
            model.init_prototypes(train = (trainEpi.X.to(device), trainEpi.time.to(device), trainEpi.y.to(device)))

            if not args.ge_rand_init:
                model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

        for param in model.encoder_main.parameters():
            param.requires_grad = False

        if args.cnn and args.lstm:
            optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.0001)
            num_epochs = 10
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
        
        model_suffix = naming_convention(args, formstyleTime)
        spath = os.path.join('models', model_suffix)
        spath = spath.format(i)
        print('saving at', spath)

        best_model = train_mv6_consistency(
            model,
            optimizer = optimizer,
            train_loader = train_loader,
            clf_criterion = clf_criterion,
            sim_criterion = sim_criterion,
            beta_exp = 2.0,
            beta_sim = 1.0,
            val_tuple = val, 
            num_epochs = args.epoch,
            save_path = spath,
            train_tuple = (trainEpi.X.to(device), trainEpi.time.to(device), trainEpi.y.to(device)),
            early_stopping = True,
            selection_criterion = selection_criterion,
            label_matching = label_matching,
            embedding_matching = embedding_matching,
            use_scheduler = False,
            batch_forward_size = 64,
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
            test=test_loader,
            gt_exps=gt_exps[:,0:1000,:],
            # scheduler=scheduler,
        )

        sdict, config = torch.load(spath)

        model.load_state_dict(sdict)


        if args.lstm or args.cnn:   # lstm  CUDA out of memory.
            split_me = 10000
        else:
            split_me = 30000
        gt_exps1 = gt_exps[:, :split_me, :]

        f1, _, results_dict1 = eval_mv4_loader(test_loader, model, gt_exps=gt_exps1)


        for k, v in results_dict1.items():
            print(k)
            if k not in "generated_exps" and k not in "gt_exps":
                print('\t{} \t = {:.4f} +- {:.4f}'.format(k, np.mean(v), np.std(v) / np.sqrt(len(v))))

        print('Test F1: {:.4f}'.format(f1))

        all_results["AUPRC"] += (results_dict1["auprc"])
        all_results["AUP"] += (results_dict1["aup"])
        all_results["AUR"] += (results_dict1["aur"])
    all_auprc, all_aup, all_aur = all_results["AUPRC"], all_results["AUP"], all_results["AUR"]
    print('============================================================================================')
    print('Saliency AUPRC: = {:.4f} +- {:.4f}'.format(np.mean(all_auprc), np.std(all_auprc) / np.sqrt(len(all_auprc))))
    print('Saliency AUP: = {:.4f} +- {:.4f}'.format(np.mean(all_aup), np.std(all_aup) / np.sqrt(len(all_aup))))
    print('Saliency AUR: = {:.4f} +- {:.4f}'.format(np.mean(all_aur), np.std(all_aur) / np.sqrt(len(all_aur))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    ablations = parser.add_mutually_exclusive_group()
    ablations.add_argument('--eq_ge', action = 'store_true', help = 'G = G_E')
    ablations.add_argument('--eq_pret', action = 'store_true', help = 'G_pret = G')
    ablations.add_argument('--ge_rand_init', action = 'store_true', help = "Randomly initialized G_E, i.e. don't copy")
    ablations.add_argument('--no_ste', action = 'store_true', help = 'Does not use STE')
    ablations.add_argument('--simclr', action = 'store_true', help = 'Uses SimCLR loss instead of consistency loss')
    ablations.add_argument('--no_la', action = 'store_true', help = 'No label alignment - just consistency loss')
    ablations.add_argument('--no_con', action = 'store_true', help = 'No consistency loss - just label')
    ablations.add_argument('--lstm', action = 'store_true', default=False)
    ablations.add_argument('--cnn', action = 'store_true', default=True)
    ablations.add_argument('--runtime_exp', action = 'store_true')

    parser.add_argument('--r', type = float, default = 0.5, help = 'r for GSAT loss')
    parser.add_argument('--lam', type = float, default = 1.0, help = 'lambda between label alignment and consistency loss')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type = float, default = 0.0001)
    parser.add_argument('--epoch', type = int, default = 5)
    parser.add_argument('--cont_layer', type = float, default = 1.0,
                        help = 'contrastive layer: 1--first layer; 2--last layer; 3--first and last layer')
    parser.add_argument('--cont_lw_first', type = float, default = 10, help = 'first_layer: cont_loss_weight')
    parser.add_argument('--cont_lw_last', type = float, default = 10.0, help = 'last_layer: cont_loss_weight')
    parser.add_argument('--sparse_lw', type = float, default = 0, help = 'sparse_loss_weight')
    parser.add_argument('--sim_lw', type = float, default = 1, help = 'similar_loss_mse_weight')
    parser.add_argument('--kl_lw', type=float, default=0, help='kl_loss_mse_weight')
    parser.add_argument('--decoupled_ste', type = bool, default = True, help = 'use decoupled forward and backward weight')
    parser.add_argument('--f_ste', type = float, default = 1, help = 'ste: forward weight')
    parser.add_argument('--b_ste', type = float, default = 3, help = 'ste: backward weight')
    parser.add_argument('--lr_ste', type = float, default = 0.001, help = 'ste: learning rate')
    parser.add_argument('--noise_level', type=float, default=1)
    args = parser.parse_args()

    formstyleTime = set_time()

    main(args, formstyleTime)



