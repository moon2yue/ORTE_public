import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import argparse
import sys
import numpy as np
from sklearn import metrics
import time
from txai.utils.predictors.loss import Poly1CrossEntropyLoss, GSATLoss_Extended, ConnectLoss_Extended
from txai.utils.predictors.loss_smoother_stats import *
from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.data.preprocess import process_Boiler_OLD, process_Epilepsy
from txai.utils.predictors.eval import eval_mv4
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.datasets import DatasetwInds
from txai.utils.predictors.loss_cl import *
from txai.utils.predictors.select_models import simloss_on_val_wboth

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
    elif args.runtime_exp:
        name = None
    else:
        name = 'bc_full_split={}.pt'
    if not is_timex:
        name = 'our_'+ time + '_' + name
    if args.lam != 1.0:
        name = name[:-3] + '_lam={}'.format(args.lam) + '.pt'
    
    return name

def main(args, formstyleTime):

    tencoder_path = ""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf_criterion = Poly1CrossEntropyLoss(
        num_classes = 2,
        epsilon = 1.0,
        weight = None,
        reduction = 'mean'
    )

    sim_criterion_label = LabelConsistencyLoss()
    sim_criterion_cons = EmbedConsistencyLoss()

    if args.no_la:
        sim_criterion = sim_criterion_cons
    elif args.no_con:
        sim_criterion = sim_criterion_label
    else:
        sim_criterion = [sim_criterion_cons, sim_criterion_label]
        selection_criterion = simloss_on_val_wboth(sim_criterion, lam = 1.0)

    targs = transformer_default_args

    for i in range(1, 6):
        trainEpi, val, test = process_Epilepsy(split_no = i, device = device, base_path = '')
        trainB = (trainEpi.X, trainEpi.time, trainEpi.y)
        train_dataset = DatasetwInds(*trainB)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)

        val = (val.X, val.time, val.y)
        test = (test.X, test.time, test.y)

        mu = trainB[0].mean(dim=1)
        std = trainB[0].std(unbiased = True, dim = 1)

        abl_params = AblationParameters(
            equal_g_gt = args.eq_ge,
            g_pret_equals_g = args.eq_pret, 
            label_based_on_mask = True,
            ptype_assimilation = True, 
            side_assimilation = True,
            use_ste = (not args.no_ste),
        )

        loss_weight_dict = {
            'gsat': 1.0,
            'connect': 2.0
        }

        targs['trans_dim_feedforward'] = 16
        targs['trans_dropout'] = 0.1
        targs['norm_embedding'] = False

        model = TimeXModel(
            d_inp = val[0].shape[-1],
            max_len = val[0].shape[0],
            n_classes = 2,
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
            model.init_prototypes(train = trainB)

            if not args.ge_rand_init:
                model.encoder_t.load_state_dict(torch.load(tencoder_path.format(i)))

        for param in model.encoder_main.parameters():
            param.requires_grad = False

        if is_timex:
            optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4, weight_decay = 0.001)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr = args.lr, weight_decay = 0.001)

        
        model_suffix = naming_convention(args, formstyleTime)
        if model_suffix is not None:
            spath = os.path.join('models', model_suffix)
            spath = spath.format(i)
            print('saving at', spath)
        else:
            spath = None

        start_time = time.time()

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
            train_tuple = trainB,
            early_stopping = True,
            selection_criterion = selection_criterion,
            label_matching = True,
            embedding_matching = True,
            use_scheduler = True,

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
        )

        end_time = time.time()

        print('Time {}'.format(end_time - start_time))
        if args.runtime_exp:
            exit()

        sdict, config = torch.load(spath)

        model.load_state_dict(sdict)

        f1, _ = eval_mv4(test, model, masked = True)
        print('Test F1: {:.4f}'.format(f1))

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
    ablations.add_argument("--runtime_exp", action = 'store_true')

    parser.add_argument('--r', type = float, default = 0.5, help = 'r for GSAT loss')
    parser.add_argument('--lam', type = float, default = 1.0, help = 'lambda between label alignment and consistency loss')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type = float, default = 0.005)
    parser.add_argument('--epoch', type = int, default = 10)
    parser.add_argument('--cont_layer', type = float, default = 1.0,
                        help = 'contrastive layer: 1--first layer; 2--last layer; 3--first and last layer')
    parser.add_argument('--cont_lw_first', type = float, default = 0.1, help = 'first_layer: cont_loss_weight')
    parser.add_argument('--cont_lw_last', type = float, default = 5.0, help = 'last_layer: cont_loss_weight')
    parser.add_argument('--sparse_lw', type = float, default = 0.0005, help = 'sparse_loss_weight')
    parser.add_argument('--sim_lw', type = float, default = 1, help = 'similar_loss_mse_weight')
    parser.add_argument('--kl_lw', type=float, default=0, help='kl_loss_mse_weight')
    parser.add_argument('--decoupled_ste', type = bool, default = True, help = 'use decoupled forward and backward weight')
    parser.add_argument('--f_ste', type = float, default = 1, help = 'ste: forward weight')
    parser.add_argument('--b_ste', type = float, default = 3, help = 'ste: backward weight')
    parser.add_argument('--lr_ste', type = float, default = 0.001, help = 'ste: learning rate')
    parser.add_argument('--noise_level', type=float, default=2)
    args = parser.parse_args()

    formstyleTime = set_time()

    main(args, formstyleTime)



