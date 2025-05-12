import re
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from txai.models.encoders.transformer_simple import TransformerMVTS
from txai.utils.predictors import eval_mvts_transformer

from txai.utils.data import process_Synth
from txai.synth_data.simple_spike import SpikeTrainDataset
from txai.utils.data.preprocess import process_MITECG, process_Epilepsy, process_PAM, process_Boiler_OLD

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_model(args, X):

    if args.dataset == 'scs_better':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            n_classes = 4,
            nlayers = 2,
            nhead = 1,
            trans_dim_feedforward = 64,
            trans_dropout = 0.25,
            d_pe = 16,
        )

    elif args.dataset == 'freqshape':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            n_classes = 4,
            trans_dim_feedforward = 16,
            trans_dropout = 0.1,
            d_pe = 16,
        )

    elif args.dataset == 'seqcomb_mv':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            nlayers = 2,
            n_classes = 4,
            trans_dim_feedforward = 128,
            trans_dropout = 0.25,
            d_pe = 16,
        )

    elif args.dataset == 'lowvardetect':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            nlayers = 1,
            n_classes = 4,
            trans_dim_feedforward = 32,
            trans_dropout = 0.25,
            d_pe = 16,
            stronger_clf_head = False,
        )
    
    elif args.dataset == 'mitecg_hard':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            n_classes = 2,
            nlayers = 1,
            nhead = 1,
            trans_dim_feedforward = 64,
            trans_dropout = 0.1,
            d_pe = 16,
            stronger_clf_head = False,
            pre_agg_transform = False,
            norm_embedding = True
        )

    elif args.dataset == 'pam':
        model = TransformerMVTS(
            d_inp = X.shape[2],
            max_len = X.shape[0],
            n_classes = 8,
        )
    
    elif args.dataset == 'epilepsy':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            n_classes = 2,
            nlayers = 1,
            trans_dim_feedforward = 16,
            trans_dropout = 0.1,
            d_pe = 16,
            norm_embedding = False,
        )
    
    elif args.dataset == 'boiler':
        model = TransformerMVTS(
            d_inp = X.shape[-1],
            max_len = X.shape[0],
            n_classes = 2,
            nlayers = 1,
            trans_dim_feedforward = 32,
            trans_dropout = 0.25,
            d_pe = 16,
            norm_embedding = True,
            stronger_clf_head = False,
        )

    return model

def main(args):

    Dname = args.dataset.lower()

    if Dname == 'freqshape':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'FreqShape')
        test = D['test']
    elif Dname == 'scs_better':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'SeqCombSingleBetter')
        test = D['test']
    elif Dname == 'seqcomb_mv':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'SeqCombMV')
        test = D['test']
    elif Dname == 'lowvardetect':
        D = process_Synth(split_no = args.split_no, device = device, base_path = Path(args.data_path) / 'LowVarDetect')
        test = D['test']
    elif Dname == 'pam':
        trainPAM, val, test = process_PAM(split_no = args.split_no, device = device, base_path = '', gethalf = True)
        test = (test.X, test.time, test.y)
    elif Dname == 'epilepsy':
        trainEpi, val, test = process_Epilepsy(split_no = args.split_no, device = device, base_path = '')
        test = (test.X, test.time, test.y)
    elif Dname == 'boiler':
        _, _, test = process_Boiler_OLD(split_no = args.split_no, device = device, base_path = '')
    elif Dname == 'mitecg_hard':
        D = process_MITECG(split_no = args.split_no, device = device, hard_split = True, need_binarize = True, exclude_pac_pvc = True, base_path = Path(args.data_path) / 'MITECG-Hard')
        _, _, test, gt_exps = D
        test = (test.X, test.time, test.y)

    X, times, y = test

    T, B, d = X.shape

    model = get_model(args, X)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    model.eval()
    
    f1, auprc, auroc = eval_mvts_transformer(test, model, batch_size = 32, auroc = True, auprc = True)

    print('F1 = {:.4f}'.format(f1))
    print('AUPRC = {:.4f}'.format(auprc))
    print('AUROC = {:.4f}'.format(auroc))

    return f1, auprc, auroc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type = str)
    parser.add_argument('--split_no', default = 1, type=int)
    parser.add_argument('--model_path', type = str, help = 'only time series transformer right now')
    parser.add_argument('--data_path', default="", type = str, help = 'path to datasets root')

    args = parser.parse_args()
    if args.split_no == -1:
        # eval results on all splits
        results = {}
        f1_list = []
        auprc_list = []
        auroc_list = []
        for split in range(1, 6):
            # replace model path with correct split
            args.model_path = re.sub("split=\d", f"split={split}", args.model_path)
            print("model path:", args.model_path)
            args.split_no = split
            f1, auprc, auroc = main(args)
            f1_list.append(f1); auprc_list.append(auprc); auroc_list.append(auroc)

        print('Results:')
        print('F1 = {:.4f} +- {:.4f}'.format(np.mean(f1_list), np.std(f1_list) / np.sqrt(len(f1_list))))
        print('AUPRC = {:.4f} +- {:.4f}'.format(np.mean(auprc_list), np.std(auprc_list) / np.sqrt(len(auprc_list))))
        print('AUROC = {:.4f} +- {:.4f}'.format(np.mean(auroc_list), np.std(auroc_list) / np.sqrt(len(auroc_list))))

    else:
        main(args)