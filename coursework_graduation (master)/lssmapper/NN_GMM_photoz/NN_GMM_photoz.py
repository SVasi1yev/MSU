import pickle
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch

import DeepEnsemble


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NN GMM модель для оценки photo-z')
    parser.add_argument('models', type=str, help='Номера моделей через запятую (18,20,21,22,35)')
    parser.add_argument('gm_params', type=int, help='Флаг сохранения параметров смеси гауссиан (0 - не сохранять)')
    parser.add_argument('pdf_num', type=int, help='Число объектов эмпирического распределения (<= 0 - не вычислять эмпирическое распределение)')
    parser.add_argument('features_file', type=str, help='Файл с признаками')
    parser.add_argument('out_file', type=str, help='Файл с результатом')
    args = parser.parse_args()

    with open(f'feature_lists/features_sdssdr16+wise_deacls8tr_QSO+GALAXY_20201212141009.pkl', 'rb') as f:
        features_18 = pickle.load(f)
    with open(f'feature_lists/features_sdssdr16+all_deacls8tr_QSO+GALAXY_20201212143658.pkl', 'rb') as f:
        features_20 = pickle.load(f)
    with open(f'feature_lists/features_psdr2+all_deacls8tr_QSO+GALAXY_20201212142333.pkl', 'rb') as f:
        features_21 = pickle.load(f)
    with open(f'feature_lists/features_deacls8tr_QSO+GALAXY_20201212135641.pkl', 'rb') as f:
        features_22 = pickle.load(f)
    with open(f'feature_lists/features_sdssdr16+psdr2+all_deacls8tr_QSO+GALAXY_20201212133711.pkl', 'rb') as f:
        features_35 = pickle.load(f)

    features_dict = {
        '18': features_18,
        '20': features_20,
        '21': features_21,
        '22': features_22,
        '35': features_35,
    }

    models_dir = 'trained_models'
    models_dict = {
        '18': f'{models_dir}/train20_18_g5_m5_model_full.pkl',
        '20': f'{models_dir}/train20_20_g5_m5_model_full.pkl',
        '21': f'{models_dir}/train20_21_g5_m5_model_full.pkl',
        '22': f'{models_dir}/train20_22_g5_m5_model_full.pkl',
        '35': f'{models_dir}/train20_35_g5_m5_model_full.pkl',
    }
    models_priors = ['35', '20', '18', '21', '22']

    np.random.seed(2)
    torch.manual_seed(2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    models = set(map(lambda x: x.strip(), args.models.split(',')))
    gm_params = args.gm_params
    pdf_num = args.pdf_num
    features_file = args.features_file
    out_file = args.out_file

    
    idx_col = '__idx__'
    data = pd.read_pickle(features_file, compression='gzip')
    data[idx_col] = range(data.shape[0])
    best = data.copy()
    idx = data[[idx_col]]
    i = 0
    for model_num in tqdm(models_priors):
        if (model_num not in models) or (model_num not in models_dict):
            continue
        features = features_dict[model_num]
        t = data[data[[idx_col] + features].isna().values.sum(1) == 0]
        X = t[features].values.astype(float)
        
        model = DeepEnsemble.DeepEnsemble_GMM.load_pickle(models_dict[model_num])

        result = model.predict(X, samples_num=pdf_num, save_gm_params=gm_params)
        result_best = result.copy()
        result.columns = list(map(lambda x: f'm{model_num}_' + x, result.columns))
        result[idx_col] = t[idx_col].values
        result_best[idx_col] = t[idx_col].values
        data = data.merge(result, how='left', on=idx_col)
        if i == 0:
            best = best.merge(result_best, how='left', on=idx_col)
        else:
            tt = idx.merge(result_best, how='left', on=idx_col)
            m = best['mode'].isna()
            best.loc[m, tt.columns] = tt.loc[m, tt.columns]
        
        i += 1
        
    data.to_pickle(out_file, compression='gzip')
    best.to_pickle('best.' + out_file, compression='gzip')
