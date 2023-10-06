import argparse
import os
import yaml
import pickle as pkl

# from torch.utils.data import DataLoader
from utils.log import init_seed
from dataloader import Dataloader, DomainDataloader
from utils.log import init_logger
from utils.yaml_loader import get_yaml_loader
from logging import getLogger
from models import LR, WDL, DSSM, FM, DeepFM, YouTubeDNN, FFM, PNN, DCN, DIN, UBR, COSMO, DIEN, GRU4REC,BST,DIN_SESSION,MISS,COSMO2,NCF,Pinet,ISN,DIN_BPR,NCF_DEBIAS,DCN_DEBIAS,MY_MODEL,MMOE,SHARED_BOTTOM,PLE,MY_MODEL_DEBIAS,AESM2,M2M
from trainer import Trainer

def get_model(model_name, model_config, data_config):
    model_name = model_name.lower()
    if model_name == 'lr':
        return LR(model_config, data_config)
    elif model_name == 'dssm':
        return DSSM(model_config, data_config)
    elif model_name == 'wdl':
        return WDL(model_config, data_config)
    elif model_name == 'fm':
        return FM(model_config, data_config)
    elif model_name == 'deepfm':
        return DeepFM(model_config, data_config)
    elif model_name == 'youtubednn':
        return YouTubeDNN(model_config, data_config)
    elif model_name == 'ffm':
        return FFM(model_config, data_config)
    elif model_name == 'pnn':
        return PNN(model_config, data_config)
    elif model_name == 'dcn':
        return DCN(model_config, data_config)
    elif model_name == 'din':
        return DIN(model_config, data_config)
    elif model_name == 'ubr':
        return UBR(model_config, data_config)
    elif model_name == 'cosmo':
        return COSMO(model_config,data_config)
    elif model_name == 'dien':
        return DIEN(model_config,data_config)
    elif model_name == 'gru4rec':
        return GRU4REC(model_config,data_config)
    elif model_name == 'bst':
        return BST(model_config,data_config)
    elif model_name == 'din_session':
        return DIN_SESSION(model_config,data_config)
    elif model_name == 'miss':
        return MISS(model_config,data_config)
    elif model_name == 'cosmo2':
        return COSMO2(model_config,data_config)
    elif model_name == 'ncf':
        return NCF(model_config,data_config)
    elif model_name == 'pinet':
        return Pinet(model_config,data_config)
    elif model_name == 'isn':
        return ISN(model_config,data_config)
    elif model_name == 'din_bpr':
        return DIN_BPR(model_config,data_config)
    elif model_name == 'ncf_debias':
        return NCF_DEBIAS(model_config,data_config)
    elif model_name == 'dcn_debias':
        return DCN_DEBIAS(model_config, data_config)
    elif model_name == 'my_model':
        return MY_MODEL(model_config, data_config)
    elif model_name == 'mmoe':
        return MMOE(model_config, data_config)
    elif model_name == 'shared_bottom':
        return SHARED_BOTTOM(model_config,data_config)
    elif model_name == 'ple':
        return PLE(model_config,data_config)
    elif model_name == 'my_model_debias':
        return MY_MODEL_DEBIAS(model_config,data_config)
    elif model_name == 'aesm2':
        return AESM2(model_config,data_config)
    elif model_name == 'm2m':
        return M2M(model_config,data_config)
    else:
        print('wrong model name: {}'.format(model_name))
        exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='model name', default='deepfm')
    parser.add_argument('-d', '--dataset', type=str, help='dataset name', default='zj')
    args = parser.parse_args()
    print('Training begin...')
    # go to root path
    root_path = '..'
    os.chdir(root_path)
    data_config_path = os.path.join('configs/data_configs', args.dataset + '.yaml')
    train_config_path = os.path.join('configs/train_configs', args.dataset + '.yaml')
    model_config_path = os.path.join('configs/model_configs', args.model + '.yaml')

    loader = get_yaml_loader()
    with open(data_config_path, 'r') as f:
        data_config = yaml.load(f, Loader=loader)
    with open(train_config_path, 'r') as f:
        train_config = yaml.load(f, Loader=loader)
    with open(model_config_path, 'r') as f:
        model_config = yaml.load(f, Loader=loader)[args.dataset]

    run_config = {'model': args.model,
                  'dataset': args.dataset,
                  'train_phase': 'ind'}
    init_seed(train_config['seed'], train_config['reproducibility'])
    # logger initialization
    init_logger(run_config)
    logger = getLogger()

    logger.info('training begin...')
    logger.info(run_config)
    logger.info(train_config)

    # datasets: train and test
    with open(train_config['train_set'], 'rb') as f:
        train_tuple = pkl.load(f)
    with open(train_config['test_set'], 'rb') as f:
        test_tuple = pkl.load(f)
    # if args.dataset == 'Cloud':
    #     train_dl = Dataloader(train_tuple, train_config['train_batch_size'], True)
    #     test_dl = Dataloader(test_tuple, train_config['eval_batch_size'], False)
    # else:
    #     train_dl = DomainDataloader(train_tuple,train_config['train_batch_size'],True)
    #     test_dl = DomainDataloader(test_tuple,train_config['eval_batch_size'],False)
    train_dl = Dataloader(train_tuple, train_config['train_batch_size'], True)
    test_dl = Dataloader(test_tuple, train_config['eval_batch_size'], False)
    
    # get model
    model = get_model(run_config['model'], model_config, data_config).to(train_config['device'])
    logger.info(model)
    # get trainer and fit
    trainer = Trainer(train_config, model, args.dataset)
    # best_eval_result = trainer.fit(train_dl, test_dl)
    if train_config['ckpt_file'] != 'None': 
        best_eval_result = trainer.fit(train_dl, test_dl,ckpt_file = train_config['ckpt_file'])
    else:
        best_eval_result = trainer.fit(train_dl,test_dl)
    # # load best model and test it
    # logger.info('Loading the best model and test...')
    # test_dl.refresh()
    # load_eval_result = trainer.evaluate(test_dl)
    # load_eval_output = set_color('loaded model\'s eval result', 'blue') + ': \n' + dict2str(load_eval_result)
    # logger.info(load_eval_output)