import os
import json
import yaml
import torch
import random
import datetime
from   test import test
from   solver import Solver
from   dataloader import getloader
from   utils.util import sample_real_data
from   utils.visualization import visualize_result


if __name__ == "__main__":
    # load important parameters
    config = yaml.load(open('./param.yml'), Loader=yaml.FullLoader)

    # choose train mode
    few_labels = config['few_labels']
    pretrain = config['pretrain']
    meta = config['meta']

    # save dir
    save_path = './Result/{}/'.format(str(random.randint(0, 10000)))
    model_dir = {"checkpoint":"checkpoint", "samples":"samples"}
    for dir_ in model_dir:
        if not os.path.exists(save_path+model_dir[dir_]):
            os.makedirs(save_path+model_dir[dir_])

    # set rand seed
    torch.manual_seed(0)

    # save parameters
    with open(save_path+'train_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # params 
    g_batch_size = config['g_batch_size']
    d_batch_size = config['d_batch_size']
    e_batch_size = config['e_batch_size']
    r_batch_size = config['r_batch_size']

    max_step   = config['max_step'  ]
    save_step  = config['save_step' ]
    test_step  = config['test_step' ]
    chain_step = config['chain_step']

    solver     = Solver(config)
    dataset    = getloader(d_batch_size)

    # load pretrained_model if needed
    if pretrain is True:
        if few_labels is True:
            if meta is True:
                load_path = './fashionmnist_data/meta'
            else:
                load_path = './fashionmnist_data/few_labels'
        else:
            load_path = './fashionmnist_data/unsupervised'
        solver.G.load_state_dict(torch.load(load_path + '/pre_G.pth', map_location=solver.device))
        solver.D.load_state_dict(torch.load(load_path + '/pre_D.pth', map_location=solver.device))
        solver.E.load_state_dict(torch.load(load_path + '/pre_E.pth', map_location=solver.device))

    print("start...")

    # load real labeled data
    if few_labels is True:
        real_datasets = torch.load('./fashionmnist_data/100_real_images.pt')

    ecloss, ezloss, dloss, gloss = (0, 0, 0, 0)
    for epoch in range(max_step):
        time_start = datetime.datetime.now()
        solver.G.train()
        solver.E.train()
        solver.D.train()

        for idx, (data, label) in enumerate(dataset):
    
            if few_labels is True:
                real_data = sample_real_data(r_batch_size, real_datasets)
            else:
                real_data = None

            if epoch >= chain_step:
                gloss = solver.optimize_parametersG(g_batch_size)
                ecloss, ezloss = solver.optimize_parametersE(e_batch_size, real_data)
            else:
                gloss = solver.optimize_parametersG(g_batch_size, real_data)

            dloss = solver.optimize_parametersD(d_batch_size, data)

        time_end = datetime.datetime.now()

        solver.G.eval()
        solver.E.eval()

        # save generated images and models
        visualize_result(epoch+1, save_path+model_dir['samples'], solver.G,\
                solver.zn_dim, solver.zc_dim, solver.device)
        if epoch % save_step == save_step - 1:
            torch.save(solver.G.state_dict(), save_path+model_dir['checkpoint'] + "/{}_G.pth".format(epoch + 1))
            torch.save(solver.D.state_dict(), save_path+model_dir['checkpoint'] + "/{}_D.pth".format(epoch + 1))
            torch.save(solver.E.state_dict(), save_path+model_dir['checkpoint'] + "/{}_E.pth".format(epoch + 1))
        # test disentangle performence
        if epoch % test_step == test_step - 1:
            test(solver.E, epoch+1, save_path)

        print('[{}/{}] dloss:{:.4f} gloss: {:.4f} ecloss: {:.4f} ezloss: {:.4f}'\
            .format(epoch+1, max_step, dloss, gloss, ecloss, ezloss))
        print("{:.4f} minutes...".format((time_end - time_start).seconds / 60.))

