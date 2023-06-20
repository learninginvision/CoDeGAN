import os
import yaml
import json
import torch
import random
import datetime
from   solver import Solver
from   dataloader import getloader
from   test.test_acc import test_acc
from   utils.util import sample_real_data
from   utils.visualization import visualize_result


if __name__ == "__main__":
    # load important parameters
    config = yaml.load(open('./param.yml'), Loader=yaml.FullLoader)
    
    # choose train mode
    few_labels = config['few_labels']
    pretrain = config['pretrain']
    meta = config['meta']
    use_ema = config['use_ema']

    # save dir
    save_path = './Result/{}/'.format(random.randint(0, 10000))
    model_dir = {"checkpoint":"./checkpoint", "samples":"./samples"}
    for dir_ in model_dir:
        if not os.path.exists(save_path+model_dir[dir_]):
            os.makedirs(save_path+model_dir[dir_])

    # set rand seed
    torch.manual_seed(0)

    # save parameters
    with open(save_path+'train_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # params
    g_batch_size  = config['g_batch_size']
    d_batch_size  = config['d_batch_size']
    r_batch_size  = config['r_batch_size']
    ec_batch_size = config['ec_batch_size']
    ez_batch_size = config['ez_batch_size']

    critic     = config['critic']
    max_step   = config['max_step']
    test_step  = config['test_step']
    save_step  = config['save_step']
    chain_step = config['chain_step']

    solver     = Solver(config)
    dataset    = getloader(d_batch_size)

    # load pretrain_model if needed
    if pretrain is True:
        if few_labels is True:
            if meta is True:
                load_path = './cifar10_data/meta'
            else:
                load_path = './cifar10_data/few_labels'
        else:
            load_path = './cifar10_data/unsupervised'
        if few_labels is True:
            solver.G.load_state_dict(torch.load(load_path + '/pre_G.pth', map_location=solver.device))
            solver.D.load_state_dict(torch.load(load_path + '/pre_D.pth', map_location=solver.device))
            if use_ema:
                solver.G_EMA.load_state_dict(torch.load(load_path + '/pre_G.pth', map_location=solver.device))
        solver.EC.load_state_dict(torch.load(load_path + '/pre_EC.pth', map_location=solver.device))

    print("start...")

    # load real labeled data
    if few_labels is True:
        real_datasets = torch.load("./cifar10_data/500_real_images.pt")

    ecloss, ezloss, dloss, gloss, ema_iter = (0, 0, 0, 0, 0)
    for epoch in range(max_step):
        time_start = datetime.datetime.now()
        solver.G.train()
        solver.D.train()
        solver.EC.train()
        solver.EZ.train()
        solver.G_EMA.train()

        for idx, (data, label) in enumerate(dataset):
            ema_iter += 1

            if few_labels is True:
                real_data = sample_real_data(r_batch_size, real_datasets)
            else:
                real_data = None

            if idx % critic == 0:
                if epoch >= chain_step-1:
                    gloss  = solver.optimize_parametersG(g_batch_size)
                    ecloss = solver.optimize_parametersEC(ec_batch_size, real_data)
                else:
                    gloss  = solver.optimize_parametersG(g_batch_size, real_data)

                ezloss = solver.optimize_parametersEZ(ez_batch_size)
                if use_ema:
                    solver.ema.update(ema_iter)

            dloss = solver.optimize_parametersD(d_batch_size, data)

            # adjust learning rate
            if idx % 5 == 4:
                _  = solver.adjust_learning_rate(solver.optimizer_G )
                _  = solver.adjust_learning_rate(solver.optimizer_EC)
                _  = solver.adjust_learning_rate(solver.optimizer_EZ)
                lr = solver.adjust_learning_rate(solver.optimizer_D )

        time_end = datetime.datetime.now()

        solver.G.eval()
        solver.EC.eval()

        # save generated images and models
        visualize_result(epoch+1, save_path+model_dir['samples'], solver.G,\
                solver.zn_dim, solver.zc_dim, solver.device)
        if epoch % save_step == save_step - 1:
            torch.save(solver.G.state_dict(),  save_path+model_dir['checkpoint'] + "/{}_G.pth".format(epoch  + 1))
            torch.save(solver.D.state_dict(),  save_path+model_dir['checkpoint'] + "/{}_D.pth".format(epoch  + 1))
            torch.save(solver.EZ.state_dict(), save_path+model_dir['checkpoint'] + "/{}_EZ.pth".format(epoch + 1))
            torch.save(solver.EC.state_dict(), save_path+model_dir['checkpoint'] + "/{}_EC.pth".format(epoch + 1))
        # test disentangle performence
        if epoch % test_step == test_step - 1:
            test_acc(solver.EC, epoch+1, save_path)

        print('[{}/{}] dloss: {:.4f} gloss: {:.4f} ecloss: {:.4f} ezloss: {:.4f} lr: {:.8f}'\
            .format(epoch+1, max_step, dloss, gloss, ecloss, ezloss, lr))
        print("{:.4f} minutes...".format((time_end - time_start).seconds / 60.))        

