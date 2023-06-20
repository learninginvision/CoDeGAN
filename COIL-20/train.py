import os
import time
import yaml
import json
import torch
import random
import datetime
from   solver import Solver
from   dataloader import getloader
from   test_acc import test_acc
from   utils.visualization import visualize_result


if __name__ == "__main__":
    # load important parameters
    config = yaml.load(open('./param.yml'), Loader=yaml.FullLoader)

    # choose train mode
    pretrain = config['pretrain']

    # save dir
    save_path = './Result/{}/'.format(random.randint(0, 10000))
    model_dir = {"checkpoint":"./checkpoint", "samples_G":"./samples_G"}
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
    solver.Classifier.load_state_dict( torch.load('./coil20_data/Classifier.pth', map_location=solver.device))
    solver.Classifier.eval()

    if pretrain is True:
        solver.EC.load_state_dict(torch.load('./coil20_data/pre_EC.pth', map_location=solver.device))

    print("start...")

    ecloss, ezloss, dloss, gloss = (0, 0, 0, 0)
    for epoch in range(max_step):
        time_start = datetime.datetime.now()
        solver.G.train()
        solver.D.train()
        solver.EC.train()
        solver.EZ.train()

        for idx, (data, label) in enumerate(dataset):

            dloss = solver.optimize_parametersD(d_batch_size, data)

            if idx % critic == 0:
                if epoch >= chain_step-1:
                    gloss  = solver.optimize_parametersG(g_batch_size)
                    ecloss = solver.optimize_parametersEC(ec_batch_size)
                else:
                    gloss  = solver.optimize_parametersG(g_batch_size)

                ezloss = solver.optimize_parametersEZ(ez_batch_size)

        time_end = datetime.datetime.now()

        solver.G.eval()
        solver.EC.eval()
        solver.Classifier.eval()

        # save generated images and models
        if epoch % 50 == 50-1:
            visualize_result(epoch+1, save_path+model_dir['samples_G'], solver.G,\
                    solver.zn_dim, solver.zc_dim, solver.device)

        if epoch % save_step == save_step - 1 and epoch > 1000:
            torch.save(solver.G.state_dict(),  save_path+model_dir['checkpoint'] + "/{}_G.pth".format(epoch  + 1))
            torch.save(solver.D.state_dict(),  save_path+model_dir['checkpoint'] + "/{}_D.pth".format(epoch  + 1))
            torch.save(solver.EZ.state_dict(), save_path+model_dir['checkpoint'] + "/{}_EZ.pth".format(epoch + 1))
            torch.save(solver.EC.state_dict(), save_path+model_dir['checkpoint'] + "/{}_EC.pth".format(epoch + 1))

        # test disentangle performence
        if epoch % test_step == test_step - 1:
            test_acc(solver.G, solver.Classifier, epoch+1, save_path)

        print('[{}/{}] dloss: {:.4f} gloss: {:.4f} ecloss: {:.4f} ezloss: {:.4f} lr: {:.8f}'\
            .format(epoch+1, max_step, dloss, gloss, ecloss, ezloss, solver.lr))
        print("{:.4f} minutes...".format((time_end - time_start).seconds / 60.))        

