import os
import yaml
import json
import torch
import random
import datetime
from   test import test
from   solver import Solver
from   dataloader import getloader
from   utils.visualization import visualize_result

if __name__ == "__main__":
    # load important parameters
    config = yaml.load(open('./param.yml'), Loader=yaml.FullLoader)

    # save dir
    save_path = './Result/{}/'.format(str(random.randint(0, 10000)))
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
    g_batch_size = config['g_batch_size']
    d_batch_size = config['d_batch_size']
    e_batch_size = config['e_batch_size']

    max_step   = config['max_step'  ]
    save_step  = config['save_step' ]
    test_step  = config['test_step' ]

    solver     = Solver(config)
    dataset    = getloader(d_batch_size)

    # training
    print("start...")

    for epoch in range(max_step):
        time_start = datetime.datetime.now()
        solver.G.train()
        solver.E.train()
        solver.D.train()

        for idx, (data, label) in enumerate(dataset):
            
            gloss = solver.optimize_parametersG(g_batch_size)
            ecloss, ezloss = solver.optimize_parametersE(e_batch_size)
            dloss = solver.optimize_parametersD(d_batch_size, data)

        solver.G.eval()
        solver.E.eval()
        time_end = datetime.datetime.now()

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


        print('[{}/{}] dloss: {:.4f} gloss: {:.4f} ecloss: {:.4f} ezloss: {:.4f}'\
            .format(epoch+1, max_step, dloss, gloss, ecloss, ezloss))
        print("{:.4f} minutes...".format((time_end - time_start).seconds / 60.))
        