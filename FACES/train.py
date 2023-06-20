import os
import yaml
import json
import torch
import random
import datetime
from   solver import Solver
from   dataloader import getloader
from   utils.util import to_numpy
from   utils.visualization import visualize_result


if __name__ == "__main__":
    # load important parameters
    config = yaml.load(open('./param.yml'), Loader=yaml.FullLoader)

    # choose train mode
    pretrain = config['pretrain']

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

    max_step    = config['max_step']
    save_step   = config['save_step']
    chain_step  = config['chain_step']
    critic      = config['critic']
    real_critic = config['real_critic']

    solver = Solver(config)
    real_data, dataset = getloader(d_batch_size)

    if pretrain is True:
        solver.E.load_state_dict(torch.load('./pretrain_models/pre_E.pth', map_location=solver.device))
        solver.G.load_state_dict(torch.load('./pretrain_models/pre_G.pth', map_location=solver.device))
        solver.D.load_state_dict(torch.load('./pretrain_models/pre_D.pth', map_location=solver.device))

    # training
    print("start...")
    ecloss, ezloss, dloss, gloss = (0, 0, 0, 0)

    for epoch in range(max_step):
        time_start = datetime.datetime.now()
        solver.G.train()
        solver.E.train()
        solver.D.train()

        for idx, (data, label) in enumerate(dataset):
            if idx % critic == 0:
                if epoch >= chain_step-1:
                    gloss = solver.optimize_parametersG(g_batch_size)
                    ecloss, ezloss = solver.optimize_parametersE\
                        (e_batch_size, real_data)
                else:
                    gloss = solver.optimize_parametersG(g_batch_size, real_data)
                dloss = solver.optimize_parametersD(d_batch_size, data)                    

        time_end = datetime.datetime.now()

        solver.G.eval()

        # save generated images and models
        visualize_result(epoch+1, save_path+model_dir['samples'], solver.G,\
            solver.zn_dim, solver.zc_dim, solver.device)
        if epoch % save_step == save_step - 1:
            torch.save(solver.G.state_dict(), save_path+model_dir['checkpoint'] + "/{}_G.pth".format(epoch + 1))
            torch.save(solver.D.state_dict(), save_path+model_dir['checkpoint'] + "/{}_D.pth".format(epoch + 1))
            torch.save(solver.E.state_dict(), save_path+model_dir['checkpoint'] + "/{}_E.pth".format(epoch + 1))

        # save and print loss
        list_strings = []
        current_losses = {'dloss':dloss, 'gloss':gloss, 'ecloss':ecloss, 'ezloss': ezloss}
        for loss_name, loss_value in current_losses.items():
            list_strings.append('{:s} = {:.8f} '.format(loss_name, to_numpy(loss_value)))
        full_string = ' '.join(list_strings)
        with open(save_path+'loss.txt', "a") as f:
            f.write('epoch = {} {} \n'.format(epoch+1, full_string))

        print('[{}/{}] dloss: {:.4f} gloss: {:.4f} ecloss: {:.4f} ezloss: {:.4f}'\
            .format(epoch, max_step, dloss, gloss, ecloss, ezloss))
        print("{:.4f} minutes...".format((time_end - time_start).seconds / 60.))
        
