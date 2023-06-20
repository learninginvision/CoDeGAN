from cleanfid import fid
import yaml
import torch
from utils.util import sample_noise, make_batches
import os
from solvers.solver import get_solvers
from torchvision.utils import save_image
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, required= True, 
                    help='Path of the model')
parser.add_argument('--config', type=str, required= True, 
                    help='Path of the config')

# download featrures
# cleanfid.features.get_reference_statistics(name, res, mode="clean", model_name="inception_v3", seed=0, split="test", metric="FID") 

if __name__ == "__main__":
    args = parser.parse_args()
    model_dir = args.model   # the path you save your model
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    solver = get_solvers(config)
    solver.G.load_state_dict(torch.load(model_dir, map_location=solver.device))
    solver.G.to(solver.device)
    solver.G.eval()

    z = sample_noise(batch_size=50000, device=solver.device)[0]
    batches = make_batches(50000, 25)
    fake_images = []
    idx = 0
    for batch_idx, (batch_start, batch_end) in enumerate(batches):
        noise_batch = z[batch_start:batch_end].to(solver.device)
        out = solver.G(noise_batch).detach().cpu()
        for i in range(out.shape[0]):
            save_image(out[0], os.path.join('./test/tmp', '{}.png'.format(i)),normalize = True)
            idx +=1
    del solver
    torch.cuda.empty_cache()
    # FID
    score = fid.compute_fid('./test/tmp', dataset_name=config['dataset'], dataset_res=32, dataset_split="train")
    