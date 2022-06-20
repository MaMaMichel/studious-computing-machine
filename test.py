from VAE import VAE
from Datasets import DancingDataset
from torch import optim, save, load
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import time


model_params = {"layer_dims": [3, 8, 16, 32, 64, 128, 256],
            "data_size": 128,
            "latent_size": 32,
            "enc_layer_depth": 2,
            "dec_layer_depth": 2}

optim_params = {"lr": 0.001,
                   "betas": (0.9, 0.999),
                   "eps": 1e-08,
                   "weight_decay": 0.2}

training_params = {"batch_size": 32,
                   "training_epochs": 1000,
                   "loss_alpha": 1,
                   "data_path": "../UnethicalSideproject/data/off_crop",
                   "cuda": False,
                   "Sampling_rate": 25,
                   "Save_rate": 500,
                   "Save_path": './models'}

# create model
model = VAE(**model_params)
model = load("./models/VAE_401")
if training_params["cuda"]:
    model.cuda()

# create training set
trainingSet = DancingDataset(training_params["data_path"])

# Create data loader
data_loader = DataLoader(trainingSet, batch_size=training_params["batch_size"],
                        shuffle=True, num_workers=2, drop_last=True)