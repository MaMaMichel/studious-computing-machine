from VAE import VAE
from Datasets import DancingDataset
from torch import optim, save, load
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import time


model_params = {"layer_dims": [3, 8, 16, 32, 64, 128, 256, 512],
            "data_size": 128,
            "latent_size": 16,
            "enc_layer_depth": 2,
            "dec_layer_depth": 3}

optim_params = {"lr": 0.001,
                   "betas": (0.9, 0.999),
                   "eps": 1e-08,
                   "weight_decay": 0.1}

training_params = {"batch_size": 192,
                   "training_epochs": 2000,
                   "loss_alpha": 1000,
                   "data_path": "../Data/boobs/images_crop",
                   "cuda": True,
                   "Sampling_rate": 30,
                   "Save_rate": 500,
                   "Save_path": './models',
                   "Load_model": False,
                   "Model_path": "./models/VAE_16"}

# create moodel
model = VAE(**model_params)

if training_params["Load_model"]:
    model = load(training_params["Model_path"])

if training_params["cuda"]:
    print('Model: Cuda')
    model.to('cuda')

# define optimizer
optimizer = optim.Adam(model.parameters(), **optim_params)


# create training set
trainingSet = DancingDataset(training_params["data_path"])

# Create data loader
data_loader = DataLoader(trainingSet, batch_size=training_params["batch_size"],
                        shuffle=True, num_workers=4, drop_last=True)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()


# Training Loop
print(f'Starting training for {training_params["training_epochs"]} epochs')

for epoch in range(training_params["training_epochs"]):
    start_time = time.time()
    print(f'\nEpoch {epoch} of {training_params["training_epochs"]}:')


    for counter, batch in enumerate(data_loader):
        batch_time = time.time()
        model.zero_grad()

        input_img = batch['image'].float()

        if training_params["cuda"]:
            input_img = input_img.cuda()

        generated_img, mu, var = model(input_img)

        loss_dict = model.calc_loss(input_img, generated_img, mu, var, alpha=training_params["loss_alpha"])

        loss_dict["loss"].backward()
        optimizer.step()

        # print(input_img[0])
        #
        # print(generated_img[0])
        #
        # print(mu)
        #
        # print(var)
        #
        # print(loss_dict)

        step_count = (counter + epoch*training_params["batch_size"])
        if step_count % training_params["Sampling_rate"] == 0:
            print(f'\rBatch Time: {(time.time() - batch_time):.2f} seconds', end='')


            in_grid = make_grid(input_img[:16])
            out_grid = make_grid(generated_img[:16])
            writer.add_image('output images', out_grid, step_count)
            writer.add_image('input images', in_grid, step_count)
            writer.add_scalar('Loss',  loss_dict["loss"].detach() , step_count)
            writer.add_scalar('MSE_Loss', loss_dict["MSE_Loss"], step_count)
            writer.add_scalar('KLD', loss_dict["KLD"], step_count)

    print(f' Epoch Time: {(time.time() - start_time):.2f} seconds')
    # print(mu)
    # print(var)

    # in_grid = make_grid(input_img)
    # out_grid = make_grid(generated_img)
    # writer.add_image('output images', out_grid, epoch)
    # writer.add_image('input images', in_grid, epoch)
    # writer.add_scalar('Loss', loss_dict["loss"].detach(), epoch)
    # writer.add_scalar('MSE_Loss', loss_dict["MSE_Loss"], epoch)
    # writer.add_scalar('KLD', loss_dict["KLD"], epoch)
    save(model, training_params["Save_path"] + f'/VAE_{epoch}')


writer.close()




