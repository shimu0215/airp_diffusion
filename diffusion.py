import torch
import random
import numpy as np
from AIRP_read_data import read_dataset
from torch_geometric.loader import DataLoader
from models import DiffusionEGNN
import torch.nn.functional as F


from torch.utils.data import SubsetRandomSampler
import numpy as np

def get_partial_val_loader(val_dataset_or_loader, fraction=0.01, batch_size=64):

    if isinstance(val_dataset_or_loader, DataLoader):
        val_dataset = val_dataset_or_loader.dataset
    else:
        val_dataset = val_dataset_or_loader

    num_samples = int(len(val_dataset) * fraction)
    indices = np.random.choice(len(val_dataset), num_samples, replace=False)
    sampler = SubsetRandomSampler(indices)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=sampler)
    return val_loader

def forward_diffusion(x, t, Alpha_t, Sigma_t, device='cpu'):

    alpha = Alpha_t[t]

    sigma = Sigma_t[t]

    noise_x = torch.randn_like(x).to(device)
    noise_x = sub_center(noise_x)

    x_noisy = alpha * x + sigma * noise_x

    return x_noisy, noise_x

def noise_schedule(T, s=1e-5, device='cpu'):

    t = torch.arange(T + 1, dtype=torch.float32).to(device)
    f_t = 1 - (t / T) ** 2
    alpha_t = (1 - 2 * s) * f_t + s
    sigma_t = torch.sqrt(1 - alpha_t ** 2)

    return alpha_t, sigma_t

def get_optim(lr, model):
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=lr, amsgrad=True,
        weight_decay=1e-12)

    return optim

def sample_time_step(x, T):
    t_val = torch.randint(
        low=0,
        high=T + 1,
        # low=5,
        # high=6,
        size=(x.size(0), 1, 1),  # 每个graph一个t
        device=x.device
    )
    t_val = t_val.expand(-1, x.size(1), -1)

    return t_val

def sub_center(x):

    center_of_mass = x.mean(dim=1, keepdim=True)
    return x - center_of_mass

def train(T, train_loader, val_loader, max_atom_number, num_class, device, dtype, batch_size):

    print("start training")

    lr = 1e-3
    epochs = 500

    generative_model = DiffusionEGNN(number_tokens=None, dim=num_class+1).to(device)
    optimizer = get_optim(lr=lr, model=generative_model)

    Alpha_t, Sigma_t = noise_schedule(T, device=device)

    best_val_loss = float('inf')
    patience = 50
    counter = 0
    val_fraction_ratio = 0.1

    # val_fraction = get_partial_val_loader(val_loader, fraction=val_fraction_ratio, batch_size=batch_size)
    # initial_val_loss = test(generative_model, val_fraction, dtype, T, device, max_atom_number=max_atom_number)
    # print(f"Initial Val Loss: {initial_val_loss:.10f}")

    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            generative_model.train()
            x = data['pos'].to(device, dtype)
            node_mask = data['atom_mask'].to(device).bool()
            # h = data['atomic_numbers'].to(device).long()
            h = data['atomic_numbers_one_hot'].to(device, dtype)

            # reformat
            curr_batch_size = x.size()[0] // max_atom_number

            x = x.view(curr_batch_size, max_atom_number, 3)
            node_mask = node_mask.view(curr_batch_size, max_atom_number)
            h = h.view(curr_batch_size, max_atom_number, -1)

            # forward diffusion process
            t_val = sample_time_step(x, T)
            t_val_tensor = t_val / T

            h = torch.cat([h, t_val_tensor], dim=-1)

            x = sub_center(x)

            x_noisy, noise_x = forward_diffusion(x, t_val, Alpha_t, Sigma_t, device=device)

            _, pos_predict = generative_model(h, noise_x, mask=node_mask)
            noise_predict = pos_predict - x_noisy
            noise_predict = sub_center(noise_predict)

            node_mask = node_mask.view(curr_batch_size, max_atom_number, -1)
            loss = F.mse_loss(noise_predict * node_mask, noise_x  * node_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.10f}")

                # val_fraction = get_partial_val_loader(val_loader, fraction=val_fraction_ratio, batch_size=batch_size)
                # val_loss = test(generative_model, val_fraction, dtype, T, device, max_atom_number=max_atom_number)
                # print(f"Epoch [{epoch + 1}/{epochs}] Val Loss: {val_loss:.10f}")
                
        val_fraction = get_partial_val_loader(val_loader, fraction=val_fraction_ratio, batch_size=batch_size)
        val_loss = test(generative_model, val_fraction, dtype, T, device, max_atom_number=max_atom_number)
        print(f"Epoch [{epoch + 1}/{epochs}] Val Loss: {val_loss:.10f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(generative_model.state_dict(), 'saved_model/gschnet_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break

    return generative_model

def test(generative_model, test_loader, dtype, T, device, noise_steps=50, max_atom_number=29, backward_steps=100):
    with torch.no_grad():
        generative_model.to(device)
        generative_model.eval()
        loss = 0.0
        initial_loss = 0.0

        Alpha_t, Sigma_t = noise_schedule(T, device=device)

        for batch_idx, data in enumerate(test_loader):
            x = data['pos'].to(device, dtype)
            node_mask = data['atom_mask'].to(device).bool()
            # h = data['atomic_numbers'].to(device).long()
            h = data['atomic_numbers_one_hot'].to(device, dtype)

            curr_batch_size = x.size()[0] // max_atom_number

            x = x.view(curr_batch_size, max_atom_number, 3)
            node_mask = node_mask.view(curr_batch_size, max_atom_number)
            h = h.view(curr_batch_size, max_atom_number, -1)

            x = sub_center(x)

            forward_steps = noise_steps
            x_noisy, noise_x = forward_diffusion(x, forward_steps, Alpha_t, Sigma_t, device=device)

            pos = reverse_diffusion(zT=x_noisy, generative_model=generative_model, T=T, device=device, h=h, node_mask=node_mask,
                                    Alpha=Alpha_t, Sigma=Sigma_t, backward_steps=backward_steps).to(device)

            node_mask = node_mask.view(curr_batch_size, max_atom_number, -1)

            loss = loss + F.mse_loss(x * node_mask, pos * node_mask)
            initial_loss = initial_loss + F.mse_loss(x * node_mask, x_noisy * node_mask)

        loss = loss / len(test_loader)
        initial_loss = initial_loss / len(test_loader)
        print(f"init Loss: {initial_loss:.8f}")

        return loss

def test2(generative_model, test_loader, dtype, T, device, noise_steps=50, max_atom_number=29, backward_steps=100):
    with torch.no_grad():
        generative_model.to(device)
        generative_model.eval()
        loss = 0.0
        initial_loss = 0.0

        Alpha_t, Sigma_t = noise_schedule(T, device=device)

        for batch_idx, data in enumerate(test_loader):
            x = data['pos'].to(device, dtype)
            node_mask = data['atom_mask'].to(device).bool()
            # h = data['atomic_numbers'].to(device).long()
            h = data['atomic_numbers_one_hot'].to(device, dtype)

            curr_batch_size = x.size()[0] // max_atom_number

            x = x.view(curr_batch_size, max_atom_number, 3)
            node_mask = node_mask.view(curr_batch_size, max_atom_number)
            h = h.view(curr_batch_size, max_atom_number, -1)

            x = sub_center(x)

            forward_steps = noise_steps
            x_noisy, noise_x = forward_diffusion(x, forward_steps, Alpha_t, Sigma_t, device=device)

            pos = reverse_diffusion(zT=x_noisy, generative_model=generative_model, T=T, device=device, h=h, node_mask=node_mask,
                                    Alpha=Alpha_t, Sigma=Sigma_t, backward_steps=backward_steps, tmp=True).to(device)

            node_mask = node_mask.view(curr_batch_size, max_atom_number, -1)

            loss = loss + F.mse_loss(x * node_mask, pos * node_mask)
            initial_loss = initial_loss + F.mse_loss(x * node_mask, x_noisy * node_mask)

        loss = loss / len(test_loader)
        initial_loss = initial_loss / len(test_loader)
        print(f"init Loss: {initial_loss:.8f}")

        return loss


def reverse_diffusion(zT, generative_model, T, device, h, node_mask, Alpha, Sigma, backward_steps, tmp=False):

    z = zT.to(device)

    for t in range(backward_steps, 0, -1):
        if tmp:
            t = t * T // backward_steps

        s = t - 1
        a_t, a_s = Alpha[t], Alpha[s]
        s_t, s_s = Sigma[t], Sigma[s]
        a_ts = a_t / a_s
        s2_ts = s_t ** 2 - (a_ts ** 2) * (s_s ** 2)
        s_t_to_s = torch.sqrt(s2_ts) * s_s / s_t

        step = t/T

        t_tensor = torch.full((z.size(0), z.size(1), 1), step, device=device)

        h_with_step = torch.cat([h, t_tensor], dim=-1)

        _, pos_predict = generative_model(h_with_step, z, mask=node_mask)
        noise_predict = pos_predict - z
        noise_predict = sub_center(noise_predict)

        mu = (z - (s2_ts / (a_ts * s_t)) * noise_predict) / a_ts

        if s > 0:
            noise = torch.randn_like(z)
            noise = sub_center(noise)
            z = mu + s_t_to_s * noise
        else:
            z = mu

    return z


def diffusion_main(path):

    # set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # load data settings
    batch_size = 1024
    data_path = path

    # dataset config
    max_atom_number = 29
    max_atom_id = 10

    T = torch.tensor(1000)

    train_list, val_list, test_list = read_dataset(data_path, max_atom_number=max_atom_number, max_atom_id = max_atom_id, train=0.8, val=0.1)

    # load and save data lists
    # save_lists(train_list, val_list, test_list)
    # train_list, val_list, test_list = load_lists(device)

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size)
    test_loader = DataLoader(test_list, batch_size=batch_size)

    # train model
    model = train(T=T, train_loader = train_loader, val_loader=val_loader,
                  max_atom_number=max_atom_number, num_class=max_atom_id,
                  device=device, dtype=dtype, batch_size=batch_size)

    torch.save(model.state_dict(), 'saved_model/gschnet_model.pth')


    # test model
    model = load_model('saved_model/gschnet_model.pth', device, dtype, num_class=max_atom_id)

    test_loss = test(model, test_loader, dtype, T, device, max_atom_number=max_atom_number)
    print(f"Test Loss: {test_loss:.10f}")


if __name__ == '__main__':

    path = "../airp/processed"
    diffusion_main(path)