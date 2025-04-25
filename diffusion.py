import torch
from AIRP_read_data import read_dataset
from torch_geometric.loader import DataLoader
from models import DiffusionEGNN
import torch.nn.functional as F


def forward_diffusion(x, t, Alpha_t, Sigma_t):

    alpha = Alpha_t[t]

    sigma = Sigma_t[t]

    noise_x = torch.randn_like(x)
    noise_x = sub_center(noise_x)

    x_noisy = alpha * x + sigma * noise_x

    return x_noisy, noise_x

def noise_schedule(T, s=1e-5):

    t = torch.arange(T + 1, dtype=torch.float32)
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

def sample_time_step(T):
    t_val = torch.randint(0, T, (1,)).item()
    return t_val

def sub_center(x):

    center_of_mass = x.mean(dim=1, keepdim=True)
    return x - center_of_mass

def train(T, train_loader, val_loader, max_atom_number, num_class, device, dtype):

    lr = 2e-4
    epochs = 10

    generative_model = DiffusionEGNN(number_tokens=None, dim=num_class+1)
    optimizer = get_optim(lr=lr, model=generative_model)

    Alpha_t, Sigma_t = noise_schedule(T)

    for epoch in range(epochs):
        for batch_idx, data in enumerate(train_loader):
            generative_model.train()
            x = data['pos'].to(device, dtype)
            node_mask = data['atom_mask'].to(device).bool()
            # h = data['atomic_numbers'].to(device).long()
            h = data['atomic_numbers_one_hot'].to(device, dtype)

            curr_batch_size = x.size()[0] // max_atom_number

            x = x.view(curr_batch_size, max_atom_number, 3)
            node_mask = node_mask.view(curr_batch_size, max_atom_number, -1)
            h = h.view(curr_batch_size, max_atom_number, -1)

            t_val = sample_time_step(T)
            t_val_tensor = torch.full((x.size(0), x.size(1), 1), t_val / T, device=device)

            h = torch.cat([h, t_val_tensor], dim=-1)

            x = sub_center(x)

            x_noisy, noise_x = forward_diffusion(x, t_val, Alpha_t, Sigma_t)

            _, pos_predict = generative_model(h, noise_x, node_mask)
            noise_predict = pos_predict - x_noisy
            noise_predict = sub_center(noise_predict)

            loss = F.mse_loss(noise_predict * node_mask, noise_x  * node_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.10f}")
                test(generative_model, val_loader, dtype, T, device)
    return generative_model

def test(generative_model, test_loader, dtype, T, device):
    with torch.no_grad():
        generative_model.eval()
        loss = 0

        Alpha_t, Sigma_t = noise_schedule(T)

        for batch_idx, data in enumerate(test_loader):
            x = data['pos'].to(device, dtype)
            node_mask = data['atom_mask'].to(device).bool()
            # h = data['atomic_numbers'].to(device).long()
            h = data['atomic_numbers_one_hot'].to(device, dtype)

            curr_batch_size = x.size()[0] // max_atom_number

            x = x.view(curr_batch_size, max_atom_number, 3)
            node_mask = node_mask.view(curr_batch_size, max_atom_number, -1)
            h = h.view(curr_batch_size, max_atom_number, -1)

            x = sub_center(x)

            forward_steps = 10
            x_noisy, noise_x = forward_diffusion(x, forward_steps, Alpha_t, Sigma_t)

            pos = reverse_diffusion(zT=x_noisy, generative_model=generative_model, T=T, device=device, h=h, node_mask=node_mask,
                                    Alpha=Alpha_t, Sigma=Sigma_t, backward_steps=200)

            loss = loss + F.mse_loss(x * node_mask, pos * node_mask)
        print(f"Test Loss: {loss.item() / len(test_loader):.8f}")


def reverse_diffusion(zT, generative_model, T, device, h, node_mask, Alpha, Sigma, backward_steps):

    z = zT.to(device)

    for t in range(backward_steps, 0, -1):
        # if t % 1 == 0: print(t) # T, T-1, …, 1
        s = t - 1
        a_t, a_s = Alpha[t], Alpha[s]
        s_t, s_s = Sigma[t], Sigma[s]
        a_ts = a_t / a_s
        s2_ts = s_t ** 2 - (a_ts ** 2) * (s_s ** 2)
        s_t_to_s = torch.sqrt(s2_ts) * s_s / s_t

        t_tensor = torch.full((z.size(0), z.size(1), 1), t / T, device=device)

        h_with_step = torch.cat([h, t_tensor], dim=-1)

        # ---- 1. 预测噪声 ε̂_t ----
        # _, noise_predict = model(h, noise_x, node_mask)
        # t_input = torch.full((z.size(0), 1), t / T, device=device)
        _, pos_predict = generative_model(h_with_step, z, node_mask)  # 同维度输出
        noise_predict = pos_predict - z
        noise_predict = sub_center(noise_predict)
        # eps_hat[..., :3] = strip_com(eps_hat[..., :3])  # 坐标部分去重心

        mu = (z - (s2_ts / (a_ts * s_t)) * noise_predict) / a_ts

        if s > 0:
            noise = torch.randn_like(z)
            noise = sub_center(noise)
            z = mu + s_t_to_s * noise
        else:
            z = mu  # t=1→s=0

    return z

def load_model(model_path, device, dtype):

    model = DiffusionEGNN()

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    batch_size = 64
    data_path = 'data/processed'
    max_atom_number = 28
    num_class = 10

    T = torch.tensor(1000)

    train_list, val_list, test_list = read_dataset(data_path, max_atom_number=max_atom_number)
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size)
    test_loader = DataLoader(test_list, batch_size=batch_size)

    model = train(T=T, train_loader = train_loader, val_loader=val_loader,
                  max_atom_number=max_atom_number, num_class=num_class,
                  device=device, dtype=dtype)

    torch.save(model.state_dict(), 'saved_model/gschnet_model.pth')

    model = load_model('saved_model/gschnet_model.pth', device, dtype)

    test(model, test_loader, dtype, T, device)