from diffusion import diffusion_main, test, test2
import pickle
import torch
from torch_geometric.loader import DataLoader
from utils import load_model

def test_model(test_loader, device):

    dtype = torch.float32

    max_atom_number = 29
    max_atom_id = 10

    model = load_model('saved_model/gschnet_model.pth', device, dtype, num_class=max_atom_id)

    noise_steps_list = [50, 100, 200, 500, 1000]
    backward_steps_list = [50, 100, 200, 500, 1000]
    
    result_list = []
    for noise_steps in noise_steps_list:
        for backward_steps in backward_steps_list:
            for i in range(0, 3):

                if i==0:
                    T=1000
                    test_loss = test(model, test_loader, dtype, T, device, max_atom_number=max_atom_number, noise_steps=noise_steps, backward_steps=backward_steps)
                    print(f"Test Loss: {test_loss:.10f}")
                elif i==1:
                    T=backward_steps
                    test_loss = test(model, test_loader, dtype, T, device, max_atom_number=max_atom_number, noise_steps=noise_steps, backward_steps=backward_steps)
                    print(f"Test Loss: {test_loss:.10f}")
                else:
                    T=1000
                    test_loss = test2(model, test_loader, dtype, T, device, max_atom_number=max_atom_number, noise_steps=noise_steps, backward_steps=backward_steps)
                    print(f"Test Loss: {test_loss:.10f}")

                # test_loss = test(model, test_loader, dtype, T, device, max_atom_number=max_atom_number, noise_steps=noise_steps, backward_steps=backward_steps)
                # print(f"Test Loss: {test_loss:.10f}")

            result_list.append(test_loss)

    with open('result_list.pkl', 'wb') as f:
        pickle.dump(result_list, f)


# path = "../airp/processed"
path = "data/processed"
diffusion_main(path)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# batch_size = 4096
#
#
# test_list = torch.load('/scratch/wzhao20/airp_saved_data/test_list.pt', weights_only=False, map_location=device)
# test_loader = DataLoader(test_list, batch_size=batch_size)
# test_model(test_loader=test_loader, device=device)


# with open('result_list.pkl', 'rb') as f:
#     result_list = pickle.load(f)

# print(result_list)
