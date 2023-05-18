import argparse
import gc
import os
import shutil
from tqdm import tqdm
import math
import numpy as np
import scipy.io as io
import torch
import torch.optim as optim
import time

import networks
import losses
import utils
from input_prepocessing import input_preparation, resize_images
from sensor import Sensor
from spectral_tools import generate_mtf_variables
from show_results import show


def main_zpnn(args):
    #torch.autograd.set_detect_anomaly(True)
    # Parameters definitions
    test_path = args.input
    method = args.method
    sensor = args.sensor
    out_dir = args.out_dir
    if method.startswith("Faster"):
        epochs_dic = [128, 64, 32, 16]
    elif method.startswith("Fast"):
        epochs_dic = [128, 64, 32, 16, 16]
    else:
        epochs_dic = [100]

    gpu_number = str(args.gpu_number)
    use_cpu = args.use_cpu
    reduce_res_flag = args.RR
    coregistration_flag = args.coregistration
    save_losses_trend_flag = args.save_loss_trend
    show_results_flag = args.show_results
    save_weights_flag = args.save_weights
    from_scratch_flag = args.from_scratch
    min_half_width = args.min_half_width

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

    # Hyperparameters definition
    semi_width = 8

    # Torch configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu")

    # Load test images
    temp = io.loadmat(test_path)

    I_PAN = temp['I_PAN'].astype('float32')
    I_MS = temp['I_MS_LR'].astype('float32')

    # class "Sensor" definition and PNN network definition
    s = Sensor(sensor)

    if 'DRPNN' in method:
        net = networks.DRPNN(s.nbands + 1)
    elif 'PanNet' in method:
        net = networks.PanNet(s.nbands, s.ratio)
    else:
        net = networks.PNN(s.nbands + 1, s.kernels, s.net_scope)

    if args.learning_rate != -1.0:
        s.learning_rate = args.learning_rate

    if 'DRPNN' in method:
        s.learning_rate = 1e-4

    print(s.learning_rate)

    if args.beta != -1.0:
        s.beta = args.beta

    # Wald's Protocol
    if reduce_res_flag:
        I_MS, I_PAN = resize_images(I_MS, I_PAN, s.ratio, s.sensor)

    mtf_kernel, r, c = generate_mtf_variables(s.ratio, sensor, I_PAN, I_MS)

    if not coregistration_flag:
        r = [2]*s.nbands
        c = [2]*s.nbands

    # Input preparation
    I_in = input_preparation(I_MS, I_PAN, s.ratio, s.nbits, s.net_scope)

    # Images reshaping for PyTorch workflow
    I_in = np.moveaxis(I_in, -1, 0)
    I_in = np.expand_dims(I_in, axis=0)
    I_inp = np.copy(I_in)
    I_in = I_in[:, :, s.net_scope:-s.net_scope, s.net_scope:-s.net_scope]

    I_in = torch.from_numpy(I_in).float()
    I_inp = torch.from_numpy(I_inp).float()

    threshold = utils.local_corr_mask(I_in, s.ratio, s.sensor, device, semi_width)
    threshold = threshold.float()

    spec_ref = I_in[:, :-1, :, :]
    struct_ref = torch.unsqueeze(I_in[:, -1, :, :], dim=1)



    # DeZooming implementation
    center = (math.floor(I_in.shape[-2] / 2), math.floor(I_in.shape[-1] / 2))

    if min_half_width > center[0]:
        min_half_width = center[0]

    if min_half_width > center[1]:
        min_half_width = center[1]
    if not method.startswith('Fast'):
        exp_x = [min(center[0], I_in.shape[-2] - center[0])]
        exp_y = [min(center[1], I_in.shape[-1] - center[1])]
    else:
        exp_ini = math.floor(math.log2(min_half_width))
        exp_x_fin = math.ceil(math.log2(center[0]))
        exp_y_fin = math.ceil(math.log2(center[1]))

        exp_x = 2 ** np.linspace(exp_ini, exp_x_fin, 5).astype(int)
        exp_y = 2 ** np.linspace(exp_ini, exp_y_fin, 5).astype(int)

        exp_x = np.clip(exp_x, min_half_width, center[0])
        exp_y = np.clip(exp_y, min_half_width, center[1])

    # Loading of pre-trained weights
    weight_path = 'weights/' + s.sensor + '_' + method + '_model.tar'
    print(weight_path)
    if os.path.exists(weight_path) and not from_scratch_flag:
        net.load_state_dict(torch.load(weight_path))
    else:
        print('Training from scratch will be performed.')

    # Losses definition
    LSpec = losses.SpectralLoss(mtf_kernel, s.ratio, device)
    LStruct = losses.StructuralLoss(s.ratio, device)

    # Fitting strategy definition
    net = net.to(device)
    if 'DRPNN' in method:
        params = []
        base_params = []
        for i, k in net.named_parameters():
            if i == 'Conv_11.weight':
                params.append(k)
            elif i == 'Conv_11.bias':
                params.append(k)
            else:
                base_params.append(k)
        optimizer = optim.Adam([
            {"params": base_params}, {"params": params, "lr": s.learning_rate * 1e-1}],
            lr=s.learning_rate)
    else:
        optimizer = optim.Adam(net.parameters(), lr=s.learning_rate)

    net.train()

    # Moving everything on the device
    I_in = I_in.to(device)
    spec_ref = spec_ref.to(device)
    struct_ref = struct_ref.to(device)
    threshold = threshold.to(device)
    LSpec = LSpec.to(device)
    LStruct = LStruct.to(device)

    total_epochs = sum(epochs_dic)
    # Training
    history_loss = []
    history_loss_spec = []
    history_loss_lambda = []
    history_loss_struct = []

    cut_index = 0


    with tqdm(total=total_epochs) as pbar:
        for epochs in tqdm(epochs_dic):
            if ('DRPNN' in method or 'PanNet' in method):
                s.net_scope = 0
            I_in_1 = I_in[:, :, center[0] - exp_x[cut_index]:center[0] + exp_x[cut_index],
                     center[1] - exp_y[cut_index]:center[1] + exp_y[cut_index]]
            spec_ref_1 = spec_ref[:, :,
                         center[0] - exp_x[cut_index] + s.net_scope:center[0] + exp_x[cut_index] - s.net_scope,
                         center[1] - exp_y[cut_index] + s.net_scope:center[1] + exp_y[cut_index] - s.net_scope]
            struct_ref_1 = struct_ref[:, :,
                           center[0] - exp_x[cut_index] + s.net_scope:center[0] + exp_x[cut_index] - s.net_scope,
                           center[1] - exp_y[cut_index] + s.net_scope:center[1] + exp_y[cut_index] - s.net_scope]
            threshold_1 = threshold[:, :,
                          center[0] - exp_x[cut_index] + s.net_scope:center[0] + exp_x[cut_index] - s.net_scope,
                          center[1] - exp_y[cut_index] + s.net_scope:center[1] + exp_y[cut_index] - s.net_scope]


            for epoch in tqdm(range(epochs), dynamic_ncols=True, initial=1):

                running_loss = 0.0
                running_spec_loss = 0.0
                running_lambda_loss = 0.0
                running_struct_loss = 0.0

                for i in range(I_in.shape[0]):

                    net.train()
                    inputs = I_in_1[i, :, :, :].view([1, I_in_1.size()[1], I_in_1.size()[2], I_in_1.size()[3]])
                    labels_spec = spec_ref_1[i, :, :, :].view(
                        [1, spec_ref_1.size()[1], spec_ref_1.size()[2], spec_ref_1.size()[3]])
                    labels_struct = struct_ref_1[i, :, :, :].view(
                        [1, struct_ref_1.size()[1], struct_ref_1.size()[2], struct_ref_1.size()[3]])

                    optimizer.zero_grad()
                    outputs = net(inputs)

                    loss_struct, loss_struct_no_threshold = LStruct(outputs, labels_struct, threshold_1)


                    loss_spec = LSpec(outputs, labels_spec, r, c)
                    loss = loss_spec + s.beta * loss_struct

                    loss.backward()
                    optimizer.step()
                    #scheduler.step(loss_val)

                    running_loss += loss.item()
                    running_spec_loss += loss_spec.item()
                    running_struct_loss += loss_struct_no_threshold

                history_loss.append(running_loss)
                history_loss_spec.append(running_spec_loss)
                history_loss_lambda.append(running_lambda_loss)
                history_loss_struct.append(running_struct_loss)
                pbar.set_postfix(
                    {'Overall Loss': running_loss, 'Spectral Loss': round(running_spec_loss, 4),
                     'Khan Loss': round(running_lambda_loss, 4), 'Structural Loss': round(running_struct_loss, 4)})
                pbar.update()

            cut_index += 1

    # Testing
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    I_inp = I_inp.to(device)
    net.eval()
    outputs = net(I_inp)

    out = outputs.cpu().detach().numpy()
    out = np.squeeze(out)
    out = np.moveaxis(out, 0, -1)
    out = out * (2 ** s.nbits)
    out = np.clip(out, 0, out.max())

    out = out.astype(np.uint16)
    save_path = os.path.join(out_dir, test_path.split(os.sep)[-1].split('.')[0] + '_' + method + '.mat')
    io.savemat(save_path, {'I_MS': out})

    if save_losses_trend_flag:
        io.savemat(
            os.path.join(out_dir, test_path.split(os.sep)[-1].split('.')[0] + '_' + method + '_losses_trend.mat'),
            {
                'overall_loss': history_loss,
                'spectral_loss': history_loss_spec,
                'D_lambda_loss': history_loss_lambda,
                'structural_loss': history_loss_struct,
            }
        )

    if show_results_flag:
        show(I_MS, I_PAN, out, s.ratio, method)

    torch.cuda.empty_cache()
    gc.collect()
    if save_weights_flag:
        torch.save(net.state_dict(), os.path.join(out_dir, s.sensor + '_' + method + '_' + test_path.split(os.sep)[-1].split('.')[0] + '_model.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Z-PNN',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='Z-PNN is a deep learning algorithm for remote sensing '
                                                 'imagery which performs pansharpening.',
                                     epilog='''\
Reference: 
Pansharpening by convolutional neural networks in the full resolution framework
M. Ciotola, S. Vitale, A. Mazza, G. Poggi, G. Scarpa 

Authors: 
Image Processing Research Group of University Federico II of Naples 
('GRIP-UNINA')
                                     '''
                                     )
    optional = parser._action_groups.pop()
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument("-i", "--input", type=str, required=True,
                               help='The path of the .mat file which contains the MS '
                                    'and PAN images. For more details, please refer '
                                    'to the GitHub documentation.')

    requiredNamed.add_argument('-s', '--sensor', type=str, required=True, choices=["WV3", "WV2", 'GE1'],
                               help='The sensor that has acquired the test image. Available sensors are '
                                    'WorldView-3 (WV3), WorldView-2 (WV2), GeoEye1 (GE1)')

    requiredNamed.add_argument('-m', '--method', type=str, required=True, choices=["Z-PNN", "Z-PanNet",
                                                                                   "Z-DRPNN", "Faster-Z-PNN", "Faster-Z-PanNet", "Faster-Z-DRPNN",
                                                                                   "Fast-Z-PNN", "Fast-Z-PanNet", "Fast-Z-DRPNN"],
                               default="Z-PNN", help='The algorithm with which perform Pansharpening.')

    default_out_path = 'Outputs_Fast/'
    optional.add_argument("-o", "--out_dir", type=str, default=default_out_path,
                          help='The directory in which save the outcome.')
    optional.add_argument('-n_gpu', "--gpu_number", type=int, default=0, help='Number of the GPU on which perform the '
                                                                              'algorithm.')
    optional.add_argument("--use_cpu", action="store_true",
                          help='Force the system to use CPU instead of GPU. It could solve OOM problems, but the '
                               'algorithm will be slower.')
    optional.add_argument("--RR", action="store_true", help='For evaluation only. The algorithm '
                                                            'will be performed at reduced '
                                                            'resolution.')
    optional.add_argument("--coregistration", action="store_true", help="Enable the co-registration feature.")
    optional.add_argument("--save_loss_trend", action="store_true", help="Option to save the trend of losses "
                                                                         "(For Debugging Purpose).")
    optional.add_argument("--show_results", action="store_true", help="Enable the visualization of the outcomes.")
    optional.add_argument("--save_weights", action="store_true", help="Save the training weights.")
    optional.add_argument("-lr", "--learning_rate", type=float, default=-1.0,
                          help='Learning rate with which perform the training.')
    optional.add_argument("-b", "--beta", type=float, default=-1.0,
                          help='Beta value with which to weight the structural loss during the training.')
    optional.add_argument("--from_scratch", action="store_true",
                          help="Train the network from scratch. Enable ReduceLROnPlateau to allow high learning-rates")
    optional.add_argument('-min_width', "--min_half_width", type=int, default=64,
                          help='The minimum crop size for input image.')

    parser._action_groups.append(optional)
    arguments = parser.parse_args()

    main_zpnn(arguments)
