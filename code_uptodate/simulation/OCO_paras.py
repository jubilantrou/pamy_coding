'''
This function is used to define the related parameters 
for the OCO training procedure.
'''
import argparse

def get_paras():
    parser = argparse.ArgumentParser(description='set parameters for the OCO script')

    # ready for all listed choices :P
    parser.add_argument('--obj', default='real', type=str, choices=['sim','real'], help='use the simulator or the real robot')
    parser.add_argument('--nn_type', default='FCN', type=str, choices=['FCN','CNN'], help='the type of neural networks used for the trainable policy blocks, Fully Connected Neural Network or Convolutional Neural Network')
    parser.add_argument('--h_l', default=10, type=int, help='the extended time points length in the past direction when constructing the training input data')
    parser.add_argument('--h_r', default=40, type=int, help='the extended time points length in the future direction when constructing the training input data')
    parser.add_argument('--ds', default=5, type=int, help='the stride for downsampling when constructing the training input data')    
    parser.add_argument('--filter_size', default=31, type=int, help='the kernel size for the height dimension when using CNN')
    parser.add_argument('--hidden_size', nargs='*', default=[96], type=int, help='the sizes for hidden layers when using FCN')   
    parser.add_argument('--method_updating_traj', default='no_delay', type=str, choices=['no_delay','with_delay'], help='whether to add a delay when updating the trajectory, where with_delay works with a delay of 10 time steps (0.1s)')
    parser.add_argument('--method_updating_policy', default='NM', type=str, choices=['GD','NM'], help='use Gradient Descent or quasi Newton Method to update the trainable parameters')
    parser.add_argument('--lr_list', nargs=3, default=[0.05, 0.05, 0.05], type=float, help='learning rates')
    parser.add_argument('--alpha_list', nargs=3, default=[1e-4, 1e-4, 1e-4], type=float, help='one of the hyperparameters for the Newton Method')
    parser.add_argument('--disturb_alpha_list', nargs=3, default=[1e-4, 1e-4, 1e-4], type=float, help='one of the hyperparameters for the Newton Method used when learning disturbances')
    parser.add_argument('--epsilon_list', nargs=3, default=[5e-2, 5e-2, 5e-2], type=float, help='one of the hyperparameters for the Newton Method')
    parser.add_argument('--disturb_epsilon_list', nargs=3, default=[5e-2, 5e-2, 5e-2], type=float, help='one of the hyperparameters for the Newton Method used when learning disturbances')
    parser.add_argument('--flag_dof', nargs=3, default=[True, True, True], type=bool, help='whether to include each DoF in the training procedure')
    parser.add_argument('--flag_wandb', action='store_false', help='whether to record the training process using wandb')
    parser.add_argument('--flag_time_analysis', action='store_true', help='whether to use pyinstrument to analyse the time consumption of different parts')
    parser.add_argument('--save_path_num', default=503, type=int, help='indicate the sequence number of the file where we restore the results')
    parser.add_argument('--message', default='disturbance + NM + w/o gradually increasing target space for comparison', type=str, help='briefly record the purpose of this experiment')
    parser.add_argument('--seed', default=3154, type=int, help='choose the seed for the reproducibility')
    parser.add_argument('--check_paras', action='store_false', help='remind the user to check the setting of parameters again before running the experiment')
    parser.add_argument('--ff_model', default=None, type=str, help='whether to use pretrained policy blocks as a start')
    parser.add_argument('--pamy_system', default='SISO', type=str, choices=['SISO','MIMO'], help='how to treat different DoFs of the robot, as three seperate SISO systems or one MIMO system')
    parser.add_argument('--fb_input_size', default=None, type=int, help='the lenght of previous differences that will be included in the inputs for a trainable feedback policy')
    # only support the default choice for now :(
    parser.add_argument('--nr_channel', default=1, type=int, help='use only position information for the training input data or use more information such as velocity and acceleration')
    parser.add_argument('--coupling', default='yes', type=str, choices=['yes','no'], help='whether to consider the coupling between dofs when constructing the training input data')
    parser.add_argument('--width', default=3, type=int, help='the width of input data when using CNN')

    paras = parser.parse_args()
    paras.__setattr__('height', int((paras.h_l+paras.h_r)/paras.ds)+1)

    print(format('used paras','=^60'))
    for para in vars(paras):
        print(format(para,'<30'),str(getattr(paras,para)))
    print('='*60)
    
    if paras.check_paras:
        print("Let's go training!")
        return paras
    else:
        raise ValueError("Check the setting of parameters before training! Especially pay attention to the address for storing results and the brief introductive message.")

if __name__=='__main__':
    paras = get_paras()
