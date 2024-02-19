'''
This function is used to define the related parameters 
for the OCO training procedure.
'''
import argparse

def get_paras():
    parser = argparse.ArgumentParser(description='set parameters for the OCO script')

    parser.add_argument('--obj', default='sim', type=str, choices=['sim','real'], help='train the simulator or the real robot')
    parser.add_argument('--nn_type', default='FCN', type=str, choices=['FCN','CNN'], help='the type of neural networks used for the trainable policy blocks')
    parser.add_argument('--nr_channel', default=1, type=int, help='use only p for the training input data or use more information such as v and a')
    parser.add_argument('--h_l', default=90, type=int, help='the extended time points length in the past direction when constructing the training input data')
    parser.add_argument('--h_r', default=90, type=int, help='the extended time points length in the future direction when constructing the training input data')
    parser.add_argument('--ds', default=6, type=int, help='the stride for downsampling when constructing the training input data')
    parser.add_argument('--coupling', default='yes', type=str, choices=['yes','no'], help='whether to consider the coupling between dofs when constructing the training input data')
    parser.add_argument('--width', default=3, type=int, help='the width of input data when using CNN')
    parser.add_argument('--filter_size', default=7, type=int, help='the kernel size for the height dimension when using CNN')
    parser.add_argument('--hidden_size', nargs='*', default=[32], type=int, help='the sizes for hidden layers when using FCN')   
    parser.add_argument('--method_updating_traj', default='no_delay', type=str, choices=['no_delay','with_delay'], help='whether to add a delay when updating the trajectory, where with_delay works with h=10')
    parser.add_argument('--method_updating_policy', default='GD', type=str, choices=['GD','NM'], help='use online gradient descent or online Newton method to update the u_ff policy parameters')
    parser.add_argument('--lr_list', nargs=3, default=[1.0, 0.5, 0.25], type=float, help='learning rates') # 1,1,0.5
    parser.add_argument('--alpha_list', nargs=3, default=[1.0, 1.0, 1.0], type=float, help='one of the hyperparameters for the Newton method')
    parser.add_argument('--epsilon_list', nargs=3, default=[1.0, 1.0, 1.0], type=float, help='one of the hyperparameters for the Newton method')
    parser.add_argument('--flag_dof', nargs=3, default=[False, True, True], type=bool, help='whether to include each dof in the training considering an error in the simulator')
    parser.add_argument('--flag_wandb', action='store_false', help='whether to record the training process using wandb')
    parser.add_argument('--flag_time_analysis', action='store_true', help='whether to use pyinstrument to analyse the time consumption of different parts')
    parser.add_argument('--save_path_num', default=12, type=int, help='indicate the sequence number of the file where we restore results')
    parser.add_argument('--seed', default=5431, type=int, help='choose the seed for the reproducibility')
    parser.add_argument('--check_paras', action='store_true', help='remind the user to check parameters again before running the experiment')

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
        raise ValueError("Check parameters before training! Especially pay attention to the address for results storing.")

if __name__=='__main__':
    paras = get_paras()
