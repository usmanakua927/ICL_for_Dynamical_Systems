import os
from random import randint
import uuid

from quinine import QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model
from tasks import mean_squared_error, mean_squared_error_state, mean_squared_error_measurement
from tasks import squared_error
import wandb
from scipy.stats import ortho_group
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

torch.backends.cudnn.benchmark = True


def sqrtm(A):
    L = torch.linalg.cholesky(A)
    return torch.linalg.solve(L.T, L)


def train_step_one_step_pred_control_non_linear(model, inputs_batch, outputs_batch, optimizer, loss_func, y_dim=2, discard=True, discard_mode='All', control=True, Non_Linear=False, non_lin_mode=2, state_est=False, input_dim=8):
    optimizer.zero_grad()
    output = model(inputs_batch)
    if state_est:
        dim=input_dim
    else:
        dim=y_dim

    if not Non_Linear or (Non_Linear and (non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==11)):
        if control:
            if not discard:
                loss = loss_func(output, outputs_batch[:,3*(model.state_dim)+1::3,:], dim)
            elif discard and discard_mode=='All':
                loss = loss_func(output, outputs_batch[:, ::3, :], dim)
            elif discard and discard_mode=='AllEM':
                loss = loss_func(output, outputs_batch[:, :, :], dim)
            elif discard and discard_mode=='Noise':
                loss = loss_func(output, outputs_batch[:, 2*model.state_dim::3, :], dim)
        else:
            if not discard:
                loss = loss_func(output, outputs_batch[:,2*(model.state_dim)+1::2,:], dim)
            elif discard and discard_mode=='All':
                loss = loss_func(output, outputs_batch[:, ::2, :], dim)
            elif discard and discard_mode=='AllEM':
                loss = loss_func(output, outputs_batch[:, :, :], dim)
            elif discard and discard_mode=='Noise':
                loss = loss_func(output, outputs_batch[:, model.state_dim::2, :], dim)

    else:
        if control:
            if discard:
                if discard_mode=='AllEM':
                    loss = loss_func(output, outputs_batch[:, :, :], dim)
                else:
                    loss = loss_func(output, outputs_batch[:, ::3, :], dim)
            else:
                if non_lin_mode==1:
                    loss = loss_func(output, outputs_batch[:, 2 * (model.state_dim) + 1::3, :], dim)
                elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                    loss = loss_func(output, outputs_batch[:, 2 * (model.state_dim) + 1+2::3, :], dim)

        else:
            if discard:
                # if  (non_lin_mode==3 or non_lin_mode==4): # Legacy
                if  (non_lin_mode==3):
                    loss = loss_func(output, outputs_batch[:, ::1, :], dim)
                else:
                    if discard_mode=='AllEM':
                        loss = loss_func(output, outputs_batch[:, :, :], dim)
                    else:
                        loss = loss_func(output, outputs_batch[:, ::2, :], dim)

            else:
                if non_lin_mode==1:
                    loss = loss_func(output, outputs_batch[:,  (model.state_dim) + 1::2, :], dim)
                elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                    loss = loss_func(output, outputs_batch[:, (model.state_dim) + 1+2::2, :], dim)

                elif non_lin_mode==3:
                    loss = loss_func(output, outputs_batch[:, 6::1, :], dim)

                # elif non_lin_mode==4: # Legacy
                #     loss = loss_func(output, outputs_batch[:, 7::1, :], y_dim)

    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()

def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds



def Gen_data_One_Step_with_Control_Non_Linear(device='cuda', batch_size=64, input_dim=8, chunk_size=40, w_sigma=1, x_sigma=1, d_curr=8, Dynamic=True, alpha_F=0.0, alpha_Q=0.0, alpha_R=0.0, F_option=2, y_dim=2, discard=True, discard_mode='All', control='True', Non_Linear=False, non_lin_mode=2, non_lin_params=[3,3], calc_CLRB=True, state_est=True):
    
    if not(Non_Linear and non_lin_mode == 3 and batch_size==1):
        calc_CLRB = False

    if non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
        a_nonlin_vec=[];
        b_nonlin_vec=[];

    if non_lin_mode==3:
        del_t=0.1;
        y_pred=[];


    for i in range(batch_size):


        if not Dynamic:
            F = torch.eye(input_dim, dtype=float)
            A_Q = 0.0*torch.eye(input_dim, dtype=float);
        else:

            if not Non_Linear:
                if F_option==2:
                    U=torch.tensor(ortho_group.rvs(input_dim), dtype=float)
                    Sigma_F=torch.diag(torch.rand((input_dim,),dtype=float));
                    F=torch.matmul(torch.matmul(U,Sigma_F), U.T)

                elif F_option==3:
                    U = torch.tensor(ortho_group.rvs(input_dim), dtype=float)
                    Sigma_F = torch.diag(2.*torch.rand((input_dim,), dtype=float)-1.);
                    F = torch.matmul(torch.matmul(U, Sigma_F), U.T)

                else:
                    F = alpha_F*torch.tensor(ortho_group.rvs(input_dim), dtype=float)+(1-alpha_F)*torch.eye(input_dim, dtype=float)

            elif non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==11:
                F = 0.15*torch.randn((input_dim, input_dim), dtype=float)+0.80*torch.eye(input_dim, dtype=float)
            



            if control:
                U_B = torch.tensor(ortho_group.rvs(input_dim), dtype=float)
                Sigma_B = torch.diag(2. * torch.rand((input_dim,), dtype=float) - 1.);
                B = torch.matmul(torch.matmul(U_B, Sigma_B), U_B.T)


            if Non_Linear and (non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==11):
                Q = 0.01 * (torch.eye(input_dim) + 0.1 * torch.randn((input_dim, input_dim)))
                Q = 0.5 * (Q + Q.T) + 1e-5 * torch.eye(input_dim)
            else:
                Sigma_Q_sqrt=torch.sqrt(alpha_Q*torch.diag(torch.rand((input_dim,),dtype=float)))
                Q_U=torch.tensor(ortho_group.rvs(input_dim), dtype=float)
                A_Q=torch.matmul(Q_U, Sigma_Q_sqrt);
                Q=torch.matmul(A_Q,A_Q.T)


        if Non_Linear and non_lin_mode==3:
            noise_var = torch.tensor([0.025*torch.rand(1), 0.000016*torch.rand(1)], dtype=float);
        elif Non_Linear and (non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==11):
            noise_var = 0.01*torch.ones((y_dim, ), dtype=float)
        else:
            noise_var = alpha_R * torch.rand((y_dim,), dtype=float)

        u_t = torch.zeros((input_dim, 1), dtype=float)



        if Non_Linear and non_lin_mode==3:
            while(True):
                q_1=0.1*torch.rand((1));
                q_2=0.00025*torch.rand((1));
                Q = torch.tensor([[q_1/3*del_t**3, q_1/2*del_t**2, 0, 0, 0],[q_1/2*del_t**2, del_t, 0, 0, 0],[0, 0, q_1/3*del_t**3, q_1/2*del_t**2,0],[0, 0, q_1/2*del_t**2, del_t, 0],[ 0, 0, 0, 0, q_2]], dtype=float);

                try:
                    junk=sqrtm(Q);
                    if calc_CLRB:
                        J_k=torch.linalg.inv(Q.float());
                    break;
                except:
                    do_nothing='do_nothing';


            w_t_m_1= torch.matmul(sqrtm(Q).double(),torch.randn((5, 1), dtype=float))+torch.tensor([[0],[10], [0], [-5], [-0.053]],dtype=float)
        else:
            w_t_m_1 = w_sigma * torch.randn((input_dim, 1), dtype=float) + u_t;

        x_t = x_sigma * torch.randn((y_dim, input_dim), dtype=float)
        if d_curr < input_dim:
            x_t[:, (d_curr - 1):] = 0.0;





        if Non_Linear and non_lin_mode==3:
            obs_noise = torch.matmul(torch.diag(torch.sqrt(noise_var)), torch.randn((2, 1), dtype=float))
            y=torch.tensor([[np.sqrt(w_t_m_1[0]**2+w_t_m_1[2]**2)],[np.arctan2(w_t_m_1[2],w_t_m_1[0])]])+obs_noise;
        
        
        else:
            obs_noise = torch.matmul(torch.diag(torch.sqrt(noise_var)), torch.randn((y_dim, 1), dtype=float))
            y = torch.matmul(x_t, w_t_m_1)+obs_noise
        x = torch.reshape(x_t, (y_dim * input_dim, 1))
        u = torch.reshape(u_t, (input_dim, 1));

        w = torch.unsqueeze(w_t_m_1, dim=0);

        if non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
            a_nonlin = 2 * non_lin_params[0] * torch.rand((1,)) - non_lin_params[0]
            b_nonlin = 2 * non_lin_params[1] * torch.rand((1,)) - non_lin_params[1]
            a_nonlin_vec+=[a_nonlin]
            b_nonlin_vec+=[b_nonlin]



        for k in range(int(chunk_size) - 1):
            if Non_Linear and non_lin_mode==3:
                innovation_noise = torch.matmul(Q, torch.randn((5, 1), dtype=float));
            
            elif Non_Linear and (non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==11):
                innovation_noise=torch.unsqueeze(MultivariateNormal(loc=torch.zeros(input_dim,), covariance_matrix=Q).sample(),-1)
            else:
                innovation_noise=torch.matmul(A_Q,torch.randn((input_dim,1), dtype=float));
            if control:
                u_t=torch.randn((input_dim,1), dtype=float)
                u_t=u_t/torch.linalg.vector_norm(torch.squeeze(u_t))
                control_part=torch.matmul(B, u_t);
            else:
                u_t = torch.zeros((input_dim, 1), dtype=float)
                control_part=0.*u_t;
            if not Non_Linear:
                 w_t_m_1 = torch.matmul(F, w_t_m_1)+innovation_noise+control_part;
            else:
                if non_lin_mode==6:
                    w_t_m_1 = torch.matmul(F, torch.tanh(2*w_t_m_1))+innovation_noise+control_part;
                elif non_lin_mode==11:
                    w_t_m_1 = torch.matmul(F, torch.tanh(2*w_t_m_1))+2/9*torch.exp(-(w_t_m_1**2))+innovation_noise+control_part;
                elif non_lin_mode==7:
                    w_t_m_1 = torch.matmul(F, torch.sin(2*w_t_m_1))+innovation_noise+control_part;
                elif non_lin_mode==8:
                    w_t_m_1 = torch.matmul(F, torch.sigmoid(2*w_t_m_1))+innovation_noise+control_part;
                elif non_lin_mode==1:
                     w_t_m_1 =  torch.tanh(w_t_m_1)+innovation_noise+control_part
                elif non_lin_mode==2:
                    w_t_m_1 = a_nonlin*torch.tanh(w_t_m_1*b_nonlin) + innovation_noise + control_part
                
                elif non_lin_mode==9:
                    w_t_m_1 = 0.5*a_nonlin*torch.tanh(w_t_m_1*b_nonlin) + 0.5*torch.sigmoid(w_t_m_1) + innovation_noise + control_part
                elif non_lin_mode==10:
                    w_t_m_1 = a_nonlin*torch.tanh(w_t_m_1*b_nonlin) + 2/9* torch.exp(-((w_t_m_1)**2)) +  innovation_noise + control_part
                elif non_lin_mode==4:
                    w_t_m_1 = a_nonlin*torch.sin(w_t_m_1*b_nonlin) + innovation_noise + control_part
                elif non_lin_mode==5:
                    w_t_m_1 = a_nonlin*torch.sin(w_t_m_1*b_nonlin) + torch.sigmoid(w_t_m_1)+ innovation_noise + control_part
                elif non_lin_mode==3:
                    om = w_t_m_1[-1];

                    x_2 = w_t_m_1[1];
                    x_4 = w_t_m_1[3];

                    f1 = torch.tensor(
                        (del_t * x_2 * np.cos(del_t * om)) / om - (x_4 * (np.cos(del_t * om) - 1)) / om ** 2 - (
                                x_2 * np.sin(del_t * om)) / om ** 2 - (del_t * x_4 * np.sin(del_t * om)) / om);
                    f2 = torch.tensor(- del_t * x_4 * np.cos(del_t * om) - del_t * x_2 * np.sin(del_t * om))
                    f3 = torch.tensor(
                        (x_2 * (np.cos(del_t * om) - 1)) / om ** 2 - (x_4 * np.sin(del_t * om)) / om ** 2 + (
                                del_t * x_4 * np.cos(del_t * om)) / om + (del_t * x_2 * np.sin(del_t * om)) / om)
                    f4 = torch.tensor(del_t * x_2 * np.cos(del_t * om) - del_t * x_4 * np.sin(del_t * om))

                    F_jacob = torch.tensor([[1, np.sin(om * del_t) / om, 0, (np.cos(om * del_t) - 1) / om, f1],
                                            [0, np.cos(om * del_t), 0, -np.sin(om * del_t), f2],
                                            [0, (1 - np.cos(om * del_t)) / om, 1, np.sin(om * del_t) / om, f3],
                                            [0, np.sin(om * del_t), 0, np.cos(om * del_t), f4], [0, 0, 0, 0, 1]],
                                           dtype=float)


                    sinc = np.sin(om * del_t) / om;
                    conc = (np.cos(om * del_t) - 1) / om;
                    w_t_m_1 = torch.matmul(torch.tensor([[1, sinc, 0, conc, 0],
                   [0,np.cos(om * del_t),0, - np.sin(om * del_t),0],
                   [0, - conc, 1, sinc,0],
                   [0, np.sin(om * del_t), 0, np.cos(om * del_t),0],
                    [0,0,0,0,1]]),w_t_m_1) + innovation_noise;



            w = torch.concat((w, torch.unsqueeze(w_t_m_1, dim=0)), dim=0)
            x_t = x_sigma * torch.randn((y_dim, input_dim), dtype=float)
            if d_curr < input_dim:
                x_t[:, (d_curr - 1):] = 0.0;


            if Non_Linear and non_lin_mode==3:
                obs_noise = torch.matmul(torch.diag(torch.sqrt(noise_var)), torch.randn((2, 1), dtype=float))
                y_t = torch.tensor(
                    [[np.sqrt(w_t_m_1[0] ** 2 + w_t_m_1[2] ** 2)], [np.arctan2(w_t_m_1[2], w_t_m_1[0])]]) + obs_noise;


                if calc_CLRB:
                    R=torch.diag(noise_var).float();
                    x_1_minus = w_t_m_1[0];
                    x_3_minus = w_t_m_1[2];

                    H_jacob = torch.tensor([[x_1_minus / (x_1_minus ** 2 + x_3_minus ** 2) ** (1 / 2), 0,
                                             x_3_minus / (x_1_minus ** 2 + x_3_minus ** 2) ** (1 / 2), 0, 0],
                                            [-(0 + x_3_minus) / ((0 + x_3_minus) ** 2 + (0 - x_1_minus) ** 2), 0,
                                             -(0 - x_1_minus) / ((0 + x_3_minus) ** 2 + (0 - x_1_minus) ** 2), 0, 0]],
                                           dtype=float).float()
                    try:
                        J_k = torch.matmul(torch.matmul(H_jacob.T, torch.linalg.pinv(R)), H_jacob) + torch.linalg.pinv(
                            Q.float() + torch.matmul(torch.matmul(F_jacob.float(), J_k), F_jacob.T.float()));


                    except:
                        Reg_Q = 0.1 * torch.max(torch.abs(torch.diag(
                            Q.float() + torch.matmul(torch.matmul(F_jacob.float(), J_k), F_jacob.T.float()))));
                        Reg_R = 0.000000001 * torch.max(torch.abs(torch.diag(R)));
                        try:
                            J_k = torch.matmul(torch.matmul(H_jacob.T, torch.linalg.pinv(
                                R + Reg_R * torch.diag(torch.tensor([1., 1.])))), H_jacob) + torch.linalg.pinv(
                                Q.float() + torch.matmul(torch.matmul(F_jacob.float(), J_k),
                                                         F_jacob.T.float()) + Reg_Q * torch.diag(
                                    torch.tensor([1., 1., 1., 1., 1.])));
                        except:
                            ill_cond_flag = True
                            print("P-INV failed: Discarding Example")
            else:
                obs_noise = torch.matmul(torch.diag(torch.sqrt(noise_var)), torch.randn((y_dim, 1), dtype=float))
                y_t = torch.matmul(x_t, w_t_m_1) + obs_noise

            x=torch.concat((x,torch.reshape(x_t, (y_dim * input_dim, 1))), dim=-1)
            u = torch.concat((u, torch.reshape(u_t, (input_dim, 1))), dim=-1)
            y= torch.concat((y,y_t), dim=-1)

        if Non_Linear and non_lin_mode==3:
            om = w_t_m_1[-1];

            x_2 = w_t_m_1[1];
            x_4 = w_t_m_1[3];

            f1 = torch.tensor(
                (del_t * x_2 * np.cos(del_t * om)) / om - (x_4 * (np.cos(del_t * om) - 1)) / om ** 2 - (
                        x_2 * np.sin(del_t * om)) / om ** 2 - (del_t * x_4 * np.sin(del_t * om)) / om);
            f2 = torch.tensor(- del_t * x_4 * np.cos(del_t * om) - del_t * x_2 * np.sin(del_t * om))
            f3 = torch.tensor(
                (x_2 * (np.cos(del_t * om) - 1)) / om ** 2 - (x_4 * np.sin(del_t * om)) / om ** 2 + (
                        del_t * x_4 * np.cos(del_t * om)) / om + (del_t * x_2 * np.sin(del_t * om)) / om)
            f4 = torch.tensor(del_t * x_2 * np.cos(del_t * om) - del_t * x_4 * np.sin(del_t * om))

            F_jacob = torch.tensor([[1, np.sin(om * del_t) / om, 0, (np.cos(om * del_t) - 1) / om, f1],
                                    [0, np.cos(om * del_t), 0, -np.sin(om * del_t), f2],
                                    [0, (1 - np.cos(om * del_t)) / om, 1, np.sin(om * del_t) / om, f3],
                                    [0, np.sin(om * del_t), 0, np.cos(om * del_t), f4], [0, 0, 0, 0, 1]],
                                   dtype=float)
            sinc = np.sin(om * del_t) / om;
            conc = (np.cos(om * del_t) - 1) / om;
            w_t_m_1 = torch.matmul(torch.tensor([[1, sinc, 0, conc, 0],
                                                  [0, np.cos(om * del_t), 0, - np.sin(om * del_t), 0],
                                                  [0, - conc, 1, sinc, 0],
                                                  [0, np.sin(om * del_t), 0, np.cos(om * del_t), 0],
                                                  [0, 0, 0, 0, 1]], dtype=float), w_t_m_1) + innovation_noise;

            obs_noise = torch.matmul(torch.diag(torch.sqrt(noise_var)), torch.randn((2, 1), dtype=float))
            y_t = torch.tensor(
                [[np.sqrt(w_t_m_1[0] ** 2 + w_t_m_1[2] ** 2)], [np.arctan2(w_t_m_1[2], w_t_m_1[0])]]) + obs_noise;

            y_pred+=[y_t];

            if calc_CLRB:
                x_1_minus = w_t_m_1[0];
                x_3_minus = w_t_m_1[2];

                H_jacob = torch.tensor([[x_1_minus / (x_1_minus ** 2 + x_3_minus ** 2) ** (1 / 2), 0,
                                         x_3_minus / (x_1_minus ** 2 + x_3_minus ** 2) ** (1 / 2), 0, 0],
                                        [-(0 + x_3_minus) / ((0 + x_3_minus) ** 2 + (0 - x_1_minus) ** 2), 0,
                                         -(0 - x_1_minus) / ((0 + x_3_minus) ** 2 + (0 - x_1_minus) ** 2), 0, 0]],
                                       dtype=float).float()

                P_k_p_1_k = Q.float() + torch.matmul(torch.matmul(F_jacob.float(), J_k), F_jacob.T.float());

                CLRB = torch.trace(torch.matmul(torch.matmul(H_jacob, P_k_p_1_k), H_jacob.T) + R);

        if not Non_Linear or (Non_Linear and (non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==11)):
            if control:
                if not discard:
                    inputs = torch.zeros(((input_dim + 1)*y_dim, 3 * chunk_size + 3*input_dim+1))

                    inputs[y_dim:input_dim + y_dim, 0:input_dim] = F.T
                    inputs[y_dim:input_dim + y_dim, input_dim:2*input_dim] = Q.T
                    inputs[0:y_dim, 2*input_dim] = noise_var;
                    inputs[y_dim:input_dim + y_dim, 2*input_dim+1:3*input_dim+1] = B.T

                    inputs[y_dim:y_dim*(input_dim + 1), (3*input_dim+1):(3 * chunk_size + 3*input_dim+1):3] = x
                    inputs[y_dim:input_dim + y_dim, (3 * input_dim + 2):(3 * chunk_size + 3 * input_dim + 1):3] = u
                    inputs[0:y_dim, (3*input_dim + 3):(3 * chunk_size + 3*input_dim+1):3] = y

                elif discard and discard_mode=='All':
                    inputs = torch.zeros(((input_dim + 1) * y_dim, 3 * chunk_size))

                    inputs[y_dim:y_dim * (input_dim + 1), 0:3 * chunk_size:3] = x
                    inputs[y_dim:input_dim + y_dim, 1:3 * chunk_size:3] = u

                    inputs[0:y_dim, 2:3 * chunk_size:3] = y

                
                elif discard and discard_mode=='AllEM':
                    inputs = torch.zeros(((input_dim + 1) * y_dim, chunk_size))
                    inputs[0:y_dim, :] = y
                    
                
                
                
                elif discard and discard_mode=='Noise':
                    inputs = torch.zeros(((input_dim + 1) * y_dim, 3 * chunk_size +2*input_dim))

                    inputs[y_dim:input_dim + y_dim, 0:input_dim] = F.T
                    inputs[y_dim:input_dim + y_dim, input_dim:2*input_dim] = B.T

                    inputs[y_dim:y_dim * (input_dim + 1), 2*input_dim :3*chunk_size+2*input_dim:3] = x

                    inputs[0:y_dim, 2*input_dim+1:3*chunk_size+2*input_dim:3] = u
                    inputs[0:y_dim, 2 * input_dim + 2:3 * chunk_size + 2 * input_dim:3] = y

            else:
                if not discard:
                    inputs = torch.zeros(((input_dim + 1) * y_dim, 2 * chunk_size + 2 * input_dim + 1))

                    inputs[y_dim:input_dim + y_dim, 0:input_dim] = F.T
                    inputs[y_dim:input_dim + y_dim, input_dim:2 * input_dim] = Q.T
                    inputs[0:y_dim, 2 * input_dim] = noise_var;
                    inputs[y_dim:y_dim * (input_dim + 1), (2 * input_dim + 1):(2 * chunk_size + 2 * input_dim + 1):2] = x

                    inputs[0:y_dim, (2 * input_dim + 2):(2 * chunk_size + 2 * input_dim + 1):2] = y

                elif discard and discard_mode == 'All':
                    inputs = torch.zeros(((input_dim + 1) * y_dim, 2 * chunk_size))

                    inputs[y_dim:y_dim * (input_dim + 1), 0:2 * chunk_size:2] = x

                    inputs[0:y_dim, 1:2 * chunk_size:2] = y
                    
                    
                elif discard and discard_mode == 'AllEM':
                    inputs = torch.zeros(((input_dim + 1) * y_dim, chunk_size))
                    inputs[0:y_dim, :] = y

                elif discard and discard_mode == 'Noise':
                    inputs = torch.zeros(((input_dim + 1) * y_dim, 2 * chunk_size + input_dim))

                    inputs[y_dim:input_dim + y_dim, 0:input_dim] = F.T

                    inputs[y_dim:y_dim * (input_dim + 1), input_dim:2 * chunk_size + input_dim:2] = x

                    inputs[0:y_dim, input_dim + 1:2 * chunk_size + input_dim:2] = y
        else:
            if control:
                if discard:
                    inputs = torch.zeros(((input_dim + 1) * y_dim, 3 * chunk_size))

                    inputs[y_dim:y_dim * (input_dim + 1), 0:3 * chunk_size:3] = x
                    inputs[y_dim:input_dim + y_dim, 1:3 * chunk_size:3] = u
                    inputs[0:y_dim, 2:3 * chunk_size:3] = y
                else:
                    if non_lin_mode==1:
                        inputs = torch.zeros(((input_dim + 1) * y_dim, 3 * chunk_size+2*input_dim+1))
                        inputs[y_dim:input_dim + y_dim, 0:input_dim] = B.T
                        inputs[y_dim:input_dim + y_dim, input_dim:2 * input_dim] = Q.T
                        inputs[0:y_dim, 2 * input_dim] = noise_var;

                        inputs[y_dim:y_dim * (input_dim + 1), (2*input_dim+1):(2*input_dim+1+3 * chunk_size):3] = x
                        inputs[y_dim:input_dim + y_dim, (2*input_dim+2):(2*input_dim+1+3 * chunk_size):3] = u

                        inputs[0:y_dim, (2*input_dim+3):(2*input_dim+1+3 * chunk_size):3] = y
                    elif  non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                        inputs = torch.zeros(((input_dim + 1) * y_dim, 3 * chunk_size + 2 * input_dim + 1+2))

                        inputs[y_dim, 0] =a_nonlin
                        inputs[y_dim, 1] =b_nonlin

                        inputs[y_dim:input_dim + y_dim, 2:input_dim+2] = B.T
                        inputs[y_dim:input_dim + y_dim, 2+input_dim:2 * input_dim+2] = Q.T
                        inputs[0:y_dim, 2 * input_dim+2] = noise_var;

                        inputs[y_dim:y_dim * (input_dim + 1),
                        (2 * input_dim + 1+2):(2 * input_dim + 1+2 + 3 * chunk_size):3] = x
                        inputs[y_dim:input_dim + y_dim, (2 * input_dim + 2+2):(2 * input_dim + 1+2 + 3 * chunk_size):3] = u

                        inputs[0:y_dim, (2 * input_dim + 3 + 2 ):(2 * input_dim + 1+2 + 3 * chunk_size):3] = y

            else:
                if discard:

                    if non_lin_mode==3:
                        inputs = torch.zeros((2, chunk_size));
                        inputs[0:, 0:] = y

                    else:
                        
                        if discard_mode=='AllEM':
                            inputs = torch.zeros(((input_dim + 1) * y_dim, chunk_size))
                            inputs[0:y_dim, :] = y
                        else:
                            inputs = torch.zeros(((input_dim + 1) * y_dim, 2 * chunk_size))
                            inputs[y_dim:y_dim * (input_dim + 1), 0:2 * chunk_size:2] = x
                            inputs[0:y_dim, 1:2 * chunk_size:2] = y



                else:
                    if non_lin_mode==1:
                        inputs = torch.zeros(((input_dim + 1) * y_dim, 2 * chunk_size+input_dim+1))
                        inputs[y_dim:input_dim + y_dim, 0:input_dim] = Q.T
                        inputs[0:y_dim, input_dim] = noise_var;
                        inputs[y_dim:y_dim * (input_dim + 1), (input_dim + 1):(2 * chunk_size + input_dim + 1):2] = x
                        inputs[0:y_dim, (input_dim + 2):(2 * chunk_size + input_dim + 1):2] = y

                    elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                        inputs = torch.zeros(((input_dim + 1) * y_dim, 2 * chunk_size + input_dim + 1+2))
                        inputs[y_dim, 0] =a_nonlin
                        inputs[y_dim, 1] =b_nonlin
                        inputs[y_dim:input_dim + y_dim, 2:input_dim+2] = Q.T
                        inputs[0:y_dim, input_dim+2] = noise_var;
                        inputs[y_dim:y_dim * (input_dim + 1), (input_dim + 1+2):(2 * chunk_size + input_dim + 1+2):2] = x
                        inputs[0:y_dim, (input_dim + 2+2):(2 * chunk_size + input_dim + 1+2):2] = y;

                    elif non_lin_mode==3:
                        inputs = torch.zeros((2+5, chunk_size + 5 + 1));
                        inputs[2:, 0:5] = Q.T;
                        inputs[0:2, 5] = noise_var;
                        inputs[0:2, 6:] = y


        
        w=torch.squeeze(w,dim=-1);
        w=w.T;

        if i == 0:
            inputs_batch = torch.unsqueeze(inputs, dim=0)
            w_batch=torch.unsqueeze(w,dim=0)

        else:
            inputs_batch = torch.concat((inputs_batch, torch.unsqueeze(inputs, dim=0)), dim=0)
            w_batch=torch.concat((w_batch, torch.unsqueeze(w, dim=0)), dim=0)

    if not Non_Linear or (Non_Linear and (non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==11)):
        if control:
            if not discard:
                if state_est:
                    outputs_batch = torch.zeros((batch_size, input_dim, 3 * chunk_size + 3 * input_dim + 1))
                    outputs_batch[:, :, (3 * input_dim + 1):(3 * chunk_size + 3 * input_dim + 1):3] = w_batch
                else:
                    outputs_batch = torch.zeros((batch_size, y_dim, 3 * chunk_size + 3 * input_dim + 1))
                    outputs_batch[:, :, (3 * input_dim + 1):(3 * chunk_size + 3 * input_dim + 1):3] = inputs_batch[:, 0:y_dim,
                                                                                                    (3 * input_dim + 3):(
                                                                                                                3 * chunk_size + 3 * input_dim + 1):3]
            elif discard and discard_mode == 'All':
                if state_est:
                    outputs_batch = torch.zeros((batch_size, input_dim, 3 * chunk_size))
                    outputs_batch[:, :, 0:3 * chunk_size:3] = w_batch
                else:
                    outputs_batch = torch.zeros((batch_size, y_dim, 3 * chunk_size))
                    outputs_batch[:, :, 0:3 * chunk_size:3] = inputs_batch[:, 0:y_dim, 2:3 * chunk_size:3]

            
            
            
            elif discard and discard_mode == 'AllEM':
                if state_est:
                    outputs_batch = torch.zeros((batch_size, input_dim, chunk_size))
                    outputs_batch[:, :, 0:] = w_batch
                else:
                    outputs_batch = torch.zeros((batch_size, y_dim, chunk_size-1))
                    outputs_batch[:, :, 0:] = inputs_batch[:, 0:y_dim, 1:]
            elif discard and discard_mode == 'Noise':
                if state_est:
                    outputs_batch = torch.zeros((batch_size, input_dim, 3 * chunk_size + 2*input_dim))
                    outputs_batch[:, :, 2*input_dim:3 * chunk_size + 2*input_dim:3] = w_batch
                else:
                    outputs_batch = torch.zeros((batch_size, y_dim, 3 * chunk_size + 2*input_dim))
                    outputs_batch[:, :, 2*input_dim:3 * chunk_size + 2*input_dim:3] = inputs_batch[:, 0:y_dim,
                                                                                2*input_dim + 2:3 * chunk_size + 2*input_dim:3]
        else:
            if not discard:
                if state_est:
                    outputs_batch = torch.zeros((batch_size,input_dim, 2 * chunk_size + 2*input_dim+1))
                    outputs_batch[:, :, (2*input_dim+1):(2 * chunk_size + 2*input_dim+1):2] = w_batch
                else:
                    outputs_batch = torch.zeros((batch_size,y_dim, 2 * chunk_size + 2*input_dim+1))
                    outputs_batch[:, :, (2*input_dim+1):(2 * chunk_size + 2*input_dim+1):2] = inputs_batch[:, 0:y_dim,
                                                                                        (2*input_dim + 2):(2 * chunk_size + 2*input_dim+1):2]
            elif discard and discard_mode=='All':
                if state_est:
                    outputs_batch = torch.zeros((batch_size, input_dim, 2 * chunk_size))
                    outputs_batch[:, :, 0:2 * chunk_size:2] = w_batch
                else:
                    outputs_batch = torch.zeros((batch_size, y_dim, 2 * chunk_size))
                    outputs_batch[:, :, 0:2 * chunk_size:2] = inputs_batch[:, 0:y_dim,1:2 * chunk_size:2]
                    
            elif discard and discard_mode=='AllEM':
                if state_est:
                    outputs_batch = torch.zeros((batch_size, input_dim, chunk_size))
                    outputs_batch[:, :, 0:] = w_batch
                else:
                    outputs_batch = torch.zeros((batch_size, y_dim, chunk_size-1))
                    outputs_batch[:, :, 0:] = inputs_batch[:, 0:y_dim, 1:]

            elif discard and discard_mode == 'Noise':
                
                if state_est:
                        
                    outputs_batch = torch.zeros((batch_size, input_dim, 2 * chunk_size +input_dim))
                    outputs_batch[:, :, input_dim:2 * chunk_size+input_dim:2] = w_batch
                else:
                    
                    outputs_batch = torch.zeros((batch_size, y_dim, 2 * chunk_size +input_dim))
                    outputs_batch[:, :, input_dim:2 * chunk_size+input_dim:2] = inputs_batch[:, 0:y_dim, input_dim + 1:2 * chunk_size +input_dim:2]
    else:

        if discard:
            if control:
                if  state_est:
                    
                    if discard_mode == 'AllEM':
                        outputs_batch = torch.zeros((batch_size, input_dim, chunk_size))
                        outputs_batch[:, :, 0:] = w_batch
                        
                    else:
                        outputs_batch = torch.zeros((batch_size, input_dim, 3 * chunk_size))
                        outputs_batch[:, :, 0:3 * chunk_size:3] = w_batch
                
                else:
                    if discard_mode == 'AllEM':
                        outputs_batch = torch.zeros((batch_size, y_dim, chunk_size-1))
                        outputs_batch[:, :, 0:] = inputs_batch[:, 0:y_dim, 1:]
                    else:
                        outputs_batch = torch.zeros((batch_size, y_dim, 3 * chunk_size))
                        outputs_batch[:, :, 0:3 * chunk_size:3] = inputs_batch[:, 0:y_dim, 2:3 * chunk_size:3]
            else:

                if non_lin_mode==3:
                    if state_est:
                        raise("Config Not Supported")
                    else:
                        outputs_batch = torch.zeros((batch_size, 2, chunk_size))
                        outputs_batch[:, :, 0:chunk_size-1] = inputs_batch[:, 0:2, 1:chunk_size]
                        for b_num in range(batch_size):
                            outputs_batch[b_num,:,-1]=y_pred[b_num];

                else:
                    if state_est:
                        if discard_mode == 'AllEM':
                            outputs_batch = torch.zeros((batch_size, input_dim, chunk_size))
                            outputs_batch[:, :, 0:] = w_batch
                        else:
                            outputs_batch = torch.zeros((batch_size, input_dim, 2 * chunk_size))
                            outputs_batch[:, :, 0:2 * chunk_size:2] = w_batch
  
                    else:
                        if discard_mode == 'AllEM':
                            outputs_batch = torch.zeros((batch_size, y_dim, chunk_size-1))
                            outputs_batch[:, :, 0:] = inputs_batch[:, 0:y_dim, 1:]
                        else:
                            outputs_batch = torch.zeros((batch_size, y_dim, 2 * chunk_size))
                            outputs_batch[:, :, 0:2 * chunk_size:2] = inputs_batch[:, 0:y_dim, 1:2 * chunk_size:2]

        else:
            if control:
                if non_lin_mode==1:
                    if state_est:
                        outputs_batch = torch.zeros((batch_size, input_dim, 3 * chunk_size+2*input_dim+1))
                        outputs_batch[:, :, (2*input_dim+1):(3 * chunk_size+2*input_dim+1):3] = w_batch
                    else:
                        outputs_batch = torch.zeros((batch_size, y_dim, 3 * chunk_size+2*input_dim+1))
                        outputs_batch[:, :, (2*input_dim+1):(3 * chunk_size+2*input_dim+1):3] = inputs[0:y_dim, (2*input_dim+3):(2*input_dim+1+3 * chunk_size):3]

                elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                    if state_est:
                        outputs_batch = torch.zeros((batch_size, input_dim, 3 * chunk_size + 2 * input_dim + 1+2))
                        outputs_batch[:, :, (2 * input_dim + 1+2):(3 * chunk_size + 2 * input_dim + 1+2):3] = w_batch

                    else:
                        outputs_batch = torch.zeros((batch_size, y_dim, 3 * chunk_size + 2 * input_dim + 1+2))
                        outputs_batch[:, :, (2 * input_dim + 1+2):(3 * chunk_size + 2 * input_dim + 1+2):3] = inputs[0:y_dim, (2 * input_dim + 3+2):(2 * input_dim + 1 + 3 * chunk_size+2):3]

            else:
                if non_lin_mode==1:
                    if state_est:
                        outputs_batch = torch.zeros((batch_size, input_dim, 2 * chunk_size+input_dim+1))
                        outputs_batch[:, :, input_dim+1:2 * chunk_size+input_dim+1:2] = w_batch
                    else:
                        outputs_batch = torch.zeros((batch_size, y_dim, 2 * chunk_size+input_dim+1))
                        outputs_batch[:, :, input_dim+1:2 * chunk_size+input_dim+1:2] = inputs[0:y_dim, (input_dim + 2):(2 * chunk_size + input_dim + 1):2]

                elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                    if state_est:
                        outputs_batch = torch.zeros((batch_size, input_dim, 2 * chunk_size + input_dim + 1+2))
                        outputs_batch[:, :, input_dim + 1+2:2 * chunk_size + input_dim + 1+2:2] = w_batch

                    else:   
                        outputs_batch = torch.zeros((batch_size, y_dim, 2 * chunk_size + input_dim + 1+2))
                        outputs_batch[:, :, input_dim + 1+2:2 * chunk_size + input_dim + 1+2:2] = inputs[0:y_dim, (input_dim + 2+2):(2 * chunk_size + input_dim + 1+2):2]

                elif non_lin_mode==3:
                    if state_est:
                        raise("Config Not Supported")
                    else:
                        outputs_batch = torch.zeros((batch_size, 2, 5+1+chunk_size))
                        outputs_batch[:, :, 6:6+chunk_size-1] = inputs_batch[:, 0:2, 6+1:6+chunk_size]
                        for b_num in range(batch_size):
                            outputs_batch[b_num, :, -1] = y_pred[b_num][:,0];



    inputs_batch = torch.transpose(inputs_batch, 1, 2);
    outputs_batch = torch.transpose(outputs_batch, 1, 2);

    inputs_batch = inputs_batch.to(device)
    outputs_batch = outputs_batch.to(device)


    if  Non_Linear and (non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10):
        return inputs_batch, outputs_batch, a_nonlin_vec, b_nonlin_vec
    else:
        if calc_CLRB:
            return inputs_batch, outputs_batch, CLRB
        else:
            return inputs_batch, outputs_batch




def train_one_step_pred_control_non_linear(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)
    y_dim=curriculum.y_dim
    discard=curriculum.discard
    discard_mode=curriculum.discard_mode
    option=curriculum.option
    control=curriculum.control
    Non_Linear=curriculum.Non_Linear;
    non_lin_mode=curriculum.non_lin_mode;
    non_lin_params=curriculum.non_lin_params
    gpu=curriculum.gpu
    state_est=curriculum.state_est

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:


        if Non_Linear and (non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10):
            Inputs_batch, Outputs_batch,_,__ = Gen_data_One_Step_with_Control_Non_Linear(device='cuda:'+str(gpu), batch_size=bsize,
                                                                                    input_dim=int(
                                                                                        n_dims / model.y_dim - 1),
                                                                                    chunk_size=curriculum.n_points,
                                                                                    w_sigma=1, x_sigma=1,
                                                                                    d_curr=curriculum.n_dims_truncated,
                                                                                    Dynamic=True,
                                                                                    alpha_F=curriculum.F_alpha,
                                                                                    alpha_Q=curriculum.Q_alpha,
                                                                                    alpha_R=curriculum.R_alpha,
                                                                                    F_option=option, y_dim=y_dim,
                                                                                    discard=discard,
                                                                                    discard_mode=discard_mode,
                                                                                    control=control,
                                                                                    Non_Linear=Non_Linear, non_lin_mode=non_lin_mode, non_lin_params=non_lin_params, state_est=state_est)
        # elif Non_Linear and (non_lin_mode==3 or non_lin_mode==4): # Legacy
        elif Non_Linear and (non_lin_mode==3):
            Inputs_batch, Outputs_batch= Gen_data_One_Step_with_Control_Non_Linear(device='cuda:'+str(gpu),
                                                                                           batch_size=bsize,
                                                                                           input_dim=int(
                                                                                               n_dims / model.y_dim - 1),
                                                                                           chunk_size=curriculum.n_points,
                                                                                           w_sigma=1, x_sigma=1,
                                                                                           d_curr=curriculum.n_dims_truncated,
                                                                                           Dynamic=True,
                                                                                           alpha_F=curriculum.F_alpha,
                                                                                           alpha_Q=curriculum.Q_alpha,
                                                                                           alpha_R=curriculum.R_alpha,
                                                                                           F_option=option, y_dim=y_dim,
                                                                                           discard=discard,
                                                                                           discard_mode=discard_mode,
                                                                                           control=control,
                                                                                           Non_Linear=Non_Linear,
                                                                                           non_lin_mode=non_lin_mode, state_est=state_est)

        else:
            Inputs_batch, Outputs_batch=Gen_data_One_Step_with_Control_Non_Linear(device='cuda:'+str(gpu), batch_size=bsize, input_dim=int(n_dims/model.y_dim-1), chunk_size=curriculum.n_points, w_sigma=1, x_sigma=1, d_curr=curriculum.n_dims_truncated, Dynamic=True, alpha_F=curriculum.F_alpha, alpha_Q=curriculum.Q_alpha, alpha_R=curriculum.R_alpha, F_option=option, y_dim=y_dim, discard=discard, discard_mode=discard_mode, control=control, Non_Linear=Non_Linear, non_lin_mode=non_lin_mode, state_est=state_est)


        loss_func = mean_squared_error_measurement

        loss, output = train_step_one_step_pred_control_non_linear(model, Inputs_batch, Outputs_batch, optimizer, loss_func, y_dim=y_dim, discard=discard, discard_mode=discard_mode, control=control, Non_Linear=Non_Linear, non_lin_mode=non_lin_mode, state_est=state_est, input_dim=int(n_dims/model.y_dim-1))

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = squared_error

        if not Non_Linear or (Non_Linear and (non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==11)):
            if control:
                if not discard:
                    point_wise_loss = point_wise_loss_func(output, Outputs_batch[:, 3 * (model.state_dim) + 1::3, :]).mean(
                        dim=0)

                elif discard and discard_mode == 'All':
                    point_wise_loss = point_wise_loss_func(output, Outputs_batch[:, ::3, :]).mean(dim=0)
                    
                elif discard and discard_mode == 'ALLEM':
                    point_wise_loss = point_wise_loss_func(output, Outputs_batch[:, :, :]).mean(dim=0)
                    
                

                elif discard and discard_mode == 'Noise':
                    point_wise_loss = point_wise_loss_func(output, Outputs_batch[:, 2*model.state_dim::3, :]).mean(dim=0)
            else:
                if not discard:
                    point_wise_loss = point_wise_loss_func(output, Outputs_batch[:,2*(model.state_dim)+1::2,:]).mean(dim=0)

                elif discard and discard_mode=='All':
                    point_wise_loss = point_wise_loss_func(output, Outputs_batch[:, ::2, :]).mean(dim=0)
                elif discard and discard_mode=='AllEM':
                    point_wise_loss = point_wise_loss_func(output, Outputs_batch[:, :, :]).mean(dim=0)

                elif discard and discard_mode == 'Noise':
                    point_wise_loss = point_wise_loss_func(output, Outputs_batch[:, model.state_dim::2, :]).mean(dim=0)

        else:
            if control:
                if discard:
                    if discard_mode=='AllEM':
                        point_wise_loss = point_wise_loss_func(output, Outputs_batch[:, :, :]).mean(dim=0)
                    else:
                        point_wise_loss = point_wise_loss_func(output, Outputs_batch[:, ::3, :]).mean(dim=0)
                else:
                    if non_lin_mode==1:
                        point_wise_loss = point_wise_loss_func(output,
                                                               Outputs_batch[:, 2 * (model.state_dim) + 1::3, :]).mean(
                            dim=0)
                    elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                        point_wise_loss = point_wise_loss_func(output,
                                                               Outputs_batch[:, 2 * (model.state_dim) + 1+2::3, :]).mean(
                            dim=0)
            else:
                if discard:
                    # if (non_lin_mode==3 or non_lin_mode==4): # Legacy
                    if (non_lin_mode==3):
                        point_wise_loss = point_wise_loss_func(output, Outputs_batch[:, ::1, :]).mean(dim=0)
                    else:
                        if discard_mode=='AllEM':
                            point_wise_loss = point_wise_loss_func(output, Outputs_batch[:, :, :]).mean(dim=0)
                        else:
                            point_wise_loss = point_wise_loss_func(output, Outputs_batch[:, ::2, :]).mean(dim=0)
                else:
                    if non_lin_mode==1:
                        point_wise_loss = point_wise_loss_func(output,
                                                               Outputs_batch[:, (model.state_dim) + 1::2, :]).mean(
                            dim=0)
                    elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                        point_wise_loss = point_wise_loss_func(output,
                                                               Outputs_batch[:, (model.state_dim) + 1+2::2, :]).mean(
                            dim=0)

                    elif non_lin_mode==3:
                        point_wise_loss = point_wise_loss_func(output,
                                                               Outputs_batch[:, 6::1, :]).mean(
                            dim=0)

                    # elif non_lin_mode==4: # Legacy
                    #     point_wise_loss = point_wise_loss_func(output,
                    #                                            Outputs_batch[:, 7::1, :]).mean(
                    #         dim=0)


        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0 and not args.test_run:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                    "F_alpha": curriculum.F_alpha,
                    "Q_alpha":curriculum.Q_alpha,
                    "R_alpha":curriculum.R_alpha
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0 and not args.test_run:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and not args.test_run
            and i > 0
        ):
            torch.save(model, os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args):

    curriculum = Curriculum(args.training.curriculum)
    y_dim=curriculum.y_dim
    discard=curriculum.discard
    discard_mode=curriculum.discard_mode
    control=curriculum.control
    Non_Linear=curriculum.Non_Linear;
    non_lin_mode=curriculum.non_lin_mode;
    gpu=curriculum.gpu;
    state_est=curriculum.state_est;


    if args.test_run:
        curriculum_args.points.start = curriculum_args.points.end
        curriculum_args.dims.start = curriculum_args.dims.end
        args.training.train_steps = 100
    else:
        wandb.login(key='916e3711e48f46e05df856d9305a511042eb09a0');
        wandb.init(
            dir=args.out_dir,
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=args.__dict__,
            notes=args.wandb.notes,
            name=args.wandb.name,
            resume=True,

        )

    # model = build_model(args.model)
    model = build_model(args.model, y_dim=y_dim, discard=discard, discard_mode=discard_mode, control=control, Non_Linear=Non_Linear, non_lin_mode=non_lin_mode, state_est=state_est)
    model.cuda(gpu)
    model.train()

    # train_mine_SS_innovation_noise_obs_noise_F_options_non_scalar_y(model, args, option=3)
    train_one_step_pred_control_non_linear(model,args)

    if not args.test_run:
        _ = get_run_metrics(args.out_dir)  # precompute metrics for eval


if __name__ == "__main__":
    parser = QuinineArgumentParser(schema=schema)
    args = parser.parse_quinfig()
    assert args.model.family in ["gpt2", "lstm"]
    print(f"Running with: {args}")

    if not args.test_run:
        run_id = args.training.resume_id
        if run_id is None:
            run_id = str(uuid.uuid4())

        out_dir = os.path.join(args.out_dir, run_id)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        args.out_dir = out_dir

        with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
            yaml.dump(args.__dict__, yaml_file, default_flow_style=False)

    main(args)
