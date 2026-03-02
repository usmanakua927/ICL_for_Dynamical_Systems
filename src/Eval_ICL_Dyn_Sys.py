import torch
import numpy
import matplotlib.pyplot as plt
import models
from scipy.stats import ortho_group
from filterpy.kalman import KalmanFilter
import pickle
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

def sigmoid(x):
  """
  Computes the sigmoid function for a given input.

  Args:
    x: A scalar, vector, or matrix.

  Returns:
    The sigmoid of x.
  """
  return 1 / (1 + np.exp(-x))

def sqrtm(A):
    L = torch.linalg.cholesky(A)
    return torch.linalg.solve(L.T, L)



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



def Stochastic_Gradient_Descent_Regression(inputs_batch, alpha=0.01, device='cuda'):
    chunk_size=int(inputs_batch.shape[-2]/2)
    input_dim=inputs_batch.shape[-1]-1
    batch_size=inputs_batch.shape[0];
    W=torch.zeros((batch_size, input_dim), device =device)
    for i in range(chunk_size-1):
        W=W-2*alpha*(inputs_batch[:,2*i,1:]*torch.unsqueeze(torch.diag(torch.matmul(W,inputs_batch[:,2*i,1:].T)),dim=-1)-inputs_batch[:,2*i,1:]*torch.unsqueeze(inputs_batch[:,2*i+1,0], dim=-1))


    MSE=torch.mean((inputs_batch[:,2*chunk_size-1,0] - torch.diag(torch.matmul(W,inputs_batch[:,2*chunk_size-2,1:].T)))**2)

    return W, MSE


def Stochastic_Gradient_Descent_Regression_explicit(inputs_batch, final_state, alpha=0.01, device='cuda'):
    chunk_size = int(inputs_batch.shape[-2] / 2)
    input_dim = inputs_batch.shape[-1] - 1
    W = torch.zeros((1, input_dim), device=device)

    for i in range(chunk_size):
        W = W - 2 * alpha * (inputs_batch[:, 2 * i, 1:] * torch.unsqueeze(
            torch.diag(torch.matmul(W, inputs_batch[:, 2 * i, 1:].T)), dim=-1) - inputs_batch[:, 2 * i,1:] * torch.unsqueeze(inputs_batch[:, 2 * i + 1, 0], dim=-1))


    MSE=torch.sqrt(sum(torch.squeeze((W.T-final_state)**2, dim=-1))).cpu()

    return W, MSE



def Stochastic_Gradient_Descent_Regression_one_step(inputs_batch, alpha=0.01, device='cuda'):
    chunk_size = int(inputs_batch.shape[-2] / 2)
    input_dim = inputs_batch.shape[-1] - 1
    W = torch.zeros((1, input_dim), device=device)

    for i in range(chunk_size-1):
        W = W - 2 * alpha * (inputs_batch[:, 2 * i, 1:] * torch.unsqueeze(
            torch.diag(torch.matmul(W, inputs_batch[:, 2 * i, 1:].T)), dim=-1) - inputs_batch[:, 2 * i,1:] * torch.unsqueeze(inputs_batch[:, 2 * i + 1, 0], dim=-1))

    out = torch.matmul(W, inputs_batch[:, -2, 1:].T)
    # MSE=torch.sqrt(sum(torch.squeeze((W.T-final_state)**2, dim=-1))).cpu()
    MSE=(out-inputs_batch[:,-1,0])**2

    return out, MSE.cpu()

def Stochastic_Gradient_Descent_Regression_one_step_non_scalar(inputs_batch, alpha=0.01, device='cuda', y_dim=2):
    chunk_size = int(inputs_batch.shape[-2] / 2)
    input_dim = int((inputs_batch.shape[-1])/y_dim-1)

    W = torch.zeros((1, input_dim), device=device)



    for i in range(chunk_size-1):
        X = torch.reshape(inputs_batch[:, 2 * i, y_dim:], (1, y_dim, input_dim))
        Y = torch.reshape(inputs_batch[:, 2 * i+1, :y_dim], (1,y_dim,1))
        W=W-2*alpha*(torch.squeeze(torch.matmul(torch.matmul(torch.transpose(X,1,2),X), torch.unsqueeze(W, dim=-1)), dim=-1) - torch.squeeze(torch.matmul(torch.transpose(X,1,2), Y), dim=-1) )

    X = torch.reshape(inputs_batch[:, -2, y_dim:], (1, y_dim, input_dim))
    out = torch.squeeze(torch.matmul(X, torch.unsqueeze(W, dim=-1)), -1)


    # MSE=torch.sqrt(sum(torch.squeeze((W.T-final_state)**2, dim=-1))).cpu()
    MSE=(out-inputs_batch[:,-1,:y_dim])**2

    return out

def Stochastic_Gradient_Descent_Regression_one_step_non_scalar_control_target_track(inputs_batch, alpha=0.01, device='cuda', y_dim=2, control=False, chunk_size=40):

    W = torch.tensor([[1, 10, 1, -5, -0.053]], device=device)



    for i in range(chunk_size-1):
        a=W[0,0].cpu();
        b=W[0,2].cpu();
        X = torch.tensor([[[a/np.sqrt(a**2+b**2),0,b/np.sqrt(a**2+b**2),0,0],[-b/(a**2+b**2),0,-a/(a**2+b**2),0,0]]], device=device)
        Y = torch.reshape(inputs_batch[:, i, :2], (1,y_dim,1))
        W=W-2*alpha*(torch.squeeze(torch.matmul(torch.matmul(torch.transpose(X,1,2),X), torch.unsqueeze(W, dim=-1)), dim=-1) - torch.squeeze(torch.matmul(torch.transpose(X,1,2), Y), dim=-1) )

    a = W[0, 0].cpu();
    b = W[0, 2].cpu();
    X = torch.tensor([[[a / np.sqrt(a ** 2 + b ** 2), 0, b / np.sqrt(a ** 2 + b ** 2), 0, 0],
                       [-b / (a ** 2 + b ** 2), 0, -a / (a ** 2 + b ** 2), 0, 0]]], device=device)
    out = torch.squeeze(torch.matmul(X, torch.unsqueeze(W, dim=-1)), -1)


    # # MSE=torch.sqrt(sum(torch.squeeze((W.T-final_state)**2, dim=-1))).cpu()
    # MSE=(out-inputs_batch[:,-1,:y_dim])**2

    return out


def Stochastic_Gradient_Descent_Regression_one_step_non_scalar_control(inputs_batch, alpha=0.01, device='cuda', y_dim=2, control=False, state_est=False):
    if control:
        skip=3
    else:
        skip=2
    chunk_size = int(inputs_batch.shape[-2] / skip)
    input_dim = int((inputs_batch.shape[-1])/y_dim-1)

    W = torch.zeros((1, input_dim), device=device)

    
    time_steps=chunk_size-1


    for i in range(time_steps):
        X = torch.reshape(inputs_batch[:, skip * i, y_dim:], (1, y_dim, input_dim))
        Y = torch.reshape(inputs_batch[:, skip * i+(skip-1), :y_dim], (1,y_dim,1))
        W=W-2*alpha*(torch.squeeze(torch.matmul(torch.matmul(torch.transpose(X,1,2),X), torch.unsqueeze(W, dim=-1)), dim=-1) - torch.squeeze(torch.matmul(torch.transpose(X,1,2), Y), dim=-1) )


    if not state_est:
        X = torch.reshape(inputs_batch[:, -skip, y_dim:], (1, y_dim, input_dim))
    else:
        X=torch.eye(input_dim)
        X=torch.unsqueeze(X, dim=0).to(inputs_batch.device)
    out = torch.squeeze(torch.matmul(X, torch.unsqueeze(W, dim=-1)), -1)


    # # MSE=torch.sqrt(sum(torch.squeeze((W.T-final_state)**2, dim=-1))).cpu()
    # MSE=(out-inputs_batch[:,-1,:y_dim])**2

    return out



def Ridge_Regression(inputs_batch, lambda_=0.1, device='cuda'):
    chunk_size=int(inputs_batch.shape[-2]/2)
    input_dim=inputs_batch.shape[-1]-1
    batch_size=inputs_batch.shape[0];
    W = torch.zeros((batch_size, input_dim), device=device)

    for i in range(batch_size):
        X=inputs_batch[i, 0:2 * chunk_size - 2:2, 1:]
        Y=inputs_batch[i, 1:2 * chunk_size - 1:2, 0].T
        try:
            inverse_term=torch.linalg.inv(torch.matmul(X.T,X)+lambda_*torch.eye(input_dim, device=device))
        except:
            continue

        cross_term=torch.matmul(X.T,Y);
        W[i,:]=torch.matmul(inverse_term, cross_term)

    MSE = torch.mean((inputs_batch[:, 2 * chunk_size - 1, 0] - torch.diag(torch.matmul(W, inputs_batch[:, 2 * chunk_size - 2, 1:].T))) ** 2)
    return W, MSE

def Ridge_Regression_explicit(inputs_batch, final_state, lambda_=0.1, device='cuda'):
    chunk_size=int(inputs_batch.shape[-2]/2)
    input_dim=inputs_batch.shape[-1]-1
    batch_size=inputs_batch.shape[0];
    W = torch.zeros((batch_size, input_dim), device=device)

    for i in range(batch_size):
        X=inputs_batch[i, 0:2 * chunk_size:2, 1:]
        Y=inputs_batch[i, 1:2 * chunk_size:2, 0].T
        try:
            inverse_term=torch.linalg.inv(torch.matmul(X.T,X)+lambda_*torch.eye(input_dim, device=device))
        except:
            continue

        cross_term=torch.matmul(X.T,Y);
        W[i,:]=torch.matmul(inverse_term, cross_term)

    MSE=torch.sqrt(sum(torch.squeeze((W.T-final_state)**2, dim=-1))).cpu()
    return W, MSE


def Ridge_Regression_one_step(inputs_batch, lambda_=0.1, device='cuda'):
    chunk_size=int(inputs_batch.shape[-2]/2)
    input_dim=inputs_batch.shape[-1]-1
    batch_size=inputs_batch.shape[0];
    W = torch.zeros((batch_size, input_dim), device=device)

    for i in range(batch_size):
        X=inputs_batch[i, 0:2 * chunk_size-2 :2, 1:]
        Y=inputs_batch[i, 1:2 * chunk_size-1:2, 0].T
        try:
            inverse_term=torch.linalg.inv(torch.matmul(X.T,X)+lambda_*torch.eye(input_dim, device=device))
        except:
            continue

        cross_term=torch.matmul(X.T,Y);
        W[i,:]=torch.matmul(inverse_term, cross_term)

    out = torch.matmul(W, inputs_batch[:, -2, 1:].T)
    # MSE=torch.sqrt(sum(torch.squeeze((W.T-final_state)**2, dim=-1))).cpu()
    MSE=(out-inputs_batch[:,-1,0])**2

    return out, MSE.cpu()


def Ridge_Regression_one_step_non_scalar(inputs_batch, lambda_=0.1, device='cuda', y_dim=2):
    chunk_size=int(inputs_batch.shape[-2]/2)
    input_dim=int((inputs_batch.shape[-1])/y_dim-1)
    batch_size=inputs_batch.shape[0];



    for k in range(chunk_size-1):
       if k==0:
           X = torch.reshape(inputs_batch[:, 2 * k, y_dim:], (1, y_dim, input_dim))
           Y = torch.reshape(inputs_batch[:, 2 * k+1, :y_dim], (1, y_dim, 1))
       else:
           X_k = torch.reshape(inputs_batch[:, 2 * k, y_dim:], (1, y_dim, input_dim))
           Y_k = torch.reshape(inputs_batch[:, 2 * k + 1, :y_dim], (1, y_dim, 1))
           X=torch.cat((X,X_k), dim=1)
           Y = torch.cat((Y, Y_k), dim=1)


    X=torch.squeeze(X,dim=0)
    Y = torch.squeeze(Y, dim=0)

    inverse_term=torch.linalg.inv(torch.matmul(X.T,X)+lambda_*torch.eye(input_dim, device=device))



    cross_term=torch.matmul(X.T,Y);
    W=torch.matmul(inverse_term, cross_term)
    X = torch.reshape(inputs_batch[:, -2, y_dim:], (y_dim, input_dim))

    out = torch.unsqueeze(torch.matmul( X, W), dim=0)

    return out

def Ridge_Regression_one_step_non_scalar_control(inputs_batch, lambda_=0.1, device='cuda', y_dim=2, control=False, state_est=False):
    if control:
        skip=3
    else:
        skip=2

    chunk_size=int(inputs_batch.shape[-2]/skip)
    input_dim=int((inputs_batch.shape[-1])/y_dim-1)
    batch_size=inputs_batch.shape[0];

    
    time_steps=chunk_size-1

    for k in range(time_steps):
       if k==0:
           X = torch.reshape(inputs_batch[:, skip * k, y_dim:], (1, y_dim, input_dim))
           Y = torch.reshape(inputs_batch[:, skip * k+(skip-1), :y_dim], (1, y_dim, 1))
       else:
           X_k = torch.reshape(inputs_batch[:, skip * k, y_dim:], (1, y_dim, input_dim))
           Y_k = torch.reshape(inputs_batch[:, skip * k + (skip-1), :y_dim], (1, y_dim, 1))
           X=torch.cat((X,X_k), dim=1)
           Y = torch.cat((Y, Y_k), dim=1)


    X=torch.squeeze(X,dim=0)
    Y = torch.squeeze(Y, dim=0)

    inverse_term=torch.linalg.inv(torch.matmul(X.T,X)+lambda_*torch.eye(input_dim, device=device))



    cross_term=torch.matmul(X.T,Y);
    W=torch.matmul(inverse_term, cross_term)
    if not state_est:
        X = torch.reshape(inputs_batch[:, -skip, y_dim:], (y_dim, input_dim))
    else:
        X=torch.eye(input_dim).to(inputs_batch.device)
    out = torch.unsqueeze(torch.matmul( X, W), dim=0)

    return out




def perform_kalman_filtering(h,y, F, input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0):
    f = KalmanFilter(dim_x=input_dim, dim_z=1)

    f.R=R.cpu().numpy()
    f.F=F.cpu()
    f.Q=Q.cpu().numpy();
    for i in range(chunk_size-1):
        f.H=torch.unsqueeze(h[2*i,1:], dim=0).cpu()
        f.predict()
        y_i=torch.unsqueeze(y[2*i], dim=0).cpu()
        f.update(y_i)

    f.predict()
    x=torch.tensor(f.x);
    h=torch.unsqueeze(h[2*chunk_size-2,1:], dim=0).cpu()

    out=torch.matmul(h.float(),x.float());
    MSE=abs(y[2*chunk_size-2].cpu()-out);
    return out,MSE




def perform_kalman_filtering_non_scalar(h,y, F, input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2):
    f = KalmanFilter(dim_x=input_dim, dim_z=y_dim)

    f.R=R.cpu().numpy()
    f.F=F.cpu()
    f.Q=Q.cpu().numpy();
    for i in range(chunk_size-1):
        # f.H=torch.unsqueeze(h[2*i,y_dim:], dim=0).cpu()
        f.H=torch.reshape(h[2*i,y_dim:], (y_dim, input_dim)).cpu()
        f.predict()
        y_i=torch.unsqueeze(y[2*i,:], dim=-1).cpu()
        f.update(y_i)

    f.predict()
    x=torch.tensor(f.x);
    # h=torch.unsqueeze(h[2*chunk_size-2,1:], dim=0).cpu()
    h=torch.reshape(h[2*chunk_size-2, y_dim:], (y_dim, input_dim)).cpu()

    out=torch.matmul(h.float(),x.float());

    return out

def perform_kalman_filtering_non_scalar_control(h,y,F, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, state_est=False):
    f = KalmanFilter(dim_x=input_dim, dim_z=y_dim, dim_u=input_dim)

    f.R=R.cpu().numpy()
    f.F=F.cpu()
    f.Q=Q.cpu().numpy();
    if control:
        f.B=B.cpu().numpy()
        skip=3
    else:
        skip=2

    time_steps=chunk_size-1;

    for i in range(time_steps):
        # f.H=torch.unsqueeze(h[2*i,y_dim:], dim=0).cpu()
        f.H=torch.reshape(h[skip*i,y_dim:], (y_dim, input_dim)).cpu()
        if control:
            u=torch.reshape(h[skip * i+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
            f.predict(u);
        else:
            f.predict()
        y_i=torch.unsqueeze(y[skip*i,:], dim=-1).cpu()
        f.update(y_i)

    
    if control:
        u = torch.reshape(h[skip*chunk_size-skip+1, y_dim: input_dim + y_dim], (input_dim, 1)).cpu()
        f.predict(u)
    else:
        f.predict()
    x=torch.tensor(f.x);
        # h=torch.unsqueeze(h[2*chunk_size-2,1:], dim=0).cpu()
    
    if not state_est:
        h=torch.reshape(h[skip*chunk_size-skip, y_dim:], (y_dim, input_dim)).cpu()

        out=torch.matmul(h.float(),x.float());
    else:
        out=torch.tensor(x.float())


    return out


def perform_Extened_Kalman_filtering_non_scalar_control_Tanh(h,y, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False):
    # f = KalmanFilter(dim_x=input_dim, dim_z=y_dim, dim_u=input_dim)

    R=R.cpu()
    Q=Q.cpu();
    if control:
        B=B.cpu()
        skip=3
    else:
        skip=2

    x=torch.zeros((input_dim,1))
    P=torch.eye(input_dim)


    for i in range(chunk_size-1):
        # f.H=torch.unsqueeze(h[2*i,y_dim:], dim=0).cpu()
        H=torch.reshape(h[skip*i,y_dim:], (y_dim, input_dim)).cpu()
        F=torch.diag(torch.squeeze(1-torch.tanh(x)**2))
        if control:
            u=torch.reshape(h[skip * i+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
            x_minus=torch.tanh(x)+torch.matmul(B,u)
        else:
            x_minus = torch.tanh(x)

        P_minus = torch.matmul(torch.matmul(F, P), F.T) + Q;
        y_i=torch.unsqueeze(y[skip*i,:], dim=-1).cpu()

        S=torch.matmul(torch.matmul(H,P_minus),H.T)+R;
        K=torch.matmul(torch.matmul(P_minus,H.T), torch.inverse(S));
        x=x_minus+torch.matmul(K,y_i-torch.matmul(H,x_minus));
        P=torch.matmul(torch.eye(input_dim)-torch.matmul(K,H),P_minus);



    if control:
        u = torch.reshape(h[skip*chunk_size-skip+1, y_dim: input_dim + y_dim], (input_dim, 1)).cpu()
        x_minus=torch.tanh(x)+torch.matmul(B,u)
    else:
        x_minus=torch.tanh(x)
    x=torch.tensor(x_minus);
    # h=torch.unsqueeze(h[2*chunk_size-2,1:], dim=0).cpu()
    h=torch.reshape(h[skip*chunk_size-skip, y_dim:], (y_dim, input_dim)).cpu()
    out=torch.matmul(h.float(),x.float());
    return out


def perform_Extened_Kalman_filtering_non_scalar_control_Tanh_mode_2(h,y, a, b, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, u=None):
    # f = KalmanFilter(dim_x=input_dim, dim_z=y_dim, dim_u=input_dim)

    R=R.cpu()
    Q=Q.cpu();
    if control:
        B=B.cpu()
        skip=3
    else:
        skip=2

    x=torch.zeros((input_dim,1))
    P=torch.eye(input_dim)

    


    for i in range(chunk_size-1):
        # f.H=torch.unsqueeze(h[2*i,y_dim:], dim=0).cpu()
        H=torch.reshape(h[skip*i,y_dim:], (y_dim, input_dim)).cpu()
        F=torch.diag(torch.squeeze(a*b*(1-torch.tanh(x*b)**2)))
        if control:
            u=torch.reshape(h[skip * i+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
            x_minus=a*torch.tanh(x*b)+torch.matmul(B,u)
        else:
            x_minus = a*torch.tanh(x*b)

        P_minus = torch.matmul(torch.matmul(F, P), F.T) + Q;
        y_i=torch.unsqueeze(y[skip*i,:], dim=-1).cpu()

        S=torch.matmul(torch.matmul(H,P_minus),H.T)+R;
        K=torch.matmul(torch.matmul(P_minus,H.T), torch.inverse(S));
        x=x_minus+torch.matmul(K,y_i-torch.matmul(H,x_minus));
        P=torch.matmul(torch.eye(input_dim)-torch.matmul(K,H),P_minus);



    if control:
        u = torch.reshape(h[skip*chunk_size-skip+1, y_dim: input_dim + y_dim], (input_dim, 1)).cpu()
        x_minus=a*torch.tanh(x*b)+torch.matmul(B,u)
    else:
        x_minus=a*torch.tanh(x*b)
    x=torch.tensor(x_minus);
    # h=torch.unsqueeze(h[2*chunk_size-2,1:], dim=0).cpu()
    h=torch.reshape(h[skip*chunk_size-skip, y_dim:], (y_dim, input_dim)).cpu()
    out=torch.matmul(h.float(),x.float());
    return out


def perform_Extened_Kalman_filtering_non_scalar_control_Tanh_mode_10(h,y, a, b, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, u=None):
    # f = KalmanFilter(dim_x=input_dim, dim_z=y_dim, dim_u=input_dim)

    R=R.cpu()
    Q=Q.cpu();
    if control:
        B=B.cpu()
        skip=3
    else:
        skip=2

    x=torch.zeros((input_dim,1))
    P=torch.eye(input_dim)

    


    for i in range(chunk_size-1):
        # f.H=torch.unsqueeze(h[2*i,y_dim:], dim=0).cpu()
        H=torch.reshape(h[skip*i,y_dim:], (y_dim, input_dim)).cpu()
        F=torch.diag(torch.squeeze(a*b*(1-torch.tanh(x*b)**2)-2*(2/9)*x*torch.exp(-x**2)))
        if control:
            u=torch.reshape(h[skip * i+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
            x_minus=a*torch.tanh(x*b)+2/9* torch.exp(-((x)**2))+torch.matmul(B,u)
        else:
            x_minus = a*torch.tanh(x*b)+2/9* torch.exp(-((x)**2))

        P_minus = torch.matmul(torch.matmul(F, P), F.T) + Q;
        y_i=torch.unsqueeze(y[skip*i,:], dim=-1).cpu()

        S=torch.matmul(torch.matmul(H,P_minus),H.T)+R;
        K=torch.matmul(torch.matmul(P_minus,H.T), torch.inverse(S));
        x=x_minus+torch.matmul(K,y_i-torch.matmul(H,x_minus));
        P=torch.matmul(torch.eye(input_dim)-torch.matmul(K,H),P_minus);



    if control:
        u = torch.reshape(h[skip*chunk_size-skip+1, y_dim: input_dim + y_dim], (input_dim, 1)).cpu()
        x_minus=a*torch.tanh(x*b)+2/9* torch.exp(-((x)**2))+torch.matmul(B,u)
    else:
        x_minus=a*torch.tanh(x*b)+2/9* torch.exp(-((x)**2))
    x=torch.tensor(x_minus);
    # h=torch.unsqueeze(h[2*chunk_size-2,1:], dim=0).cpu()
    h=torch.reshape(h[skip*chunk_size-skip, y_dim:], (y_dim, input_dim)).cpu()
    out=torch.matmul(h.float(),x.float());
    return out



def perform_Extened_Kalman_filtering_non_scalar_control_Tanh_mode_9(h,y, a, b, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, u=None):
    # f = KalmanFilter(dim_x=input_dim, dim_z=y_dim, dim_u=input_dim)
    raise("Mode 9 not supported as the jacobian of F is incorrect: Fix it first !!!")
    R=R.cpu()
    Q=Q.cpu();
    if control:
        B=B.cpu()
        skip=3
    else:
        skip=2

    x=torch.zeros((input_dim,1))
    P=torch.eye(input_dim)

    


    for i in range(chunk_size-1):
        # f.H=torch.unsqueeze(h[2*i,y_dim:], dim=0).cpu()
        H=torch.reshape(h[skip*i,y_dim:], (y_dim, input_dim)).cpu()
        F=0.5*torch.diag(torch.squeeze(a*b*(1-torch.tanh(x*b)**2)))+0.5*torch.sigmoid(x)*(1-torch.sigmoid(x))
        if control:
            u=torch.reshape(h[skip * i+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
            x_minus=0.5*a*torch.tanh(x*b)+0.5*torch.sigmoid(x)+torch.matmul(B,u)
        else:
            x_minus = 0.5*a*torch.tanh(x*b)+0.5*torch.sigmoid(x)

        P_minus = torch.matmul(torch.matmul(F, P), F.T) + Q;
        y_i=torch.unsqueeze(y[skip*i,:], dim=-1).cpu()

        S=torch.matmul(torch.matmul(H,P_minus),H.T)+R;
        K=torch.matmul(torch.matmul(P_minus,H.T), torch.inverse(S));
        x=x_minus+torch.matmul(K,y_i-torch.matmul(H,x_minus));
        P=torch.matmul(torch.eye(input_dim)-torch.matmul(K,H),P_minus);



    if control:
        u = torch.reshape(h[skip*chunk_size-skip+1, y_dim: input_dim + y_dim], (input_dim, 1)).cpu()
        x_minus=0.5*a*torch.tanh(x*b)+0.5*torch.sigmoid(x)+torch.matmul(B,u)
    else:
        x_minus=0.5*a*torch.tanh(x*b)+0.5*torch.sigmoid(x)
    x=torch.tensor(x_minus);
    # h=torch.unsqueeze(h[2*chunk_size-2,1:], dim=0).cpu()
    h=torch.reshape(h[skip*chunk_size-skip, y_dim:], (y_dim, input_dim)).cpu()
    out=torch.matmul(h.float(),x.float());
    return out







def perform_Extened_Kalman_filtering_non_scalar_control_Tanh_mode_6(h,y, F_mat, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, u=None):
    # f = KalmanFilter(dim_x=input_dim, dim_z=y_dim, dim_u=input_dim)

    R=R.cpu()
    Q=Q.cpu();
    F_mat=F_mat.cpu()
    if control:
        B=B.cpu()
        skip=3
    else:
        skip=2

    x=torch.zeros((input_dim,1))
    P=torch.eye(input_dim)

    b=2


    for i in range(chunk_size-1):
        # f.H=torch.unsqueeze(h[2*i,y_dim:], dim=0).cpu()
        H=torch.reshape(h[skip*i,y_dim:], (y_dim, input_dim)).cpu()
        F=F_mat@torch.diag(torch.squeeze(b*(1-torch.tanh(x*b)**2)))
        if control:
            u=torch.reshape(h[skip * i+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
            x_minus=F_mat@torch.tanh(x*b)+torch.matmul(B,u)
        else:
            x_minus = F_mat@torch.tanh(x*b)

        P_minus = torch.matmul(torch.matmul(F, P), F.T) + Q;
        y_i=torch.unsqueeze(y[skip*i,:], dim=-1).cpu()

        S=torch.matmul(torch.matmul(H,P_minus),H.T)+R;
        K=torch.matmul(torch.matmul(P_minus,H.T), torch.inverse(S));
        x=x_minus+torch.matmul(K,y_i-torch.matmul(H,x_minus));
        P=torch.matmul(torch.eye(input_dim)-torch.matmul(K,H),P_minus);



    if control:
        u = torch.reshape(h[skip*chunk_size-skip+1, y_dim: input_dim + y_dim], (input_dim, 1)).cpu()
        x_minus=F_mat@torch.tanh(x*b)+torch.matmul(B,u)
    else:
        x_minus=F_mat@torch.tanh(x*b)
    x=torch.tensor(x_minus);
    # h=torch.unsqueeze(h[2*chunk_size-2,1:], dim=0).cpu()
    h=torch.reshape(h[skip*chunk_size-skip, y_dim:], (y_dim, input_dim)).cpu()
    out=torch.matmul(h.float(),x.float());
    return out






def perform_Extened_Kalman_filtering_non_scalar_control_Tanh_mode_11(h,y, F_mat, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, u=None):
    # f = KalmanFilter(dim_x=input_dim, dim_z=y_dim, dim_u=input_dim)

    R=R.cpu()
    Q=Q.cpu();
    F_mat=F_mat.cpu()
    if control:
        B=B.cpu()
        skip=3
    else:
        skip=2

    x=torch.zeros((input_dim,1))
    P=torch.eye(input_dim)

    b=2


    for i in range(chunk_size-1):
        # f.H=torch.unsqueeze(h[2*i,y_dim:], dim=0).cpu()
        H=torch.reshape(h[skip*i,y_dim:], (y_dim, input_dim)).cpu()
        F=F_mat@torch.diag(torch.squeeze(b*(1-torch.tanh(x*b)**2)-2*(2/9)*x*torch.exp(-(x)**2)))  
        if control:
            u=torch.reshape(h[skip * i+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
            x_minus=F_mat@torch.tanh(x*b)+2/9* torch.exp(-((x)**2))+torch.matmul(B,u)
        else:
            x_minus = F_mat@torch.tanh(x*b)+2/9* torch.exp(-((x)**2))

        P_minus = torch.matmul(torch.matmul(F, P), F.T) + Q;
        y_i=torch.unsqueeze(y[skip*i,:], dim=-1).cpu()

        S=torch.matmul(torch.matmul(H,P_minus),H.T)+R;
        K=torch.matmul(torch.matmul(P_minus,H.T), torch.inverse(S));
        x=x_minus+torch.matmul(K,y_i-torch.matmul(H,x_minus));
        P=torch.matmul(torch.eye(input_dim)-torch.matmul(K,H),P_minus);



    if control:
        u = torch.reshape(h[skip*chunk_size-skip+1, y_dim: input_dim + y_dim], (input_dim, 1)).cpu()
        x_minus=F_mat@torch.tanh(x*b)+torch.matmul(B,u)+2/9* torch.exp(-((x)**2))
    else:
        x_minus=F_mat@torch.tanh(x*b)+2/9* torch.exp(-((x)**2))
    x=torch.tensor(x_minus);
    # h=torch.unsqueeze(h[2*chunk_size-2,1:], dim=0).cpu()
    h=torch.reshape(h[skip*chunk_size-skip, y_dim:], (y_dim, input_dim)).cpu()
    out=torch.matmul(h.float(),x.float());
    return out



def perform_Extened_Kalman_filtering_non_scalar_control_Sin_mode_7(h,y, F_mat, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, u=None):
    # f = KalmanFilter(dim_x=input_dim, dim_z=y_dim, dim_u=input_dim)

    R=R.cpu()
    Q=Q.cpu();
    F_mat=F_mat.cpu()
    if control:
        B=B.cpu()
        skip=3
    else:
        skip=2

    x=torch.zeros((input_dim,1))
    P=torch.eye(input_dim)

    b=2


    for i in range(chunk_size-1):
        # f.H=torch.unsqueeze(h[2*i,y_dim:], dim=0).cpu()
        H=torch.reshape(h[skip*i,y_dim:], (y_dim, input_dim)).cpu()
        F=F_mat@torch.diag(torch.squeeze(b*torch.cos(b*x)))
        if control:
            u=torch.reshape(h[skip * i+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
            x_minus=F_mat@torch.sin(x*b)+torch.matmul(B,u)
        else:
            x_minus = F_mat@torch.sin(x*b)

        P_minus = torch.matmul(torch.matmul(F, P), F.T) + Q;
        y_i=torch.unsqueeze(y[skip*i,:], dim=-1).cpu()

        S=torch.matmul(torch.matmul(H,P_minus),H.T)+R;
        K=torch.matmul(torch.matmul(P_minus,H.T), torch.inverse(S));
        x=x_minus+torch.matmul(K,y_i-torch.matmul(H,x_minus));
        P=torch.matmul(torch.eye(input_dim)-torch.matmul(K,H),P_minus);



    if control:
        u = torch.reshape(h[skip*chunk_size-skip+1, y_dim: input_dim + y_dim], (input_dim, 1)).cpu()
        x_minus=F_mat@torch.sin(x*b)+torch.matmul(B,u)
    else:
        x_minus=F_mat@torch.sin(x*b)
    x=torch.tensor(x_minus);
    # h=torch.unsqueeze(h[2*chunk_size-2,1:], dim=0).cpu()
    h=torch.reshape(h[skip*chunk_size-skip, y_dim:], (y_dim, input_dim)).cpu()
    out=torch.matmul(h.float(),x.float());
    return out



def perform_Extened_Kalman_filtering_non_scalar_control_Sigmoid_mode_8(h,y, F_mat, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, u=None):
    # f = KalmanFilter(dim_x=input_dim, dim_z=y_dim, dim_u=input_dim)

    R=R.cpu()
    Q=Q.cpu();
    F_mat=F_mat.cpu()
    if control:
        B=B.cpu()
        skip=3
    else:
        skip=2

    x=torch.zeros((input_dim,1))
    P=torch.eye(input_dim)

    b=2


    for i in range(chunk_size-1):
        # f.H=torch.unsqueeze(h[2*i,y_dim:], dim=0).cpu()
        H=torch.reshape(h[skip*i,y_dim:], (y_dim, input_dim)).cpu()
        F=F_mat@torch.diag(torch.squeeze(b*torch.sigmoid(b*x)*(1-torch.sigmoid(b*x))))
        if control:
            u=torch.reshape(h[skip * i+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
            x_minus=F_mat@torch.sigmoid(x*b)+torch.matmul(B,u)
        else:
            x_minus = F_mat@torch.sigmoid(x*b)

        P_minus = torch.matmul(torch.matmul(F, P), F.T) + Q;
        y_i=torch.unsqueeze(y[skip*i,:], dim=-1).cpu()

        S=torch.matmul(torch.matmul(H,P_minus),H.T)+R;
        K=torch.matmul(torch.matmul(P_minus,H.T), torch.inverse(S));
        x=x_minus+torch.matmul(K,y_i-torch.matmul(H,x_minus));
        P=torch.matmul(torch.eye(input_dim)-torch.matmul(K,H),P_minus);



    if control:
        u = torch.reshape(h[skip*chunk_size-skip+1, y_dim: input_dim + y_dim], (input_dim, 1)).cpu()
        x_minus=F_mat@torch.sigmoid(x*b)+torch.matmul(B,u)
    else:
        x_minus=F_mat@torch.sigmoid(x*b)
    x=torch.tensor(x_minus);
    # h=torch.unsqueeze(h[2*chunk_size-2,1:], dim=0).cpu()
    h=torch.reshape(h[skip*chunk_size-skip, y_dim:], (y_dim, input_dim)).cpu()
    out=torch.matmul(h.float(),x.float());
    return out




def perform_Extened_Kalman_filtering_non_scalar_control_sin_mode_4(h,y, a, b, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, u=None):
    # f = KalmanFilter(dim_x=input_dim, dim_z=y_dim, dim_u=input_dim)

    R=R.cpu()
    Q=Q.cpu();
    if control:
        B=B.cpu()
        skip=3
    else:
        skip=2

    x=torch.zeros((input_dim,1))
    P=torch.eye(input_dim)

    


    for i in range(chunk_size-1):
        # f.H=torch.unsqueeze(h[2*i,y_dim:], dim=0).cpu()
        H=torch.reshape(h[skip*i,y_dim:], (y_dim, input_dim)).cpu()
        F=torch.diag(torch.squeeze(a*b*torch.cos(x*b)))
        if control:
            u=torch.reshape(h[skip * i+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
            x_minus=a*torch.sin(x*b)+torch.matmul(B,u)
        else:
            x_minus = a*torch.sin(x*b)

        P_minus = torch.matmul(torch.matmul(F, P), F.T) + Q;
        y_i=torch.unsqueeze(y[skip*i,:], dim=-1).cpu()

        S=torch.matmul(torch.matmul(H,P_minus),H.T)+R;
        K=torch.matmul(torch.matmul(P_minus,H.T), torch.inverse(S));
        x=x_minus+torch.matmul(K,y_i-torch.matmul(H,x_minus));
        P=torch.matmul(torch.eye(input_dim)-torch.matmul(K,H),P_minus);



    if control:
        u = torch.reshape(h[skip*chunk_size-skip+1, y_dim: input_dim + y_dim], (input_dim, 1)).cpu()
        x_minus=a*torch.sin(x*b)+torch.matmul(B,u)
    else:
        x_minus=a*torch.sin(x*b)
    x=torch.tensor(x_minus);
    # h=torch.unsqueeze(h[2*chunk_size-2,1:], dim=0).cpu()
    h=torch.reshape(h[skip*chunk_size-skip, y_dim:], (y_dim, input_dim)).cpu()
    out=torch.matmul(h.float(),x.float());
    return out



def perform_particle_filtering_tanh_mode_2func_with_control(h,y, a, b, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, N=600):
    # --- Particle Filter (with MCMC move) ---
    n=input_dim
    m=y_dim
    y_obs=y.cpu().numpy()
    T=chunk_size

    Rinv=numpy.linalg.inv(R.cpu().numpy())
    Qinv=numpy.linalg.inv(Q.cpu().numpy())

    Q=Q.cpu()
    R=R.cpu()

    sigma0 = 1.0
    P0 = sigma0**2 * np.eye(n)
    P0inv = np.linalg.inv(P0)
    J = np.zeros((T, n, n)) 



    particles = np.random.multivariate_normal(np.zeros(n), P0, size=N)
    weights = np.ones(N) / N
    a=a.cpu().numpy();
    b=b.cpu().numpy();
    


    if control:
        B=B.cpu()

    for t in range(T):
        # Predict
        if t > 0:
            if control:
                u=torch.reshape(h[skip * t+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
                particles = a*np.tanh((b*particles.T).T) + (B @ u).T + np.random.multivariate_normal(np.zeros(n), Q, N)
            else:
                particles = a*np.tanh((b*particles.T).T) + np.random.multivariate_normal(np.zeros(n), Q, N)
        # Update (likelihood)
        
        H=torch.reshape(h[skip*t,y_dim:], (y_dim, input_dim)).cpu().numpy()
        
        y_pred = (H @ particles.T).T
        if t==chunk_size-1:
            # CLRB=np.trace(H @  np.linalg.inv(J[-1] )@ H.T)
            CLRB=torch.tensor(float('inf'))

            break
        
        innov = y_obs[skip*t] - y_pred
        # Rinv = np.linalg.inv(R)
        exponent = -0.5 * np.sum(innov @ Rinv * innov, axis=1)
        exponent -= np.max(exponent)  # for stability
        likelihood = np.exp(exponent)
        weights *= likelihood
        weights += 1e-300
        weights /= np.sum(weights)
        
        # Resample
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.
        indexes = np.searchsorted(cumsum, np.linspace(0, 1-1/N, N))
        particles = particles[indexes]
        weights = np.ones(N) / N
        # MCMC move
        for i in range(N):
            prop = particles[i] + np.random.randn(n) * 0.1
            y_pred_curr = H @ particles[i]
            y_pred_prop = H @ prop
            dcurr = y_obs[skip*t] - y_pred_curr
            dprop = y_obs[skip*t] - y_pred_prop
            log_alpha = (
                -0.5 * dprop @ Rinv @ dprop
                + 0.5 * dcurr @ Rinv @ dcurr
            )
            if np.log(np.random.rand()) < log_alpha:
                particles[i] = prop

    
        if t == 0:
            J[t] = P0inv
        else:
            # Jacobians
            jacob_f = np.zeros((N, n, n))
            for i in range(N):
                bx = b*particles[i]
                diag = 1 - np.tanh(bx)**2
                jacob_f[i] = np.diagflat(diag)*b  # shape (n, n)
            jacob_h = np.zeros((N, m, n))
            for i in range(N):
                jacob_h[i] =  H  # (m, n)
            # Fisher matrices (average after forming)
            FJFs = np.array([jacob_f[i] @ J[t-1] @ jacob_f[i].T for i in range(N)])
            HJHs = np.array([jacob_h[i].T @ Rinv @ jacob_h[i] for i in range(N)])
            J[t] = Qinv + FJFs.mean(axis=0) + HJHs.mean(axis=0)
            J[t] = 0.5 * (J[t] + J[t].T)


    return torch.tensor(np.mean(y_pred,0)).float(),  CLRB


def perform_particle_filtering_tanh_mode_10func_with_control(h,y, a, b, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, N=600):
    # --- Particle Filter (with MCMC move) ---
    n=input_dim
    m=y_dim
    y_obs=y.cpu().numpy()
    T=chunk_size

    Rinv=numpy.linalg.inv(R.cpu().numpy())
    Qinv=numpy.linalg.inv(Q.cpu().numpy())

    Q=Q.cpu()
    R=R.cpu()

    sigma0 = 1.0
    P0 = sigma0**2 * np.eye(n)
    P0inv = np.linalg.inv(P0)
    J = np.zeros((T, n, n)) 



    particles = np.random.multivariate_normal(np.zeros(n), P0, size=N)
    weights = np.ones(N) / N
    a=a.cpu().numpy();
    b=b.cpu().numpy();
    


    if control:
        B=B.cpu()

    for t in range(T):
        # Predict
        if t > 0:
            if control:
                u=torch.reshape(h[skip * t+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
                particles = a*np.tanh((b*particles.T).T) + 2/9* np.exp(-((particles)**2)) +(B @ u).T + np.random.multivariate_normal(np.zeros(n), Q, N)
            else:
                particles = a*np.tanh((b*particles.T).T) +  2/9* np.exp(-((particles)**2)) + np.random.multivariate_normal(np.zeros(n), Q, N)
        # Update (likelihood)
        
        H=torch.reshape(h[skip*t,y_dim:], (y_dim, input_dim)).cpu().numpy()
        
        y_pred = (H @ particles.T).T
        if t==chunk_size-1:
            # CLRB=np.trace(H @  np.linalg.inv(J[-1] )@ H.T)
            CLRB=torch.tensor(float('inf'))

            break
        
        innov = y_obs[skip*t] - y_pred
        # Rinv = np.linalg.inv(R)
        exponent = -0.5 * np.sum(innov @ Rinv * innov, axis=1)
        exponent -= np.max(exponent)  # for stability
        likelihood = np.exp(exponent)
        weights *= likelihood
        weights += 1e-300
        weights /= np.sum(weights)
        
        # Resample
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.
        indexes = np.searchsorted(cumsum, np.linspace(0, 1-1/N, N))
        particles = particles[indexes]
        weights = np.ones(N) / N
        # MCMC move
        for i in range(N):
            prop = particles[i] + np.random.randn(n) * 0.1
            y_pred_curr = H @ particles[i]
            y_pred_prop = H @ prop
            dcurr = y_obs[skip*t] - y_pred_curr
            dprop = y_obs[skip*t] - y_pred_prop
            log_alpha = (
                -0.5 * dprop @ Rinv @ dprop
                + 0.5 * dcurr @ Rinv @ dcurr
            )
            if np.log(np.random.rand()) < log_alpha:
                particles[i] = prop

    
        if t == 0:
            J[t] = P0inv
        else:
            # Jacobians
            jacob_f = np.zeros((N, n, n))
            for i in range(N):
                bx = b*particles[i]
                diag = 1 - np.tanh(bx)**2
                diag_2 = -2 * (2/9) * particles[i] * np.exp(-particles[i]**2)
                jacob_f[i] = np.diagflat(diag)*b +np.diagflat(diag_2)  # shape (n, n)
            jacob_h = np.zeros((N, m, n))
            for i in range(N):
                jacob_h[i] =  H  # (m, n)
            # Fisher matrices (average after forming)
            FJFs = np.array([jacob_f[i] @ J[t-1] @ jacob_f[i].T for i in range(N)])
            HJHs = np.array([jacob_h[i].T @ Rinv @ jacob_h[i] for i in range(N)])
            J[t] = Qinv + FJFs.mean(axis=0) + HJHs.mean(axis=0)
            J[t] = 0.5 * (J[t] + J[t].T)


    return torch.tensor(np.mean(y_pred,0)).float(),  CLRB


def perform_particle_filtering_tanh_mode_9func_with_control(h,y, a, b, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, N=600):
    # --- Particle Filter (with MCMC move) ---
    n=input_dim
    m=y_dim
    y_obs=y.cpu().numpy()
    T=chunk_size

    Rinv=numpy.linalg.inv(R.cpu().numpy())
    Qinv=numpy.linalg.inv(Q.cpu().numpy())

    Q=Q.cpu()
    R=R.cpu()

    sigma0 = 1.0
    P0 = sigma0**2 * np.eye(n)
    P0inv = np.linalg.inv(P0)
    J = np.zeros((T, n, n)) 



    particles = np.random.multivariate_normal(np.zeros(n), P0, size=N)
    weights = np.ones(N) / N
    a=a.cpu().numpy();
    b=b.cpu().numpy();
    


    if control:
        B=B.cpu()

    for t in range(T):
        # Predict
        if t > 0:
            if control:
                u=torch.reshape(h[skip * t+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
                particles = 0.5*a*np.tanh((b*particles.T).T) + 0.5*sigmoid(particles) + (B @ u).T + np.random.multivariate_normal(np.zeros(n), Q, N)
            else:
                particles = 0.5*a*np.tanh((b*particles.T).T) + 0.5*sigmoid(particles)+ np.random.multivariate_normal(np.zeros(n), Q, N)
        # Update (likelihood)
        
        H=torch.reshape(h[skip*t,y_dim:], (y_dim, input_dim)).cpu().numpy()
        
        y_pred = (H @ particles.T).T
        if t==chunk_size-1:
            # CLRB=np.trace(H @  np.linalg.inv(J[-1] )@ H.T)
            CLRB=torch.tensor(float('inf'))

            break
        
        innov = y_obs[skip*t] - y_pred
        # Rinv = np.linalg.inv(R)
        exponent = -0.5 * np.sum(innov @ Rinv * innov, axis=1)
        exponent -= np.max(exponent)  # for stability
        likelihood = np.exp(exponent)
        weights *= likelihood
        weights += 1e-300
        weights /= np.sum(weights)
        
        # Resample
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.
        indexes = np.searchsorted(cumsum, np.linspace(0, 1-1/N, N))
        particles = particles[indexes]
        weights = np.ones(N) / N
        # MCMC move
        for i in range(N):
            prop = particles[i] + np.random.randn(n) * 0.1
            y_pred_curr = H @ particles[i]
            y_pred_prop = H @ prop
            dcurr = y_obs[skip*t] - y_pred_curr
            dprop = y_obs[skip*t] - y_pred_prop
            log_alpha = (
                -0.5 * dprop @ Rinv @ dprop
                + 0.5 * dcurr @ Rinv @ dcurr
            )
            if np.log(np.random.rand()) < log_alpha:
                particles[i] = prop

    
        if t == 0:
            J[t] = P0inv
        else:
            # Jacobians
            jacob_f = np.zeros((N, n, n))
            for i in range(N):
                bx = b*particles[i]
                diag = 1 - np.tanh(bx)**2
                jacob_f[i] = 0.5*np.diagflat(diag)*b  # shape (n, n)
                jacob_f[i] += 0.5*sigmoid(particles[i])*(1-sigmoid(particles[i]))  # shape (n, n)
            jacob_h = np.zeros((N, m, n))
            for i in range(N):
                jacob_h[i] =  H  # (m, n)
            # Fisher matrices (average after forming)
            FJFs = np.array([jacob_f[i] @ J[t-1] @ jacob_f[i].T for i in range(N)])
            HJHs = np.array([jacob_h[i].T @ Rinv @ jacob_h[i] for i in range(N)])
            J[t] = Qinv + FJFs.mean(axis=0) + HJHs.mean(axis=0)
            J[t] = 0.5 * (J[t] + J[t].T)


    return torch.tensor(np.mean(y_pred,0)).float(),  CLRB



def perform_particle_filtering_tanh_mode_6func_with_control(
        h, y, F_mat, B=0*np.eye(8), input_dim=8, chunk_size=40,
        Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, N=200):
    # --- Particle Filter (with MCMC move) ---
    n = input_dim
    m = y_dim
    y_obs = y.cpu().numpy()
    T = chunk_size
    b = 2

    R = R.cpu().numpy()
    Q = Q.cpu().numpy()
    Rinv = np.linalg.inv(R)
    Qinv = np.linalg.inv(Q)

    # sigma0 = 1.0
    
    sigma0=1;
    P0 = sigma0**2 * np.eye(n)
    P0inv = np.linalg.inv(P0)

    particles = np.random.multivariate_normal(np.zeros(n), P0, size=N)
    weights = np.ones(N) / N

    F_mat = F_mat.cpu().numpy()
    if control:
        B = B.cpu().numpy()

    for t in range(T):
        # Predict
        if t > 0:
            if control:
                u = h[skip*t+1, y_dim: input_dim + y_dim].cpu().reshape(input_dim, 1)
                particles = (F_mat @ np.tanh(b * particles.T)).T \
                          + (B @ u).T \
                          + np.random.multivariate_normal(np.zeros(n), Q, N)
            else:
                particles = (F_mat @ np.tanh(b * particles.T)).T \
                          + np.random.multivariate_normal(np.zeros(n), Q, N)

        # Update (likelihood)
        H = h[skip*t, y_dim:].cpu().numpy().reshape(y_dim, input_dim)
        y_pred = (H @ particles.T).T

        if t == chunk_size - 1:
            CLRB = torch.tensor(float('inf'))
            break

        innov = y_obs[skip*t] - y_pred
        exponent = -0.5 * np.sum(innov @ Rinv * innov, axis=1)
        exponent -= np.max(exponent)
        likelihood = np.exp(exponent)

        weights *= likelihood
        weights += 1e-300
        weights /= np.sum(weights)

        # --- MULTINOMIAL RESAMPLING ---
        indices = np.random.choice(N, size=N, replace=True, p=weights)
        particles = particles[indices]
        weights[:] = 1.0 / N

        #MCMC move
        # accept=0;
        for i in range(N):
            prop = particles[i] + np.random.randn(n) * 0.025*n
            dcurr = y_obs[skip*t] - (H @ particles[i])
            dprop = y_obs[skip*t] - (H @ prop)
            log_alpha = (
                -0.5 * dprop @ Rinv @ dprop
                + 0.5 * dcurr @ Rinv @ dcurr
            )
            if np.log(np.random.rand()) < log_alpha:
                particles[i] = prop
                # accept+=1;
        #print("alpha: ", accept/N);

        # Fisher / J-tracking (unchanged)
        if t == 0:
            J = np.zeros((T, n, n))
            J[t] = P0inv
        else:
            jacob_f = np.zeros((N, n, n))
            for i in range(N):
                bx = b * particles[i]
                diag = 1 - np.tanh(bx)**2
                jacob_f[i] = F_mat @ np.diagflat(diag) * b
            jacob_h = np.tile(H, (N, 1, 1))  # shape (N, m, n)
            FJFs = np.array([jacob_f[i] @ J[t-1] @ jacob_f[i].T for i in range(N)])
            HJHs = np.array([jacob_h[i].T @ Rinv @ jacob_h[i] for i in range(N)])
            J[t] = Qinv + FJFs.mean(axis=0) + HJHs.mean(axis=0)
            J[t] = 0.5 * (J[t] + J[t].T)

    return torch.tensor(y_pred.mean(0)).float(), CLRB




def perform_particle_filtering_tanh_mode_11func_with_control(
        h, y, F_mat, B=0*np.eye(8), input_dim=8, chunk_size=40,
        Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, N=200):
    # --- Particle Filter (with MCMC move) ---
    n = input_dim
    m = y_dim
    y_obs = y.cpu().numpy()
    T = chunk_size
    b = 2

    R = R.cpu().numpy()
    Q = Q.cpu().numpy()
    Rinv = np.linalg.inv(R)
    Qinv = np.linalg.inv(Q)

    # sigma0 = 1.0
    
    sigma0=1;
    P0 = sigma0**2 * np.eye(n)
    P0inv = np.linalg.inv(P0)

    particles = np.random.multivariate_normal(np.zeros(n), P0, size=N)
    weights = np.ones(N) / N

    F_mat = F_mat.cpu().numpy()
    if control:
        B = B.cpu().numpy()

    for t in range(T):
        # Predict
        if t > 0:
            if control:
                u = h[skip*t+1, y_dim: input_dim + y_dim].cpu().reshape(input_dim, 1)
                particles = (F_mat @ np.tanh(b * particles.T)).T \
                          +  2/9* np.exp(-((particles)**2))+ (B @ u).T \
                          + np.random.multivariate_normal(np.zeros(n), Q, N)
            else:
                particles = (F_mat @ np.tanh(b * particles.T)).T \
                          +  2/9* np.exp(-((particles)**2))+ np.random.multivariate_normal(np.zeros(n), Q, N)

        # Update (likelihood)
        H = h[skip*t, y_dim:].cpu().numpy().reshape(y_dim, input_dim)
        y_pred = (H @ particles.T).T

        if t == chunk_size - 1:
            CLRB = torch.tensor(float('inf'))
            break

        innov = y_obs[skip*t] - y_pred
        exponent = -0.5 * np.sum(innov @ Rinv * innov, axis=1)
        exponent -= np.max(exponent)
        likelihood = np.exp(exponent)

        weights *= likelihood
        weights += 1e-300
        weights /= np.sum(weights)

        # --- MULTINOMIAL RESAMPLING ---
        indices = np.random.choice(N, size=N, replace=True, p=weights)
        particles = particles[indices]
        weights[:] = 1.0 / N

        #MCMC move
        # accept=0;
        for i in range(N):
            prop = particles[i] + np.random.randn(n) * 0.025*n
            dcurr = y_obs[skip*t] - (H @ particles[i])
            dprop = y_obs[skip*t] - (H @ prop)
            log_alpha = (
                -0.5 * dprop @ Rinv @ dprop
                + 0.5 * dcurr @ Rinv @ dcurr
            )
            if np.log(np.random.rand()) < log_alpha:
                particles[i] = prop
                # accept+=1;
        #print("alpha: ", accept/N);

        # Fisher / J-tracking (unchanged)
        if t == 0:
            J = np.zeros((T, n, n))
            J[t] = P0inv
        else:
            jacob_f = np.zeros((N, n, n))
            for i in range(N):
                bx = b * particles[i]
                diag = 1 - np.tanh(bx)**2
                diag_2 = -2 * (2/9) * particles[i] * np.exp(-particles[i]**2)
                jacob_f[i] = F_mat @ np.diagflat(diag) * b +np.diagflat(diag_2)  # shape (n, n)
            jacob_h = np.tile(H, (N, 1, 1))  # shape (N, m, n)
            FJFs = np.array([jacob_f[i] @ J[t-1] @ jacob_f[i].T for i in range(N)])
            HJHs = np.array([jacob_h[i].T @ Rinv @ jacob_h[i] for i in range(N)])
            J[t] = Qinv + FJFs.mean(axis=0) + HJHs.mean(axis=0)
            J[t] = 0.5 * (J[t] + J[t].T)

    return torch.tensor(y_pred.mean(0)).float(), CLRB




def perform_particle_filtering_sin_mode_7func_with_control(
        h, y, F_mat, B=0*np.eye(8), input_dim=8, chunk_size=40,
        Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, N=200):
    # --- Particle Filter (with MCMC move) ---
    n = input_dim
    m = y_dim
    y_obs = y.cpu().numpy()
    T = chunk_size
    b = 2

    R = R.cpu().numpy()
    Q = Q.cpu().numpy()
    Rinv = np.linalg.inv(R)
    Qinv = np.linalg.inv(Q)

    # sigma0 = 1.0
    
    sigma0=1;
    P0 = sigma0**2 * np.eye(n)
    P0inv = np.linalg.inv(P0)

    particles = np.random.multivariate_normal(np.zeros(n), P0, size=N)
    weights = np.ones(N) / N

    F_mat = F_mat.cpu().numpy()
    if control:
        B = B.cpu().numpy()

    for t in range(T):
        # Predict
        if t > 0:
            if control:
                u = h[skip*t+1, y_dim: input_dim + y_dim].cpu().reshape(input_dim, 1)
                particles = (F_mat @ np.sin(b * particles.T)).T \
                          + (B @ u).T \
                          + np.random.multivariate_normal(np.zeros(n), Q, N)
            else:
                particles = (F_mat @ np.sin(b * particles.T)).T \
                          + np.random.multivariate_normal(np.zeros(n), Q, N)

        # Update (likelihood)
        H = h[skip*t, y_dim:].cpu().numpy().reshape(y_dim, input_dim)
        y_pred = (H @ particles.T).T

        if t == chunk_size - 1:
            CLRB = torch.tensor(float('inf'))
            break

        innov = y_obs[skip*t] - y_pred
        exponent = -0.5 * np.sum(innov @ Rinv * innov, axis=1)
        exponent -= np.max(exponent)
        likelihood = np.exp(exponent)

        weights *= likelihood
        weights += 1e-300
        weights /= np.sum(weights)

        # --- MULTINOMIAL RESAMPLING ---
        indices = np.random.choice(N, size=N, replace=True, p=weights)
        particles = particles[indices]
        weights[:] = 1.0 / N

        #MCMC move
        # accept=0;
        for i in range(N):
            prop = particles[i] + np.random.randn(n) * 0.05
            dcurr = y_obs[skip*t] - (H @ particles[i])
            dprop = y_obs[skip*t] - (H @ prop)
            log_alpha = (
                -0.5 * dprop @ Rinv @ dprop
                + 0.5 * dcurr @ Rinv @ dcurr
            )
            if np.log(np.random.rand()) < log_alpha:
                particles[i] = prop
                # accept+=1;
        #print("alpha: ", accept/N);

        # Fisher / J-tracking (unchanged)
        if t == 0:
            J = np.zeros((T, n, n))
            J[t] = P0inv
        else:
            jacob_f = np.zeros((N, n, n))
            for i in range(N):
                bx = b * particles[i]
                diag = b*np.cos(bx)
                jacob_f[i] = F_mat @ np.diagflat(diag) * b
            jacob_h = np.tile(H, (N, 1, 1))  # shape (N, m, n)
            FJFs = np.array([jacob_f[i] @ J[t-1] @ jacob_f[i].T for i in range(N)])
            HJHs = np.array([jacob_h[i].T @ Rinv @ jacob_h[i] for i in range(N)])
            J[t] = Qinv + FJFs.mean(axis=0) + HJHs.mean(axis=0)
            J[t] = 0.5 * (J[t] + J[t].T)

    return torch.tensor(y_pred.mean(0)).float(), CLRB




def perform_particle_filtering_sigmoid_mode_8func_with_control(
        h, y, F_mat, B=0*np.eye(8), input_dim=8, chunk_size=40,
        Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, N=200):
    # --- Particle Filter (with MCMC move) ---
    n = input_dim
    m = y_dim
    y_obs = y.cpu().numpy()
    T = chunk_size
    b = 2

    R = R.cpu().numpy()
    Q = Q.cpu().numpy()
    Rinv = np.linalg.inv(R)
    Qinv = np.linalg.inv(Q)

    # sigma0 = 1.0
    
    sigma0=1;
    P0 = sigma0**2 * np.eye(n)
    P0inv = np.linalg.inv(P0)

    particles = np.random.multivariate_normal(np.zeros(n), P0, size=N)
    weights = np.ones(N) / N

    F_mat = F_mat.cpu().numpy()
    if control:
        B = B.cpu().numpy()

    for t in range(T):
        # Predict
        if t > 0:
            if control:
                u = h[skip*t+1, y_dim: input_dim + y_dim].cpu().reshape(input_dim, 1)
                particles = (F_mat @ sigmoid(b * particles.T)).T \
                          + (B @ u).T \
                          + np.random.multivariate_normal(np.zeros(n), Q, N)
            else:
                particles = (F_mat @ sigmoid(b * particles.T)).T \
                          + np.random.multivariate_normal(np.zeros(n), Q, N)

        # Update (likelihood)
        H = h[skip*t, y_dim:].cpu().numpy().reshape(y_dim, input_dim)
        y_pred = (H @ particles.T).T

        if t == chunk_size - 1:
            CLRB = torch.tensor(float('inf'))
            break

        innov = y_obs[skip*t] - y_pred
        exponent = -0.5 * np.sum(innov @ Rinv * innov, axis=1)
        exponent -= np.max(exponent)
        likelihood = np.exp(exponent)

        weights *= likelihood
        weights += 1e-300
        weights /= np.sum(weights)

        # --- MULTINOMIAL RESAMPLING ---
        indices = np.random.choice(N, size=N, replace=True, p=weights)
        particles = particles[indices]
        weights[:] = 1.0 / N

        #MCMC move
        # accept=0;
        for i in range(N):
            prop = particles[i] + np.random.randn(n) * 0.05
            dcurr = y_obs[skip*t] - (H @ particles[i])
            dprop = y_obs[skip*t] - (H @ prop)
            log_alpha = (
                -0.5 * dprop @ Rinv @ dprop
                + 0.5 * dcurr @ Rinv @ dcurr
            )
            if np.log(np.random.rand()) < log_alpha:
                particles[i] = prop
                # accept+=1;
        #print("alpha: ", accept/N);

        # Fisher / J-tracking (unchanged)
        if t == 0:
            J = np.zeros((T, n, n))
            J[t] = P0inv
        else:
            jacob_f = np.zeros((N, n, n))
            for i in range(N):
                bx = b * particles[i]
                diag = b*sigmoid(bx)*(1.-sigmoid(bx))
                jacob_f[i] = F_mat @ np.diagflat(diag) * b
            jacob_h = np.tile(H, (N, 1, 1))  # shape (N, m, n)
            FJFs = np.array([jacob_f[i] @ J[t-1] @ jacob_f[i].T for i in range(N)])
            HJHs = np.array([jacob_h[i].T @ Rinv @ jacob_h[i] for i in range(N)])
            J[t] = Qinv + FJFs.mean(axis=0) + HJHs.mean(axis=0)
            J[t] = 0.5 * (J[t] + J[t].T)

    return torch.tensor(y_pred.mean(0)).float(), CLRB









def perform_particle_filtering_sin_mode_4func_with_control(h,y, a, b, B=0*numpy.eye(8), input_dim=8, chunk_size=40, Q=0.0*torch.eye(8), R=0.0, y_dim=2, control=False, N=200):
    # --- Particle Filter (with MCMC move) ---
    n=input_dim
    m=y_dim
    y_obs=y.cpu().numpy()
    T=chunk_size

    Rinv=numpy.linalg.inv(R.cpu().numpy())
    Qinv=numpy.linalg.inv(Q.cpu().numpy())

    Q=Q.cpu()
    R=R.cpu()

    sigma0 = 1.0
    P0 = sigma0**2 * np.eye(n)
    P0inv = np.linalg.inv(P0)
    J = np.zeros((T, n, n)) 



    particles = np.random.multivariate_normal(np.zeros(n), P0, size=N)
    weights = np.ones(N) / N
    a=a.cpu().numpy();
    b=b.cpu().numpy();
    


    if control:
        B=B.cpu()

    for t in range(T):
        # Predict
        if t > 0:
            if control:
                u=torch.reshape(h[skip * t+1, y_dim: input_dim + y_dim], (input_dim,1)).cpu()
                particles = a*np.sin((b*particles.T).T) + (B @ u).T + np.random.multivariate_normal(np.zeros(n), Q, N)
            else:
                particles = a*np.sin((b*particles.T).T) + np.random.multivariate_normal(np.zeros(n), Q, N)
        # Update (likelihood)
        
        H=torch.reshape(h[skip*t,y_dim:], (y_dim, input_dim)).cpu().numpy()
        
        y_pred = (H @ particles.T).T
        if t==chunk_size-1:
            # CLRB=np.trace(H @  np.linalg.inv(J[-1] )@ H.T)
            CLRB=torch.tensor(float('inf'))

            break
        
        innov = y_obs[skip*t] - y_pred
        Rinv = np.linalg.inv(R)
        exponent = -0.5 * np.sum(innov @ Rinv * innov, axis=1)
        exponent -= np.max(exponent)  # for stability
        likelihood = np.exp(exponent)
        weights *= likelihood
        weights += 1e-300
        weights /= np.sum(weights)
        
        # Resample
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.
        indexes = np.searchsorted(cumsum, np.linspace(0, 1-1/N, N))
        particles = particles[indexes]
        weights = np.ones(N) / N
        # MCMC move
        for i in range(N):
            prop = particles[i] + np.random.randn(n) * 0.1
            y_pred_curr = H @ particles[i]
            y_pred_prop = H @ prop
            dcurr = y_obs[skip*t] - y_pred_curr
            dprop = y_obs[skip*t] - y_pred_prop
            log_alpha = (
                -0.5 * dprop @ Rinv @ dprop
                + 0.5 * dcurr @ Rinv @ dcurr
            )
            if np.log(np.random.rand()) < log_alpha:
                particles[i] = prop

    
        if t == 0:
            J[t] = P0inv
        else:
            # Jacobians
            jacob_f = np.zeros((N, n, n))
            for i in range(N):
                bx = b*particles[i]
                diag = a*b*np.cos(bx) 
                jacob_f[i] = np.diagflat(diag)*b  # shape (n, n)
            jacob_h = np.zeros((N, m, n))
            for i in range(N):
                jacob_h[i] =  H  # (m, n)
            # Fisher matrices (average after forming)
            FJFs = np.array([jacob_f[i] @ J[t-1] @ jacob_f[i].T for i in range(N)])
            HJHs = np.array([jacob_h[i].T @ Rinv @ jacob_h[i] for i in range(N)])
            J[t] = Qinv + FJFs.mean(axis=0) + HJHs.mean(axis=0)
            J[t] = 0.5 * (J[t] + J[t].T)


    return torch.tensor(np.mean(y_pred,0)).float(),  CLRB



def perform_Extened_Kalman_filtering_non_scalar_control_target_tracking(y, Q=torch.eye(5), R=torch.eye(2), chunk_size=40, y_dim=2):
    # f = KalmanFilter(dim_x=input_dim, dim_z=y_dim, dim_u=input_dim)

    del_t=0.1


    R=R.cpu()
    Q=Q.cpu();
    #
    # x=torch.zeros((input_dim,1))
    # P=torch.eye(input_dim)

    x=torch.matmul(sqrtm(Q).double(), torch.randn((5, 1), dtype=float)) + torch.tensor([[0], [10], [0], [-5], [-0.053]],
                                                                                     dtype=float)

    # x = torch.tensor(
    #     [[0], [10], [0], [-5], [-0.053]],
    #     dtype=float)
    P=Q.float(); #Q.float()
    J_k=torch.linalg.inv(Q.float());


    for i in range(chunk_size-1):
        # f.H=torch.unsqueeze(h[2*i,y_dim:], dim=0).cpu()
        om=x[-1,0];
        x_2=x[1,0];
        x_4 = x[3, 0];

        f1=torch.tensor((del_t*x_2*np.cos(del_t*om))/om - (x_4*(np.cos(del_t*om) - 1))/om**2 - (x_2*np.sin(del_t*om))/om**2 - (del_t*x_4*np.sin(del_t*om))/om);
        f2=torch.tensor(- del_t*x_4*np.cos(del_t*om) - del_t*x_2*np.sin(del_t*om))
        f3=torch.tensor((x_2*(np.cos(del_t*om) - 1))/om**2 - (x_4*np.sin(del_t*om))/om**2 + (del_t*x_4*np.cos(del_t*om))/om + (del_t*x_2*np.sin(del_t*om))/om)
        f4=torch.tensor(del_t*x_2*np.cos(del_t*om) - del_t*x_4*np.sin(del_t*om))
        F_jacob=torch.tensor([[1, np.sin(om*del_t)/om,0,(np.cos(om*del_t)-1)/om,f1],[0, np.cos(om*del_t), 0, -np.sin(om*del_t), f2],[0, (1-np.cos(om*del_t))/om, 1, np.sin(om*del_t)/om, f3],[0, np.sin(om*del_t), 0, np.cos(om*del_t),f4],[0,0,0,0,1]], dtype=float)
        F_act=torch.tensor([[1, np.sin(om*del_t)/om,0,(np.cos(om*del_t)-1)/om,0],[0, np.cos(om*del_t), 0, -np.sin(om*del_t), 0],[0, (1-np.cos(om*del_t))/om, 1, np.sin(om*del_t)/om, 0],[0, np.sin(om*del_t), 0, np.cos(om*del_t),0],[0,0,0,0,1]], dtype=float)

        x_minus=torch.matmul(F_act,x);
        P_minus=torch.matmul(torch.matmul(F_jacob.float(), P), F_jacob.T.float()) + Q;

        x_1_minus=x_minus[0,0];
        x_3_minus=x_minus[2, 0];



        H_jacob=torch.tensor([[x_1_minus/(x_1_minus**2 + x_3_minus**2)**(1/2), 0, x_3_minus/(x_1_minus**2 + x_3_minus**2)**(1/2), 0, 0],[-(0 + x_3_minus)/((0 + x_3_minus)**2 + (0 - x_1_minus)**2),0, -(0 - x_1_minus)/((0 + x_3_minus)**2 + (0 - x_1_minus)**2) ,0,0]], dtype=float).float()

        try:
            J_k=torch.matmul(torch.matmul(H_jacob.T, torch.linalg.pinv(R)), H_jacob)+torch.linalg.pinv(Q.float()+torch.matmul(torch.matmul(F_jacob.float(), J_k),F_jacob.T.float()));


        except:
            Reg_Q=0.1*torch.max(torch.abs(torch.diag(Q.float()+torch.matmul(torch.matmul(F_jacob.float(), J_k),F_jacob.T.float()))));
            Reg_R=0.000000001*torch.max(torch.abs(torch.diag(R)));
            try:
                J_k=torch.matmul(torch.matmul(H_jacob.T, torch.linalg.pinv(R+Reg_R*torch.diag(torch.tensor([1.,1.])))), H_jacob)+torch.linalg.pinv(Q.float()+torch.matmul(torch.matmul(F_jacob.float(), J_k),F_jacob.T.float())+Reg_Q*torch.diag(torch.tensor([1.,1.,1.,1.,1.])));
            except:
                ill_cond_flag=True
                print("P-INV failed: Discarding Example")
        y_i=torch.unsqueeze(y[i,:], dim=-1).cpu()

        S=torch.matmul(torch.matmul(H_jacob,P_minus),H_jacob.T)+R;
        K=torch.matmul(torch.matmul(P_minus,H_jacob.T), torch.linalg.pinv(S));
        x=x_minus+torch.matmul(K,y_i-torch.tensor([[np.sqrt(x_1_minus** 2 + x_3_minus ** 2)], [np.arctan2(x_3_minus, x_1_minus)]], dtype=float).float());
        P=torch.matmul(torch.eye(5)-torch.matmul(K,H_jacob),P_minus);

    om = x[-1, 0];
    x_2 = x[1, 0];
    x_4 = x[3, 0];

    f1 = torch.tensor((del_t * x_2 * np.cos(del_t * om)) / om - (x_4 * (np.cos(del_t * om) - 1)) / om ** 2 - (
                x_2 * np.sin(del_t * om)) / om ** 2 - (del_t * x_4 * np.sin(del_t * om)) / om);
    f2 = torch.tensor(- del_t * x_4 * np.cos(del_t * om) - del_t * x_2 * np.sin(del_t * om))
    f3 = torch.tensor((x_2 * (np.cos(del_t * om) - 1)) / om ** 2 - (x_4 * np.sin(del_t * om)) / om ** 2 + (
                del_t * x_4 * np.cos(del_t * om)) / om + (del_t * x_2 * np.sin(del_t * om)) / om)
    f4 = torch.tensor(del_t * x_2 * np.cos(del_t * om) - del_t * x_4 * np.sin(del_t * om))
    F_act = torch.tensor([[1, np.sin(om * del_t) / om, 0, (np.cos(om * del_t) - 1)/om, 0],
                          [0, np.cos(om * del_t), 0, -np.sin(om * del_t), 0],
                          [0, (1 - np.cos(om * del_t)) / om, 1, np.sin(om * del_t) / om, 0],
                          [0, np.sin(om * del_t), 0, np.cos(om * del_t), 0], [0, 0, 0, 0, 1]], dtype=float)

    F_jacob = torch.tensor([[1, np.sin(om * del_t) / om, 0, (np.cos(om * del_t) - 1) / om, f1],
                            [0, np.cos(om * del_t), 0, -np.sin(om * del_t), f2],
                            [0, (1 - np.cos(om * del_t)) / om, 1, np.sin(om * del_t) / om, f3],
                            [0, np.sin(om * del_t), 0, np.cos(om * del_t), f4], [0, 0, 0, 0, 1]], dtype=float)
    x_minus=torch.matmul(F_act,x);

    x_1_minus = x_minus[0, 0];
    x_3_minus = x_minus[2, 0];

    H_jacob = torch.tensor([[x_1_minus / (x_1_minus ** 2 + x_3_minus ** 2) ** (1 / 2), 0,
                             x_3_minus / (x_1_minus ** 2 + x_3_minus ** 2) ** (1 / 2), 0, 0],
                            [-(0 + x_3_minus) / ((0 + x_3_minus) ** 2 + (0 - x_1_minus) ** 2), 0,
                             -(0 - x_1_minus) / ((0 + x_3_minus) ** 2 + (0 - x_1_minus) ** 2), 0, 0]],
                           dtype=float).float()


    P_k_p_1_k = Q.float() + torch.matmul(torch.matmul(F_jacob.float(), J_k), F_jacob.T.float());


    out=torch.tensor([[np.sqrt(x_1_minus** 2 + x_3_minus ** 2)], [np.arctan2(x_3_minus, x_1_minus)]], dtype=float).cpu();
    CLRB = torch.trace(torch.matmul(torch.matmul(H_jacob, P_k_p_1_k), H_jacob.T) + R);

    return out, CLRB


def particle_filter_professor_vikalos_code_target_tracking(y, n=5, N=200, T=40, Q=torch.eye(5), R=torch.eye(2), m=2):

    y_obs=y.cpu().numpy()
    # print(y_obs.shape)


    Rinv=numpy.linalg.inv(R.cpu().numpy())
    Qinv=numpy.linalg.inv(Q.cpu().numpy())

    Q=Q.cpu()
    R=R.cpu()

    particles=torch.matmul(sqrtm(Q).double(), torch.randn((n, N), dtype=float)) + torch.tensor([[0], [10], [0], [-5], [-0.053]],
                                                                                       dtype=float)
    particles=particles.numpy().T

    weights = np.ones(N) / N
    pf_est = np.zeros((T, n))
    pf_var = np.zeros((T, n, n))


    del_t=0.1

    for t in range(T):
        # Prediction
        if t > 0:

            for p in range(N):
                x=particles[p];

                om=x[-1];
                F_act=torch.tensor([[1, np.sin(om*del_t)/om,0,(np.cos(om*del_t)-1)/om,0],[0, np.cos(om*del_t), 0, -np.sin(om*del_t), 0],[0, (1-np.cos(om*del_t))/om, 1, np.sin(om*del_t)/om, 0],[0, np.sin(om*del_t), 0, np.cos(om*del_t),0],[0,0,0,0,1]], dtype=float).numpy()

                x_minus=np.matmul(F_act,x);
                particles[p] = x_minus + np.random.multivariate_normal(np.zeros(n), Q, 1)
        # Update

        y_pred=np.zeros((N,m));
        for p in range(N):
            x_minus=particles[p];
            x_1_minus=x_minus[0];
            x_3_minus=x_minus[2];

            y_pred[p,:]=np.array([[np.sqrt(x_1_minus** 2 + x_3_minus ** 2), np.arctan2(x_3_minus, x_1_minus)]])

        if t==T-1:
            break;


        innov = y_obs[t,:] - y_pred
        exponent = -0.5 * np.sum(innov @ Rinv * innov, axis=1)
        exponent -= np.max(exponent)  # Avoid large exponents
        likelihood = np.exp(exponent)
        weights *= likelihood
        weights += 1e-300
        weights /= np.sum(weights)
        # Estimate and covariance
        pf_est[t] = np.average(particles, axis=0, weights=weights)
        dx = particles - pf_est[t]
        pf_var[t] = (dx.T @ (dx * weights[:, None]))
        # Resample (systematic)
        cumsum = np.cumsum(weights)
        cumsum[-1] = 1.
        indexes = np.searchsorted(cumsum, np.linspace(0, 1-1/N, N))
        particles = particles[indexes]
        weights = np.ones(N) / N
        # MCMC move: simple random walk for each particle
        for i in range(N):
            prop = particles[i] + np.random.randn(n) * 0.1

            x_minus=particles[i];
            x_1_minus=x_minus[0];
            x_3_minus=x_minus[2];
            y_pred_curr=np.array([[np.sqrt(x_1_minus** 2 + x_3_minus ** 2), np.arctan2(x_3_minus, x_1_minus)]])

            x_minus=prop;
            x_1_minus=x_minus[0];
            x_3_minus=x_minus[2];
            y_pred_prop=np.array([[np.sqrt(x_1_minus** 2 + x_3_minus ** 2), np.arctan2(x_3_minus, x_1_minus)]])

            dcurr = y_obs[t,:] - y_pred_curr
            dprop = y_obs[t,:] - y_pred_prop
            log_alpha = (
                    -0.5 * dprop @ Rinv @ dprop.T
                    + 0.5 * dcurr @ Rinv @ dcurr.T
            )
            if np.log(np.random.rand()) < log_alpha:
                particles[i] = prop

        # ---- CRLB (multivariate, correct averaging of matrices) ----

        # For CRLB
        J = np.zeros((T, n, n))  # Fisher info
        sigma0 = 1.0
        P0 = sigma0**2 * np.eye(n)

        if t == 0:
            J[t] = np.linalg.inv(P0)
        else:
            # Jacobians
            jacob_f = np.zeros((N, n, n))
            for i in range(N):
                x= particles[i]
                om=x[-1];
                x_2=x[1];
                x_4 = x[3];

                f1=torch.tensor((del_t*x_2*np.cos(del_t*om))/om - (x_4*(np.cos(del_t*om) - 1))/om**2 - (x_2*np.sin(del_t*om))/om**2 - (del_t*x_4*np.sin(del_t*om))/om);
                f2=torch.tensor(- del_t*x_4*np.cos(del_t*om) - del_t*x_2*np.sin(del_t*om))
                f3=torch.tensor((x_2*(np.cos(del_t*om) - 1))/om**2 - (x_4*np.sin(del_t*om))/om**2 + (del_t*x_4*np.cos(del_t*om))/om + (del_t*x_2*np.sin(del_t*om))/om)
                f4=torch.tensor(del_t*x_2*np.cos(del_t*om) - del_t*x_4*np.sin(del_t*om))
                jacob_f[i]=torch.tensor([[1, np.sin(om*del_t)/om,0,(np.cos(om*del_t)-1)/om,f1],[0, np.cos(om*del_t), 0, -np.sin(om*del_t), f2],[0, (1-np.cos(om*del_t))/om, 1, np.sin(om*del_t)/om, f3],[0, np.sin(om*del_t), 0, np.cos(om*del_t),f4],[0,0,0,0,1]], dtype=float).numpy()

            jacob_h = np.zeros((N, m, n))
            for i in range(N):
                x_minus=particles[i];
                x_1_minus=x_minus[0];
                x_3_minus=x_minus[2];
                jacob_h[i] =torch.tensor([[x_1_minus/(x_1_minus**2 + x_3_minus**2)**(1/2), 0, x_3_minus/(x_1_minus**2 + x_3_minus**2)**(1/2), 0, 0],[-(0 + x_3_minus)/((0 + x_3_minus)**2 + (0 - x_1_minus)**2),0, -(0 - x_1_minus)/((0 + x_3_minus)**2 + (0 - x_1_minus)**2) ,0,0]], dtype=float).float().numpy()


            # Fisher matrices (average after forming)
            FJFs = np.array([jacob_f[i] @ J[t-1] @ jacob_f[i].T for i in range(N)])
            HJHs = np.array([jacob_h[i].T @ Rinv @ jacob_h[i] for i in range(N)])
            J[t] = Qinv + FJFs.mean(axis=0) + HJHs.mean(axis=0)
            # Symmetrize for safety
            J[t] = 0.5 * (J[t] + J[t].T)



    return torch.tensor(np.mean(y_pred,0)).float(),  numpy.array([float('inf')])


def perform_kalman_filtering_explicit(h, F, final_state, input_dim=8, chunk_size=40, Q=0.0 * torch.eye(8), R=0.0):
    f = KalmanFilter(dim_x=input_dim, dim_z=1)

    f.R = R.cpu().numpy()
    f.F = F.cpu()
    f.Q = Q.cpu().numpy();
    for i in range(chunk_size):
        f.H = torch.unsqueeze(h[2 * i, 1:], dim=0).cpu()
        f.predict()
        y_i = torch.unsqueeze(h[2 * i+1, 0], dim=0).cpu()
        f.update(y_i)


    x = torch.tensor(f.x);
    MSE = numpy.sqrt(sum(final_state.cpu() - x)**2);
    return x, MSE



    print("stop_here")



setting='Scalar_no_stats'; #['Non_scalar', ]

loss_fcn=torch.nn.MSELoss()

trained_model=torch.load('Saved_Model/ICL_Lin_sys_option_1_dim_1_larger_model_500000_steps.pt', weights_only=False, map_location='cuda')

trained_model.eval()
 


trained_model=trained_model.to('cuda')


state_est=False
drop_mode='AllEM' # 'Noise', 'All', AllEM
input_dim = 8
y_dim = 1;
option=1; # For generating F; 1 for strategy 1 and 3 for strategy 2
drop_stats = False
saved_model_returns_att = False
control=False
Non_Linear=False
non_lin_mode=11
MC_runs=100 # Number of Monte-Carlo runs for each context length
cz_list=numpy.array(list(numpy.arange(2,8))+list(numpy.arange(8,42,3))) # Context Lengths to evaluate

calc_true_CLRB=False

MSPD_list_cz_param = [];
MSPD_list_cz_SGD_0_pt_01_param = []
MSPD_list_cz_SGD_0_pt_05_param = []
MSPD_list_cz_Ridge_0_pt_01_param = []
MSPD_list_cz_Ridge_0_pt_05_param = []
MSPD_list_cz_OLS_param = []
MSPD_list_cz_partical_param = []

MSE_transformer_list_cz_param = [];
MSE_KF_list_cz_param = [];
CLRB_KF_list_cz_param = [];
CLRB_true_KF_list_cz_param = [];
MSE_SGD_0_pt_01_list_cz_param=[]
MSE_list_Ridge_0_pt_01_list_cz_param=[]
MSE_SGD_0_pt_05_list_cz_param = []
MSE_list_Ridge_0_pt_05_list_cz_param = []
MSE_particle_list_cz_param = [];
CLRB_particle_list_cz_param = [];
MSE_OLS_list_cz_param= [];

MSPD_list_cz_param_std = [];
MSPD_list_cz_particle_param_std = [];
MSPD_list_cz_SGD_0_pt_01_param_std = []
MSPD_list_cz_SGD_0_pt_05_param_std = []
MSPD_list_cz_Ridge_0_pt_01_param_std = []
MSPD_list_cz_Ridge_0_pt_05_param_std = []
MSPD_list_cz_OLS_param_std = []

MSE_transformer_list_cz_param_std = [];
MSE_KF_list_cz_param_std = [];
MSE_particle_list_cz_param_std = [];
CLRB_particle_list_cz_param_std = [];
CLRB_KF_list_cz_param_std = [];
CLRB_true_KF_list_cz_param_std = [];
MSE_SGD_0_pt_01_list_cz_param_std=[]
MSE_list_Ridge_0_pt_01_list_cz_param_std=[]
MSE_SGD_0_pt_05_list_cz_param_std = []
MSE_list_Ridge_0_pt_05_list_cz_param_std = []
MSE_OLS_list_cz_param_std= [];



for Q_alpha_factor in [0.025]: #  From 0.025
    MSPD_list_cz = [];
    MSPD_list_cz_SGD_0_pt_01 = []
    MSPD_list_cz_SGD_0_pt_05 = []
    MSPD_list_cz_Ridge_0_pt_01 = []
    MSPD_list_cz_Ridge_0_pt_05 = []
    MSPD_list_cz_OLS = []
    MSPD_list_cz_particle = []

    MSE_transformer_list_cz = [];
    CLRB_KF_list_cz = [];
    CLRB_true_KF_list_cz = [];
    MSE_KF_list_cz = [];
    MSE_SGD_0_pt_01_list_cz = []
    MSE_list_Ridge_0_pt_01_list_cz = []
    MSE_SGD_0_pt_05_list_cz = []
    MSE_list_Ridge_0_pt_05_list_cz = []
    MSE_particle_list_cz= [];
    MSE_OLS_list_cz= [];
    CLRB_particle_list_cz = [];

    MSPD_list_cz_std = [];
    MSPD_list_cz_SGD_0_pt_01_std = []
    MSPD_list_cz_SGD_0_pt_05_std = []
    MSPD_list_cz_Ridge_0_pt_01_std = []
    MSPD_list_cz_Ridge_0_pt_05_std = []
    MSPD_list_cz_OLS_std = []
    MSPD_list_cz_particle_std = [];

    MSE_transformer_list_cz_std = [];
    CLRB_KF_list_cz_std = [];
    CLRB_true_KF_list_cz_std = [];
    MSE_KF_list_cz_std = [];
    MSE_SGD_0_pt_01_list_cz_std = []
    MSE_list_Ridge_0_pt_01_list_cz_std = []
    MSE_SGD_0_pt_05_list_cz_std = []
    MSE_list_Ridge_0_pt_05_list_cz_std = []
    MSE_particle_list_cz_std = [];
    CLRB_particle_list_cz_std = [];
    MSE_OLS_list_cz_std= [];

    for chunk_size in cz_list: #chunk_size in numpy.arange(2,42)
        print(chunk_size)
        MSPD_list = []
        MSPD_list_SGD_0_pt_01 = []
        MSPD_list_Ridge_0_pt_01 = []
        MSPD_list_SGD_0_pt_05 = []
        MSPD_list_Ridge_0_pt_05 = []
        MSPD_list_OLS = []
        MSPD_list_particle = []



        MSE_transformer_list = [];
        CLRB_KF_list = [];
        CLRB_true_KF_list = [];
        MSE_KF_list = [];
        MSE_OLS_list= [];

        MSE_particle_list= [];
        CLRB_particle_list = [];

        MSE_SGD_0_pt_01_list=[]
        MSE_list_Ridge_0_pt_01_list=[]
        MSE_SGD_0_pt_05_list = []
        MSE_list_Ridge_0_pt_05_list = []

        for i in range(MC_runs):
            print("Chunk Size: ", chunk_size, "; MC: ", i )
            ill_cond_flag = False
            F_alpha = torch.rand((1,))
            Q_alpha = Q_alpha_factor * torch.rand((1,))
            R_alpha = 0.025 * torch.rand((1,))
            non_lin_params=[1,1]


            if Non_Linear and (non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10):
                inputs_batch, outputs_batch, a_vec, b_vec = Gen_data_One_Step_with_Control_Non_Linear(
                    chunk_size=chunk_size, batch_size=1, alpha_F=F_alpha, alpha_Q=Q_alpha, alpha_R=R_alpha, F_option=option, y_dim=y_dim, control=control, discard=False, Non_Linear=Non_Linear, non_lin_mode=non_lin_mode, non_lin_params=non_lin_params,input_dim=input_dim, state_est=state_est)

            else:
                if (Non_Linear and non_lin_mode == 3 and calc_true_CLRB):
                    inputs_batch, outputs_batch, true_CLRB = Gen_data_One_Step_with_Control_Non_Linear(
                        chunk_size=chunk_size, batch_size=1, alpha_F=F_alpha, alpha_Q=Q_alpha, alpha_R=R_alpha,
                        F_option=option, y_dim=y_dim, control=control, discard=False, Non_Linear=Non_Linear,
                        non_lin_mode=non_lin_mode, calc_CLRB=calc_true_CLRB, state_est=state_est)

                else:
                    inputs_batch, outputs_batch= Gen_data_One_Step_with_Control_Non_Linear(
                    chunk_size=chunk_size, batch_size=1, alpha_F=F_alpha, alpha_Q=Q_alpha, alpha_R=R_alpha,
                    F_option=option, y_dim=y_dim, control=control, discard=False, Non_Linear=Non_Linear,
                    non_lin_mode=non_lin_mode, calc_CLRB=calc_true_CLRB, input_dim=input_dim, state_est=state_est)



            if not Non_Linear or (Non_Linear and (non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==11)):
                if control:
                    R = torch.diag(inputs_batch[0, 2 * input_dim, :y_dim])
                    Q = inputs_batch[0, input_dim:2 * input_dim, y_dim:y_dim + input_dim]
                    F = inputs_batch[0, :input_dim, y_dim:y_dim + input_dim]
                    B = inputs_batch[0, (2*input_dim+1):(3*input_dim+1), y_dim:input_dim+y_dim]
                    skip=3


                else:
                    R = torch.diag(inputs_batch[0, 2 * input_dim, :y_dim])
                    Q = inputs_batch[0, input_dim:2 * input_dim, y_dim:y_dim + input_dim]
                    F = inputs_batch[0, :input_dim, y_dim:y_dim + input_dim]
                    B= None
                    skip=2

                if not Non_Linear:
                    if not state_est:
                        ys=outputs_batch[0, skip * input_dim + 1:]
                    else:
                        ys=inputs_batch[0, skip * input_dim + 1 + (skip-1):, :y_dim]
                    out_filtering = perform_kalman_filtering_non_scalar_control(inputs_batch[0, skip * input_dim + 1:, :], ys
                                                                    , F=F, Q=Q, R=R,B=B,
                                                                    chunk_size=chunk_size, y_dim=y_dim, control=control, state_est=state_est, input_dim=input_dim)
                elif non_lin_mode==6:

                    out_filtering = perform_Extened_Kalman_filtering_non_scalar_control_Tanh_mode_6 (
                        inputs_batch[0, skip * input_dim + 1:, :], outputs_batch[0, skip * input_dim + 1:], F_mat=F,
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                    
                    out_filtering_particle, CLRB_particle = perform_particle_filtering_tanh_mode_6func_with_control(
                        inputs_batch[0, skip * input_dim + 1:, :], outputs_batch[0, skip * input_dim + 1:], F_mat=F,
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                
                elif non_lin_mode==7:

                    out_filtering = perform_Extened_Kalman_filtering_non_scalar_control_Sin_mode_7 (
                        inputs_batch[0, skip * input_dim + 1:, :], outputs_batch[0, skip * input_dim + 1:], F_mat=F,
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                    
                    out_filtering_particle, CLRB_particle = perform_particle_filtering_sin_mode_7func_with_control(
                        inputs_batch[0, skip * input_dim + 1:, :], outputs_batch[0, skip * input_dim + 1:], F_mat=F,
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                elif non_lin_mode==8:
                    out_filtering = perform_Extened_Kalman_filtering_non_scalar_control_Sigmoid_mode_8 (
                        inputs_batch[0, skip * input_dim + 1:, :], outputs_batch[0, skip * input_dim + 1:], F_mat=F,
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                    
                    out_filtering_particle, CLRB_particle = perform_particle_filtering_sigmoid_mode_8func_with_control(
                        inputs_batch[0, skip * input_dim + 1:, :], outputs_batch[0, skip * input_dim + 1:], F_mat=F,
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                    
                elif non_lin_mode==11:
                    
                    out_filtering = perform_Extened_Kalman_filtering_non_scalar_control_Tanh_mode_11 (
                        inputs_batch[0, skip * input_dim + 1:, :], outputs_batch[0, skip * input_dim + 1:], F_mat=F,
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                    
                    out_filtering_particle, CLRB_particle = perform_particle_filtering_tanh_mode_11func_with_control(
                        inputs_batch[0, skip * input_dim + 1:, :], outputs_batch[0, skip * input_dim + 1:], F_mat=F,
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                add_skip=0;
            else:
                if control:

                    if non_lin_mode==1:

                        B = inputs_batch[0, 0:input_dim , y_dim:input_dim + y_dim]
                        Q = inputs_batch[0, input_dim:2 * input_dim, y_dim:y_dim + input_dim]
                        R = torch.diag(inputs_batch[0, 2 * input_dim, :y_dim])
                        add_skip=0
                    elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                        B = inputs_batch[0, 0+2:input_dim+2, y_dim:input_dim + y_dim]
                        Q = inputs_batch[0, input_dim+2:2 * input_dim+2, y_dim:y_dim + input_dim]
                        R = torch.diag(inputs_batch[0, 2 * input_dim+2, :y_dim])
                        add_skip=2;



                    skip = 2


                else:
                    if non_lin_mode==1:
                        R = torch.diag(inputs_batch[0, input_dim, :y_dim])
                        Q = inputs_batch[0, 0*input_dim:input_dim, y_dim:y_dim + input_dim]
                        B = None
                        add_skip=0;

                    elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                        R = torch.diag(inputs_batch[0, input_dim+2, :y_dim])
                        Q = inputs_batch[0, 0 * input_dim+2:input_dim+2, y_dim:y_dim + input_dim]
                        B = None
                        add_skip = 2;

                    elif non_lin_mode==3:
                        Q = inputs_batch[0, 0:5, 2:]
                        R = torch.diag(inputs_batch[0, 5, 0:2])
                    skip = 1
                if non_lin_mode==2:
                    out_filtering = perform_Extened_Kalman_filtering_non_scalar_control_Tanh_mode_2(
                        inputs_batch[0, (skip) * input_dim + 1+2:, :],
                        outputs_batch[0, (skip) * input_dim + 1+2:], a=a_vec[0], b=b_vec[0],
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                    

                    out_filtering_particle, CLRB_particle = perform_particle_filtering_tanh_mode_2func_with_control(
                        inputs_batch[0, (skip) * input_dim + 1+2:, :],
                        outputs_batch[0, (skip) * input_dim + 1+2:], a=a_vec[0], b=b_vec[0],
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                elif non_lin_mode==9:
                    out_filtering = perform_Extened_Kalman_filtering_non_scalar_control_Tanh_mode_9(
                        inputs_batch[0, (skip) * input_dim + 1+2:, :],
                        outputs_batch[0, (skip) * input_dim + 1+2:], a=a_vec[0], b=b_vec[0],
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                    

                    out_filtering_particle, CLRB_particle = perform_particle_filtering_tanh_mode_9func_with_control(
                        inputs_batch[0, (skip) * input_dim + 1+2:, :],
                        outputs_batch[0, (skip) * input_dim + 1+2:], a=a_vec[0], b=b_vec[0],
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                    
                elif non_lin_mode==10:
                    out_filtering = perform_Extened_Kalman_filtering_non_scalar_control_Tanh_mode_10(
                        inputs_batch[0, (skip) * input_dim + 1+2:, :],
                        outputs_batch[0, (skip) * input_dim + 1+2:], a=a_vec[0], b=b_vec[0],
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                    

                    out_filtering_particle, CLRB_particle = perform_particle_filtering_tanh_mode_10func_with_control(
                        inputs_batch[0, (skip) * input_dim + 1+2:, :],
                        outputs_batch[0, (skip) * input_dim + 1+2:], a=a_vec[0], b=b_vec[0],
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)

                elif  non_lin_mode==4:  
                    out_filtering = perform_Extened_Kalman_filtering_non_scalar_control_sin_mode_4(
                        inputs_batch[0, (skip) * input_dim + 1+2:, :],
                        outputs_batch[0, (skip) * input_dim + 1+2:], a=a_vec[0], b=b_vec[0],
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)
                    

                    out_filtering_particle, CLRB_particle = perform_particle_filtering_sin_mode_4func_with_control(
                        inputs_batch[0, (skip) * input_dim + 1+2:, :],
                        outputs_batch[0, (skip) * input_dim + 1+2:], a=a_vec[0], b=b_vec[0],
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)  

                elif non_lin_mode==1:
                    out_filtering = perform_Extened_Kalman_filtering_non_scalar_control_Tanh(inputs_batch[0, (skip) * input_dim + 1:, :],
                                                                            outputs_batch[0, (skip) * input_dim + 1:],
                                                                            Q=Q, R=R, B=B,
                                                                            chunk_size=chunk_size, y_dim=y_dim,
                                                                            control=control,input_dim=input_dim)
                    
                    out_filtering_particle, CLRB_particle = perform_particle_filtering_tanh_mode_2func_with_control(
                        inputs_batch[0, (skip) * input_dim + 1:, :],
                        outputs_batch[0, (skip) * input_dim + 1:], a=1, b=1,
                        Q=Q, R=R, B=B,
                        chunk_size=chunk_size, y_dim=y_dim,
                        control=control, input_dim=input_dim)

                elif non_lin_mode==3:
                    out_filtering, CLRB = perform_Extened_Kalman_filtering_non_scalar_control_target_tracking(y=inputs_batch[0, 6:, 0:2], Q=Q, R=R, chunk_size=chunk_size)
                    out_filtering_particle, CLRB_particle = particle_filter_professor_vikalos_code_target_tracking(y=inputs_batch[0, 6:, 0:2], Q=Q, R=R, T=chunk_size)

            if not (Non_Linear and non_lin_mode==3):
                out_SGD_0_pt_01 = Stochastic_Gradient_Descent_Regression_one_step_non_scalar_control(
                    inputs_batch[:, skip * input_dim + 1+add_skip:, :], alpha=0.01, device='cuda', y_dim=y_dim, control=control, state_est=state_est)
                out_Ridge_0_pt_01 = Ridge_Regression_one_step_non_scalar_control(
                    inputs_batch=inputs_batch[:, skip * input_dim + 1+add_skip:, :], lambda_=0.01, device='cuda', y_dim=y_dim,
                    control=control, state_est=state_est)
                out_SGD_0_pt_05 = Stochastic_Gradient_Descent_Regression_one_step_non_scalar_control(
                    inputs_batch[:, skip * input_dim + 1+add_skip:, :], alpha=0.05, device='cuda', y_dim=y_dim, control=control,state_est=state_est)
                out_Ridge_0_pt_05 = Ridge_Regression_one_step_non_scalar_control(
                    inputs_batch=inputs_batch[:, skip * input_dim + 1+add_skip:, :], lambda_=0.05,
                    device='cuda', y_dim=y_dim, control=control, state_est=state_est)
                if chunk_size >= 9:
                    out_OLS = Ridge_Regression_one_step_non_scalar_control(
                        inputs_batch=inputs_batch[:, skip * input_dim + 1+add_skip:, :], lambda_=0.0,
                        device='cuda', y_dim=y_dim, control=control, state_est=state_est)

            else:
                if not control:
                    if not drop_stats:
                        out_SGD_0_pt_01 = Stochastic_Gradient_Descent_Regression_one_step_non_scalar_control_target_track(
                            inputs_batch[:, 6:, 0:2], alpha=0.01, device='cuda',
                            chunk_size=chunk_size, control=False)
                        out_SGD_0_pt_05 = Stochastic_Gradient_Descent_Regression_one_step_non_scalar_control_target_track(
                            inputs_batch[:, 6:, 0:2], alpha=0.05, device='cuda',
                            chunk_size=chunk_size, control=False)

                    else:
                        out_SGD_0_pt_01 = Stochastic_Gradient_Descent_Regression_one_step_non_scalar_control_target_track(
                            inputs_batch, alpha=0.01, device='cuda',
                            chunk_size=chunk_size, control=False)
                        out_SGD_0_pt_05 = Stochastic_Gradient_Descent_Regression_one_step_non_scalar_control_target_track(
                            inputs_batch, alpha=0.05, device='cuda',
                            chunk_size=chunk_size, control=False)
            with torch.no_grad():
                if not Non_Linear or (Non_Linear and (non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==11)):
                    if control:
                        if drop_stats:
                            if drop_mode == 'Noise':
                                out = trained_model(torch.cat((torch.cat((inputs_batch[:, :input_dim, :],
                                                                          inputs_batch[:,
                                                                          2 * input_dim + 1:3 * input_dim + 1, :]),
                                                                         dim=1), inputs_batch[:,
                                                                                 3 * input_dim + 1:3 * chunk_size + 3 * input_dim + 1 - 1,
                                                                                 :]), dim=1))
                            elif drop_mode == 'All':
                                out = trained_model(
                                    inputs_batch[:, 3 * input_dim + 1:3 * chunk_size + 3 * input_dim + 1 - 1, :])
                                
                            elif drop_mode == 'AllEM':
                                out = trained_model(inputs_batch[:, 3 * input_dim + 3:3 * (chunk_size-1) + 3 * input_dim + 1 - 1:3, :])

                        else:
                            out = trained_model(inputs_batch[:, 0:3 * chunk_size + 3 * input_dim + 1 - 1, :])
                    else:
                        if drop_stats:
                            if drop_mode == 'Noise':
                                out = trained_model(torch.cat((inputs_batch[:, :input_dim, :], inputs_batch[:,
                                                                                               2 * input_dim + 1:2 * chunk_size + 2 * input_dim + 1 - 1,
                                                                                               :]), dim=1))
                            elif drop_mode == 'All':
                                out = trained_model(
                                    inputs_batch[:, 2 * input_dim + 1:2 * chunk_size + 2 * input_dim + 1 - 1, :])
                                
                            elif drop_mode == 'AllEM':
                                out = trained_model(
                                    inputs_batch[:, 2 * input_dim + 2:2 * (chunk_size-1) + 2 * input_dim + 1 - 1, :])

                        else:
                            out = trained_model(inputs_batch[:, 0:2 * chunk_size + 2 * input_dim + 1 - 1, :])

                else:
                    if drop_stats:
                        if control:
                            if non_lin_mode==1:
                                out = trained_model(
                                    inputs_batch[:, 2 * input_dim + 1:3 * chunk_size + 2 * input_dim + 1 - 1, :])
                            elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                                out = trained_model(
                                    inputs_batch[:, 2 * input_dim + 1+2:3 * chunk_size + 2 * input_dim +2 + 1 - 1, :])
                        else:
                            if non_lin_mode==1:
                                out = trained_model(
                                    inputs_batch[:,  input_dim + 1:2 * chunk_size + input_dim + 1 - 1, :])
                            elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                                out = trained_model(
                                    inputs_batch[:, input_dim + 1+2:2 * chunk_size + input_dim + 1 +2 - 1, :])

                            elif non_lin_mode==3:
                                out = trained_model(
                                    inputs_batch[:, 6:, 0:2])


                    else:
                        if control:
                            if non_lin_mode == 1:
                                out = trained_model(
                                    inputs_batch[:, 0:3 * chunk_size + 2 * input_dim + 1 - 1, :])
                            elif (non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10):
                                out = trained_model(
                                    inputs_batch[:, 0:3 * chunk_size + 2 * input_dim + 2 + 1 - 1,
                                    :])
                        else:
                            if non_lin_mode == 1:
                                out = trained_model(
                                    inputs_batch[:, 0:2 * chunk_size + input_dim + 1 - 1, :])
                            elif non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==9 or non_lin_mode==10:
                                out = trained_model(
                                    inputs_batch[:, 0:2 * chunk_size + input_dim + 1 + 2 - 1, :])

                            elif non_lin_mode==3:
                                out = trained_model(
                                    inputs_batch[:, 0:, :])

            if not (Non_Linear and non_lin_mode == 3):

                if not state_est:
                    gt=inputs_batch[:,-1,:y_dim]
                else:
                    gt=outputs_batch[:,-2,:]
                MSE_transformer_list+=[torch.sum((gt-out[:,-1])**2).cpu()];
                MSE_KF_list+=[torch.sum((torch.squeeze(out_filtering) - gt.cpu()) ** 2).cpu()];



                if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
                    # MSE_particle_list+= [torch.sum((torch.squeeze(out_filtering_particle) - outputs_batch[:, -1, :2].cpu()) ** 2).cpu()];
                    MSE_particle_list+= [torch.sum((torch.squeeze(out_filtering_particle) - inputs_batch[:, -1, :y_dim].cpu()) ** 2).cpu()];
                    # print("Debug: Particle MSE: ",torch.sum((torch.squeeze(out_filtering_particle) - inputs_batch[:, -1, :y_dim].cpu()) ** 2).cpu())
                    CLRB_particle_list += [CLRB_particle];
                    MSPD_list_particle += [[torch.sum(((torch.squeeze(out_filtering_particle) - out[:, -1].cpu()) ** 2))]]



                MSE_SGD_0_pt_01_list += [torch.sum((((out_SGD_0_pt_01.cpu() - gt.cpu()) ** 2)))]
                MSE_list_Ridge_0_pt_01_list += [torch.sum((((torch.squeeze(out_Ridge_0_pt_01.cpu()) - gt.cpu()) ** 2)))]
                MSE_SGD_0_pt_05_list += [torch.sum((((out_SGD_0_pt_05.cpu() - gt.cpu()) ** 2)))]
                MSE_list_Ridge_0_pt_05_list += [torch.sum((((torch.squeeze(out_Ridge_0_pt_05.cpu()) - gt.cpu()) ** 2)))]

                
                MSPD_list += [torch.sum(((torch.squeeze(out_filtering) - out[:, -1].cpu()) ** 2))]
                MSPD_list_SGD_0_pt_01 += [torch.sum((((out_SGD_0_pt_01.cpu() - out[:, -1].cpu()) ** 2)))]
                MSPD_list_Ridge_0_pt_01 += [torch.sum((((torch.squeeze(out_Ridge_0_pt_01.cpu()) - out[:, -1].cpu()) ** 2)))]
                MSPD_list_SGD_0_pt_05 += [torch.sum((((out_SGD_0_pt_05.cpu() - out[:, -1].cpu()) ** 2)))]
                MSPD_list_Ridge_0_pt_05 += [
                    torch.sum((((torch.squeeze(out_Ridge_0_pt_05.cpu()) - out[:, -1].cpu()) ** 2)))]

                if chunk_size >= 9:
                    MSPD_list_OLS += [
                        torch.sum((((torch.squeeze(out_OLS.cpu()) - out[:, -1].cpu()) ** 2)))]
                    MSE_OLS_list+=  [torch.sum((((torch.squeeze(out_OLS.cpu()) - gt.cpu()) ** 2)))];
            else:
                if not ill_cond_flag:
                    MSE_transformer_list += [torch.sum((outputs_batch[:, -1, :2] - out[:, -1]) ** 2).cpu()];
                    MSE_KF_list += [torch.sum((torch.squeeze(out_filtering) - outputs_batch[:, -1, :2].cpu()) ** 2).cpu()];
                    CLRB_KF_list+=[CLRB]
                    MSE_particle_list+= [torch.sum((torch.squeeze(out_filtering_particle) - outputs_batch[:, -1, :2].cpu()) ** 2).cpu()];
                    CLRB_particle_list += [CLRB_particle];

                    if (Non_Linear and non_lin_mode == 3 and calc_true_CLRB):
                        CLRB_true_KF_list += [true_CLRB]

                    MSE_SGD_0_pt_01_list += [
                        torch.sum((((out_SGD_0_pt_01.cpu() - outputs_batch[:, -1, :2].cpu()) ** 2)))]
                    MSE_SGD_0_pt_05_list += [
                        torch.sum((((out_SGD_0_pt_05.cpu() - outputs_batch[:, -1, :2].cpu()) ** 2)))]
                    MSPD_list += [torch.sum(((torch.squeeze(out_filtering) - out[:, -1].cpu()) ** 2))]
                    MSPD_list_particle += [[torch.sum(((torch.squeeze(out_filtering_particle) - out[:, -1].cpu()) ** 2))]]

                    MSPD_list_SGD_0_pt_05 += [torch.sum((((out_SGD_0_pt_05.cpu() - out[:, -1].cpu()) ** 2)))]
                    MSPD_list_SGD_0_pt_01 += [torch.sum((((out_SGD_0_pt_01.cpu() - out[:, -1].cpu()) ** 2)))]

        if not (Non_Linear and non_lin_mode == 3):


            if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
                 MSPD_list_cz_particle += [numpy.mean(torch.tensor(MSPD_list_particle).numpy() / input_dim)]
                 MSE_particle_list_cz += [numpy.mean(MSE_particle_list) / input_dim];
                 CLRB_particle_list_cz += [numpy.mean(CLRB_particle_list) /input_dim];
                 MSPD_list_cz_particle_std += [numpy.std(torch.tensor(MSPD_list_particle).numpy() / input_dim)]
                 MSE_particle_list_cz_std += [numpy.std(MSE_particle_list) / input_dim];
                 CLRB_particle_list_cz_std += [numpy.std(CLRB_particle_list) / input_dim];



            MSPD_list_cz += [numpy.mean(torch.tensor(MSPD_list).numpy() / input_dim)]
            MSPD_list_cz_SGD_0_pt_01 += [numpy.mean(torch.tensor(MSPD_list_SGD_0_pt_01).numpy() / input_dim)]
            MSPD_list_cz_SGD_0_pt_05 += [numpy.mean(torch.tensor(MSPD_list_SGD_0_pt_05).numpy() / input_dim)]
            MSPD_list_cz_Ridge_0_pt_01 += [numpy.mean(torch.tensor(MSPD_list_Ridge_0_pt_01).numpy() / input_dim)]
            MSPD_list_cz_Ridge_0_pt_05 += [numpy.mean(torch.tensor(MSPD_list_Ridge_0_pt_05).numpy() / input_dim)]
            if chunk_size >= 9:
                MSPD_list_cz_OLS += [numpy.mean(torch.tensor(MSPD_list_OLS).numpy() / input_dim)]
                MSE_OLS_list_cz+= [numpy.mean(MSE_OLS_list) / input_dim];


            MSE_transformer_list_cz += [numpy.mean(MSE_transformer_list)/input_dim];
            MSE_KF_list_cz += [numpy.mean(MSE_KF_list)/input_dim];

            MSE_SGD_0_pt_01_list_cz += [numpy.mean(MSE_SGD_0_pt_01_list)/input_dim]
            MSE_list_Ridge_0_pt_01_list_cz += [numpy.mean(MSE_list_Ridge_0_pt_01_list)/input_dim];
            MSE_SGD_0_pt_05_list_cz += [numpy.mean(MSE_SGD_0_pt_05_list) / input_dim]
            MSE_list_Ridge_0_pt_05_list_cz += [numpy.mean(MSE_list_Ridge_0_pt_05_list) / input_dim];
            
            MSPD_list_cz_std += [numpy.std(torch.tensor(MSPD_list).numpy() / input_dim)]
            MSPD_list_cz_SGD_0_pt_01_std += [numpy.std(torch.tensor(MSPD_list_SGD_0_pt_01).numpy() / input_dim)]
            MSPD_list_cz_SGD_0_pt_05_std += [numpy.std(torch.tensor(MSPD_list_SGD_0_pt_05).numpy() / input_dim)]
            MSPD_list_cz_Ridge_0_pt_01_std += [numpy.std(torch.tensor(MSPD_list_Ridge_0_pt_01).numpy() / input_dim)]
            MSPD_list_cz_Ridge_0_pt_05_std += [numpy.std(torch.tensor(MSPD_list_Ridge_0_pt_05).numpy() / input_dim)]
            if chunk_size >= 9:
                MSPD_list_cz_OLS_std += [numpy.std(torch.tensor(MSPD_list_OLS).numpy() / input_dim)]
                MSE_OLS_list_cz_std+=[numpy.std(MSE_OLS_list) / input_dim]

            MSE_transformer_list_cz_std += [numpy.std(MSE_transformer_list) / input_dim];
            MSE_KF_list_cz_std += [numpy.std(MSE_KF_list) / input_dim];

            MSE_SGD_0_pt_01_list_cz_std += [numpy.std(MSE_SGD_0_pt_01_list) / input_dim]
            MSE_list_Ridge_0_pt_01_list_cz_std += [numpy.std(MSE_list_Ridge_0_pt_01_list) / input_dim];
            MSE_SGD_0_pt_05_list_cz_std += [numpy.std(MSE_SGD_0_pt_05_list) / input_dim]
            MSE_list_Ridge_0_pt_05_list_cz_std += [numpy.std(MSE_list_Ridge_0_pt_05_list) / input_dim];



        else:
            MSPD_list_cz += [numpy.mean(torch.tensor(MSPD_list).numpy() / 5)]

            MSPD_list_cz_particle += [numpy.mean(torch.tensor(MSPD_list_particle).numpy() / 5)]

            MSE_transformer_list_cz += [numpy.mean(MSE_transformer_list) / 5];

            MSE_KF_list_cz += [numpy.mean(MSE_KF_list) / 5];
            CLRB_KF_list_cz += [numpy.mean(CLRB_KF_list) / 5];



            MSE_particle_list_cz += [numpy.mean(MSE_particle_list) / 5];
            CLRB_particle_list_cz += [numpy.mean(CLRB_particle_list) / 5];

            if (Non_Linear and non_lin_mode == 3 and calc_true_CLRB):
                CLRB_true_KF_list_cz += [numpy.mean(CLRB_true_KF_list) / 5]
            MSE_SGD_0_pt_01_list_cz += [numpy.mean(MSE_SGD_0_pt_01_list) / 5]
            MSE_SGD_0_pt_05_list_cz += [numpy.mean(MSE_SGD_0_pt_05_list) / 5]
            MSPD_list_cz_SGD_0_pt_01 += [numpy.mean(torch.tensor(MSPD_list_SGD_0_pt_01).numpy() / 5)]
            MSPD_list_cz_SGD_0_pt_05 += [numpy.mean(torch.tensor(MSPD_list_SGD_0_pt_05).numpy() / 5)]

            MSPD_list_cz_std += [numpy.std(torch.tensor(MSPD_list).numpy() / 5)]
            MSPD_list_cz_particle_std += [numpy.std(torch.tensor(MSPD_list_particle).numpy() / 5)]

            MSE_transformer_list_cz_std += [numpy.std(MSE_transformer_list) / 5];


            MSE_particle_list_cz_std += [numpy.std(MSE_particle_list) / 5];
            CLRB_particle_list_cz_std += [numpy.std(CLRB_particle_list) / 5];

            MSE_KF_list_cz_std += [numpy.std(MSE_KF_list) / 5];
            CLRB_KF_list_cz_std += [numpy.std(CLRB_KF_list) / 5]
            if (Non_Linear and non_lin_mode == 3 and calc_true_CLRB):
                CLRB_true_KF_list_cz_std += [numpy.std(CLRB_true_KF_list) / 5]
            MSE_SGD_0_pt_01_list_cz_std += [numpy.std(MSE_SGD_0_pt_01_list) / 5]
            MSE_SGD_0_pt_05_list_cz_std += [numpy.std(MSE_SGD_0_pt_05_list) / 5]
            MSPD_list_cz_SGD_0_pt_01_std += [numpy.std(torch.tensor(MSPD_list_SGD_0_pt_01).numpy() / 5)]
            MSPD_list_cz_SGD_0_pt_05_std += [numpy.std(torch.tensor(MSPD_list_SGD_0_pt_05).numpy() / 5)]






    if not (Non_Linear and non_lin_mode == 3):
        MSE_transformer_list_cz_param += [MSE_transformer_list_cz];
        MSE_KF_list_cz_param += [MSE_KF_list_cz];
        MSE_SGD_0_pt_01_list_cz_param += [MSE_SGD_0_pt_01_list_cz]
        MSE_list_Ridge_0_pt_01_list_cz_param += [MSE_list_Ridge_0_pt_01_list_cz];
        MSE_SGD_0_pt_05_list_cz_param += [MSE_SGD_0_pt_05_list_cz]
        MSE_list_Ridge_0_pt_05_list_cz_param += [MSE_list_Ridge_0_pt_05_list_cz];
        MSE_OLS_list_cz_param+=[MSE_OLS_list_cz]

        MSE_transformer_list_cz_param_std += [MSE_transformer_list_cz_std];
        MSE_KF_list_cz_param_std += [MSE_KF_list_cz_std];
        MSE_SGD_0_pt_01_list_cz_param_std += [MSE_SGD_0_pt_01_list_cz_std]
        MSE_list_Ridge_0_pt_01_list_cz_param_std += [MSE_list_Ridge_0_pt_01_list_cz_std];
        MSE_SGD_0_pt_05_list_cz_param_std += [MSE_SGD_0_pt_05_list_cz_std]
        MSE_list_Ridge_0_pt_05_list_cz_param_std += [MSE_list_Ridge_0_pt_05_list_cz_std];

        MSE_OLS_list_cz_param_std+=[MSE_OLS_list_cz_std]
    

        if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
            MSE_particle_list_cz_param += [MSE_particle_list_cz];
            CLRB_particle_list_cz_param += [CLRB_particle_list_cz];
            MSE_particle_list_cz_param_std += [MSE_particle_list_cz_std];
            CLRB_particle_list_cz_param_std += [CLRB_particle_list_cz_std];

    else:
        MSE_transformer_list_cz_param += [MSE_transformer_list_cz];
        MSE_KF_list_cz_param += [MSE_KF_list_cz];

        MSE_particle_list_cz_param += [MSE_particle_list_cz];

        CLRB_KF_list_cz_param += [CLRB_KF_list_cz];
        CLRB_particle_list_cz_param += [CLRB_particle_list_cz];

        if (Non_Linear and non_lin_mode == 3 and calc_true_CLRB):
            CLRB_true_KF_list_cz_param += [CLRB_true_KF_list_cz];

        MSE_SGD_0_pt_01_list_cz_param += [MSE_SGD_0_pt_01_list_cz]
        MSE_SGD_0_pt_05_list_cz_param += [MSE_SGD_0_pt_05_list_cz]
        MSE_SGD_0_pt_01_list_cz_param += [MSE_SGD_0_pt_01_list_cz]
        MSE_SGD_0_pt_05_list_cz_param += [MSE_SGD_0_pt_05_list_cz]

        MSE_transformer_list_cz_param_std += [MSE_transformer_list_cz_std];
        MSE_KF_list_cz_param_std += [MSE_KF_list_cz_std];
        CLRB_KF_list_cz_param_std += [CLRB_KF_list_cz_std];

        MSE_particle_list_cz_param_std += [MSE_particle_list_cz_std];
        CLRB_particle_list_cz_param_std += [CLRB_particle_list_cz_std];

        if (Non_Linear and non_lin_mode == 3 and calc_true_CLRB):
            CLRB_true_KF_list_cz_param_std += [CLRB_true_KF_list_cz_std];

        MSE_SGD_0_pt_01_list_cz_param_std += [MSE_SGD_0_pt_01_list_cz_std]
        MSE_SGD_0_pt_05_list_cz_param_std += [MSE_SGD_0_pt_05_list_cz_std]
        MSE_SGD_0_pt_01_list_cz_param_std += [MSE_SGD_0_pt_01_list_cz_std]
        MSE_SGD_0_pt_05_list_cz_param_std += [MSE_SGD_0_pt_05_list_cz_std]


    if not (Non_Linear and non_lin_mode == 3):
        MSPD_list_cz_param += [MSPD_list_cz];
        MSPD_list_cz_SGD_0_pt_01_param += [MSPD_list_cz_SGD_0_pt_01]
        MSPD_list_cz_SGD_0_pt_05_param += [MSPD_list_cz_SGD_0_pt_05]
        MSPD_list_cz_Ridge_0_pt_01_param += [MSPD_list_cz_Ridge_0_pt_01]
        MSPD_list_cz_Ridge_0_pt_05_param += [MSPD_list_cz_Ridge_0_pt_05]
        MSPD_list_cz_OLS_param += [MSPD_list_cz_OLS]

        MSPD_list_cz_param_std += [MSPD_list_cz_std];
        MSPD_list_cz_SGD_0_pt_01_param_std += [MSPD_list_cz_SGD_0_pt_01_std]
        MSPD_list_cz_SGD_0_pt_05_param_std += [MSPD_list_cz_SGD_0_pt_05_std]
        MSPD_list_cz_Ridge_0_pt_01_param_std += [MSPD_list_cz_Ridge_0_pt_01_std]
        MSPD_list_cz_Ridge_0_pt_05_param_std += [MSPD_list_cz_Ridge_0_pt_05_std]
        MSPD_list_cz_OLS_param_std += [MSPD_list_cz_OLS_std]


        if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
            MSPD_list_cz_partical_param += [MSPD_list_cz_particle];                
            MSPD_list_cz_particle_param_std += [MSPD_list_cz_particle_std]; 
    
    
    else:
        MSPD_list_cz_param += [MSPD_list_cz];
        MSPD_list_cz_partical_param += [MSPD_list_cz_particle];

        MSPD_list_cz_SGD_0_pt_01_param += [MSPD_list_cz_SGD_0_pt_01]
        MSPD_list_cz_SGD_0_pt_05_param += [MSPD_list_cz_SGD_0_pt_05]

        MSPD_list_cz_param_std += [MSPD_list_cz_std];

        MSPD_list_cz_particle_param_std += [MSPD_list_cz_particle_std];

        MSPD_list_cz_SGD_0_pt_01_param_std += [MSPD_list_cz_SGD_0_pt_01_std]
        MSPD_list_cz_SGD_0_pt_05_param_std += [MSPD_list_cz_SGD_0_pt_05_std]


plt.clf()
param_vec = [0.025] #from 0.025
for i in range(len(param_vec)):
    if not (Non_Linear and non_lin_mode == 3):
        plt.figure()

        plt.plot(cz_list, MSE_transformer_list_cz_param[i])
        plt.plot(cz_list, MSE_KF_list_cz_param[i])
        plt.plot(cz_list, MSE_SGD_0_pt_01_list_cz_param[i])
        plt.plot(cz_list, MSE_SGD_0_pt_05_list_cz_param[i])
        plt.plot(cz_list, MSE_list_Ridge_0_pt_01_list_cz_param[i])
        plt.plot(cz_list, MSE_list_Ridge_0_pt_05_list_cz_param[i])
        plt.plot( cz_list[numpy.where(cz_list>8)[0]], MSE_OLS_list_cz_param[i])


        if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
            plt.plot(cz_list, MSE_particle_list_cz_param[i])
            # plt.plot(numpy.arange(2, 42), CLRB_particle_list_cz_param[i])    




        if not Non_Linear:
            plt.legend(['Transformer', 'Kalman Filter', 'SGD 0.01', 'SGD 0.05', 'Ridge 0.01', 'Ridge 0.05', 'OLS'])

        else:
            if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
                plt.legend(
                ['Transformer', ' Extended Kalman Filter', 'SGD 0.01', 'SGD 0.05', 'Ridge 0.01', 'Ridge 0.05', 'OLS', 'Particle Filter'])
            
            else:
                plt.legend(
                ['Transformer', ' Extended Kalman Filter', 'SGD 0.01', 'SGD 0.05', 'Ridge 0.01', 'Ridge 0.05'])

        plt.xlabel('Context Length')
        plt.ylabel('1/n MSE')
        # plt.ylim([0.0, 5.0])
        filename_fig = 'MSE_SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_'+str(control)+'_nonlin_'+str(Non_Linear)+'non_lin_mode'+str(non_lin_mode)+'_state_est_'+str(state_est)+'.png'
        plt.savefig(filename_fig)


        if Non_Linear and ( non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11) :
            list_to_save = [MSE_transformer_list_cz_param[i], MSE_KF_list_cz_param[i], MSE_SGD_0_pt_01_list_cz_param[i], MSE_SGD_0_pt_05_list_cz_param[i], MSE_list_Ridge_0_pt_01_list_cz_param[i], MSE_list_Ridge_0_pt_05_list_cz_param[i], MSE_particle_list_cz_param[i], CLRB_particle_list_cz_param[i], MSE_OLS_list_cz_param[i]]

        else:
            list_to_save = [MSE_transformer_list_cz_param[i], MSE_KF_list_cz_param[i], MSE_SGD_0_pt_01_list_cz_param[i], MSE_SGD_0_pt_05_list_cz_param[i], MSE_list_Ridge_0_pt_01_list_cz_param[i], MSE_list_Ridge_0_pt_05_list_cz_param[i], MSE_OLS_list_cz_param[i]]
        filename = 'MSE_SS_one_step_non_scalar_Option_'+str(option)+'_y_dim_' + str(y_dim) + '_param_q_' + str(param_vec[i]) + '_disc_'+str(drop_stats)+'_md_'+drop_mode+'_cont_'+str(control)+'_nonlin_'+str(Non_Linear)+'non_lin_mode'+str(non_lin_mode)+'_state_est_'+str(state_est)+'_with_particle_filter.pkl';
        with open(filename, 'wb') as file:
            pickle.dump(list_to_save, file)

        plt.figure()

        plt.plot(cz_list, MSE_transformer_list_cz_param_std[i])
        plt.plot(cz_list, MSE_KF_list_cz_param_std[i])
        plt.plot(cz_list, MSE_SGD_0_pt_01_list_cz_param_std[i])
        plt.plot(cz_list, MSE_SGD_0_pt_05_list_cz_param_std[i])
        plt.plot(cz_list, MSE_list_Ridge_0_pt_01_list_cz_param_std[i])
        plt.plot(cz_list, MSE_list_Ridge_0_pt_05_list_cz_param_std[i])
        plt.plot( cz_list[numpy.where(cz_list>8)[0]], MSE_OLS_list_cz_param_std[i])


        if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
            plt.plot(cz_list, MSE_particle_list_cz_param_std[i])
            # plt.plot(numpy.arange(2, 42), CLRB_particle_list_cz_param_std[i])
            

        if not Non_Linear:
            plt.legend(['Transformer', 'Kalman Filter', 'SGD 0.01', 'SGD 0.05', 'Ridge 0.01', 'Ridge 0.05', 'OLS'])

        else:
            if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7  or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
                plt.legend(
                ['Transformer', ' Extended Kalman Filter', 'SGD 0.01', 'SGD 0.05', 'Ridge 0.01', 'Ridge 0.05', 'OLS', 'Particle Filter'])
            
            else:
                plt.legend(
                ['Transformer', ' Extended Kalman Filter', 'SGD 0.01', 'SGD 0.05', 'Ridge 0.01', 'Ridge 0.05'])

        plt.xlabel('Context Length')
        plt.ylabel('1/n MSE')



        # plt.ylim([0.0, 5.0])
        filename_fig = 'MSE_SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(
            param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_' + str(
            control) + '_nonlin_' + str(Non_Linear) + 'non_lin_mode' + str(non_lin_mode) +'_state_est_'+str(state_est)+ '_std.png'
        plt.savefig(filename_fig)

        if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
            list_to_save = [MSE_transformer_list_cz_param_std[i], MSE_KF_list_cz_param_std[i], MSE_SGD_0_pt_01_list_cz_param_std[i],
                        MSE_SGD_0_pt_05_list_cz_param_std[i], MSE_list_Ridge_0_pt_01_list_cz_param_std[i],
                        MSE_list_Ridge_0_pt_05_list_cz_param_std[i], MSE_particle_list_cz_param_std[i], CLRB_particle_list_cz_param_std[i], MSE_OLS_list_cz_param_std[i] ]

        else:    
            list_to_save = [MSE_transformer_list_cz_param_std[i], MSE_KF_list_cz_param_std[i], MSE_SGD_0_pt_01_list_cz_param_std[i],
                        MSE_SGD_0_pt_05_list_cz_param_std[i], MSE_list_Ridge_0_pt_01_list_cz_param_std[i],
                        MSE_list_Ridge_0_pt_05_list_cz_param_std[i], MSE_OLS_list_cz_param_std[i]]
        filename = 'MSE_SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(
            param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_' + str(
            control) + '_nonlin_' + str(Non_Linear) + 'non_lin_mode' + str(non_lin_mode) + '_state_est_'+str(state_est)+'_std_with_particle_filter.pkl';
        with open(filename, 'wb') as file:
            pickle.dump(list_to_save, file)

    else:
        plt.figure()

        plt.semilogy(cz_list, MSE_transformer_list_cz_param[i])
        plt.semilogy(cz_list, MSE_KF_list_cz_param[i])
        plt.semilogy(cz_list, MSE_particle_list_cz_param[i])
        plt.semilogy(cz_list, MSE_SGD_0_pt_01_list_cz_param[i])
        plt.semilogy(cz_list, MSE_SGD_0_pt_05_list_cz_param[i])
        # plt.semilogy(numpy.arange(2, 42), CLRB_particle_list_cz_param[i])
        if (Non_Linear and non_lin_mode == 3 and calc_true_CLRB):
            plt.semilogy(cz_list, CLRB_true_KF_list_cz_param[i])

        if (Non_Linear and non_lin_mode == 3 and calc_true_CLRB):
            plt.legend(
                ['Transformer', ' Extended Kalman Filter', 'Particle Filter', 'SGD 0.01', 'SGD 0.05', 'CRLB', 'CRLB True'])
        else:

            plt.legend(
                ['Transformer', ' Extended Kalman Filter','Particle Filter', 'SGD 0.01', 'SGD 0.05'])

        plt.xlabel('Context Length')
        plt.ylabel('1/n MSE')
        # plt.ylim([0.0, 5.0])
        filename_fig = 'MSE_SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(
            param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_' + str(
            control) + '_nonlin_' + str(Non_Linear) + 'non_lin_mode' + str(non_lin_mode) + '_state_est_'+str(state_est)+'_with_particle_filter.png'
        plt.savefig(filename_fig)

        plt.figure()

        plt.semilogy(cz_list, MSE_transformer_list_cz_param_std[i])
        plt.semilogy(cz_list, MSE_KF_list_cz_param_std[i])
        plt.semilogy(cz_list, MSE_particle_list_cz_param_std[i])
        plt.semilogy(cz_list, MSE_SGD_0_pt_01_list_cz_param_std[i])
        plt.semilogy(cz_list, MSE_SGD_0_pt_05_list_cz_param_std[i])
        # plt.semilogy(numpy.arange(2, 42), CLRB_particle_list_cz_param_std[i])
        if (Non_Linear and non_lin_mode == 3 and calc_true_CLRB):
            plt.semilogy(cz_list, CLRB_true_KF_list_cz_param_std[i])

        if (Non_Linear and non_lin_mode == 3 and calc_true_CLRB):
            plt.legend(
                ['Transformer', ' Extended Kalman Filter','Particle Filter', 'SGD 0.01', 'SGD 0.05', 'CRLB', 'CRLB True'])
        else:

            plt.legend(
                ['Transformer', ' Extended Kalman Filter','Particle Filter', 'SGD 0.01', 'SGD 0.05'])

        plt.xlabel('Context Length')
        plt.ylabel('1/n MSE')
        # plt.ylim([0.0, 5.0])
        filename_fig = 'MSE_SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(
            param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_' + str(
            control) + '_nonlin_' + str(Non_Linear) + 'non_lin_mode' + str(non_lin_mode) +'_state_est_'+str(state_est)+ '_std_with_particle_filter.png'
        plt.savefig(filename_fig)


        if (Non_Linear and non_lin_mode == 3 and calc_true_CLRB):
            list_to_save = [MSE_transformer_list_cz_param[i], MSE_KF_list_cz_param[i], MSE_particle_list_cz_param[i], MSE_SGD_0_pt_01_list_cz_param[i],
                            MSE_SGD_0_pt_05_list_cz_param[i], CLRB_KF_list_cz_param[i],CLRB_particle_list_cz_param[i], CLRB_true_KF_list_cz_param[i]]
        else:
            list_to_save = [MSE_transformer_list_cz_param[i], MSE_KF_list_cz_param[i], MSE_particle_list_cz_param[i], MSE_SGD_0_pt_01_list_cz_param[i],
                        MSE_SGD_0_pt_05_list_cz_param[i], CLRB_KF_list_cz_param[i], CLRB_particle_list_cz_param[i]]
        filename = 'MSE_SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(
            param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_' + str(
            control) + '_nonlin_' + str(Non_Linear) + 'non_lin_mode' + str(non_lin_mode) + '_state_est_'+str(state_est)+'_with_particle_filter.pkl';
        with open(filename, 'wb') as file:
            pickle.dump(list_to_save, file)


        if (Non_Linear and non_lin_mode == 3 and calc_true_CLRB):
            list_to_save = [MSE_transformer_list_cz_param_std[i], MSE_KF_list_cz_param_std[i],MSE_particle_list_cz_param_std[i], MSE_SGD_0_pt_01_list_cz_param_std[i],
                            MSE_SGD_0_pt_05_list_cz_param_std[i], CLRB_KF_list_cz_param_std[i],CLRB_particle_list_cz_param_std[i], CLRB_true_KF_list_cz_param_std[i]]
        else:
            list_to_save = [MSE_transformer_list_cz_param_std[i], MSE_KF_list_cz_param_std[i],MSE_particle_list_cz_param_std[i], MSE_SGD_0_pt_01_list_cz_param_std[i],
                        MSE_SGD_0_pt_05_list_cz_param_std[i], CLRB_KF_list_cz_param_std[i], CLRB_particle_list_cz_param_std[i]]
        filename = 'MSE_SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(
            param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_' + str(
            control) + '_nonlin_' + str(Non_Linear) + 'non_lin_mode' + str(non_lin_mode) + '_state_est_'+str(state_est)+'_std_with_particle_filter.pkl';
        with open(filename, 'wb') as file:
            pickle.dump(list_to_save, file)





plt.clf()
param_vec = [0.025] # Was 0.025
for i in range(len(param_vec)):
    if not (Non_Linear and non_lin_mode == 3):
        plt.figure()
        plt.plot(cz_list, MSPD_list_cz_param[i])
        plt.plot(cz_list, MSPD_list_cz_SGD_0_pt_01_param[i])
        plt.plot(cz_list, MSPD_list_cz_Ridge_0_pt_01_param[i])
        plt.plot(cz_list, MSPD_list_cz_SGD_0_pt_05_param[i])
        plt.plot(cz_list, MSPD_list_cz_Ridge_0_pt_05_param[i])
        plt.plot( cz_list[numpy.where(cz_list>8)[0]], MSPD_list_cz_OLS_param[i])

        if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
            plt.plot(cz_list, MSPD_list_cz_partical_param[i])



        if not Non_Linear:
            plt.legend(
                ['ICL and Kalman Filter', 'ICL and SGD 0.01', 'ICL and Ridge 0.01', 'ICL and SGD 0.05', 'ICL and Ridge 0.05',
                 'ICL and OLS'])

        else:
            if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
                plt.legend(
                ['ICL and Extended Kalman Filter', 'ICL and SGD 0.01', 'ICL and Ridge 0.01', 'ICL and SGD 0.05',
                 'ICL and Ridge 0.05',
                 'ICL and OLS', 'ICL and Particle Filter'])

            else:
                plt.legend(
                ['ICL and Extended Kalman Filter', 'ICL and SGD 0.01', 'ICL and Ridge 0.01', 'ICL and SGD 0.05',
                 'ICL and Ridge 0.05',
                 'ICL and OLS'])

        plt.xlabel('Context Length')
        plt.ylabel('1/n MSPD')
        plt.ylim([0.0, 5.0])
        filename_fig = 'SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_'+str(control)+'_non_lin'+str(Non_Linear)+'non_lin_mode'+str(non_lin_mode)+'_state_est_'+str(state_est)+'.png'
        plt.savefig(filename_fig)


        if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
            list_to_save = [MSPD_list_cz_param[i], MSPD_list_cz_SGD_0_pt_01_param[i], MSPD_list_cz_Ridge_0_pt_01_param[i],
                        MSPD_list_cz_SGD_0_pt_05_param[i], MSPD_list_cz_Ridge_0_pt_05_param[i], MSPD_list_cz_OLS_param[i], MSPD_list_cz_partical_param[i]]

        else:
            list_to_save = [MSPD_list_cz_param[i], MSPD_list_cz_SGD_0_pt_01_param[i], MSPD_list_cz_Ridge_0_pt_01_param[i],
                        MSPD_list_cz_SGD_0_pt_05_param[i], MSPD_list_cz_Ridge_0_pt_05_param[i], MSPD_list_cz_OLS_param[i]]
        filename = 'SS_one_step_non_scalar_Option_'+str(option)+'_y_dim_' + str(y_dim) + '_param_q_' + str(param_vec[i]) + '_disc_'+str(drop_stats)+'_md_'+drop_mode+'_cont_'+str(control)+'_non_lin'+str(Non_Linear)+'non_lin_mode'+str(non_lin_mode)+'_state_est_'+str(state_est)+'_with_particle_filter.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(list_to_save, file)

        plt.figure()
        plt.plot(cz_list, MSPD_list_cz_param_std[i])
        plt.plot(cz_list, MSPD_list_cz_SGD_0_pt_01_param_std[i])
        plt.plot(cz_list, MSPD_list_cz_Ridge_0_pt_01_param_std[i])
        plt.plot(cz_list, MSPD_list_cz_SGD_0_pt_05_param_std[i])
        plt.plot(cz_list, MSPD_list_cz_Ridge_0_pt_05_param_std[i])
        plt.plot(cz_list[numpy.where(cz_list>8)[0]], MSPD_list_cz_OLS_param_std[i])


        if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
            plt.plot(cz_list, MSPD_list_cz_particle_param_std[i])


        if not Non_Linear:
            plt.legend(
                ['ICL and Kalman Filter', 'ICL and SGD 0.01', 'ICL and Ridge 0.01', 'ICL and SGD 0.05',
                 'ICL and Ridge 0.05',
                 'ICL and OLS'])

        else:

            if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11):
            
                plt.legend(
                ['ICL and Extended Kalman Filter', 'ICL and SGD 0.01', 'ICL and Ridge 0.01', 'ICL and SGD 0.05',
                 'ICL and Ridge 0.05',
                 'ICL and OLS', 'ICL and Particle FIlter'])
            
            else:    
                plt.legend(
                ['ICL and Extended Kalman Filter', 'ICL and SGD 0.01', 'ICL and Ridge 0.01', 'ICL and SGD 0.05',
                 'ICL and Ridge 0.05',
                 'ICL and OLS'])

        plt.xlabel('Context Length')
        plt.ylabel('1/n MSPD')
        plt.ylim([0.0, 5.0])
        filename_fig = 'SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(
            param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_' + str(
            control) + '_non_lin' + str(Non_Linear) + 'non_lin_mode' + str(non_lin_mode) + '_state_est_'+str(state_est)+'_std.png'
        plt.savefig(filename_fig)


        if Non_Linear and (non_lin_mode==1 or non_lin_mode==2 or non_lin_mode==4 or non_lin_mode==5 or non_lin_mode==6 or non_lin_mode==7 or non_lin_mode==8 or non_lin_mode==9 or non_lin_mode==10 or non_lin_mode==11) :
            list_to_save = [MSPD_list_cz_param_std[i], MSPD_list_cz_SGD_0_pt_01_param_std[i], MSPD_list_cz_Ridge_0_pt_01_param_std[i],
                        MSPD_list_cz_SGD_0_pt_05_param_std[i], MSPD_list_cz_Ridge_0_pt_05_param_std[i],
                        MSPD_list_cz_OLS_param_std[i],MSPD_list_cz_particle_param_std[i]]

        else:
            list_to_save = [MSPD_list_cz_param_std[i], MSPD_list_cz_SGD_0_pt_01_param_std[i], MSPD_list_cz_Ridge_0_pt_01_param_std[i],
                        MSPD_list_cz_SGD_0_pt_05_param_std[i], MSPD_list_cz_Ridge_0_pt_05_param_std[i],
                        MSPD_list_cz_OLS_param_std[i]]
        filename = 'SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(
            param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_' + str(
            control) + '_non_lin' + str(Non_Linear) + 'non_lin_mode' + str(non_lin_mode) + '_state_est_'+str(state_est)+'_std_with_particle_filter.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(list_to_save, file)

    else:
        plt.figure()
        plt.semilogy(cz_list, MSPD_list_cz_param[i])
        plt.semilogy(cz_list, MSPD_list_cz_partical_param[i])
        plt.semilogy(cz_list, MSPD_list_cz_SGD_0_pt_01_param[i])
        plt.semilogy(cz_list, MSPD_list_cz_SGD_0_pt_05_param[i])

        plt.xlabel('Context Length')
        plt.ylabel('1/n MSPD')
        plt.legend(
            ['ICL and Extended Kalman Filter',  'ICL and Particle Filter', 'ICL and SGD 0.01', 'ICL and SGD 0.05'])

        filename_fig = 'SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(
            param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_' + str(
            control) + '_non_lin' + str(Non_Linear) + 'non_lin_mode' + str(non_lin_mode) +'_state_est_'+str(state_est)+ '_with_particle_filter.png'
        plt.savefig(filename_fig)

        list_to_save = [MSPD_list_cz_param[i], MSPD_list_cz_SGD_0_pt_01_param[i], MSPD_list_cz_SGD_0_pt_05_param[i], MSPD_list_cz_partical_param[i]]
        filename = 'SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(
            param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_' + str(
            control) + '_non_lin' + str(Non_Linear) + 'non_lin_mode' + str(non_lin_mode) +'_state_est_'+str(state_est)+ '_with_particle_filter.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(list_to_save, file)

        plt.figure()
        plt.semilogy(cz_list, MSPD_list_cz_param_std[i])
        plt.semilogy(cz_list, MSPD_list_cz_particle_param_std[i])
        plt.semilogy(cz_list, MSPD_list_cz_SGD_0_pt_01_param_std[i])
        plt.semilogy(cz_list, MSPD_list_cz_SGD_0_pt_05_param_std[i])

        plt.xlabel('Context Length')
        plt.ylabel('1/n MSPD')
        plt.legend(
            ['ICL and Extended Kalman Filter', 'ICL and Particle Filter','ICL and SGD 0.01', 'ICL and SGD 0.05'])

        filename_fig = 'SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(
            param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_' + str(
            control) + '_non_lin' + str(Non_Linear) + 'non_lin_mode' + str(non_lin_mode) + '_state_est_'+str(state_est)+'_std_with_particle_filter.png'
        plt.savefig(filename_fig)

        list_to_save = [MSPD_list_cz_param_std[i], MSPD_list_cz_SGD_0_pt_01_param_std[i], MSPD_list_cz_SGD_0_pt_05_param_std[i], MSPD_list_cz_particle_param_std[i]]
        filename = 'SS_one_step_non_scalar_Option_' + str(option) + '_y_dim_' + str(y_dim) + '_param_q_' + str(
            param_vec[i]) + '_disc_' + str(drop_stats) + '_md_' + drop_mode + '_cont_' + str(
            control) + '_non_lin' + str(Non_Linear) + 'non_lin_mode' + str(non_lin_mode) + '_state_est_'+str(state_est)+'_std_with_particle_filter.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(list_to_save, file)


print('Stop Here')




