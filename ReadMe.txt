#################### Acknowledgement  #################

We adapted this code for this work from the projects of Garg et. al. and Akyurek et. al.

#################### Training the Model ###############

1. To train the model, use src/train.py --config src/conf/linear_regression.yaml
2. Model Hyperparameters and model save directories are set in src/conf/base.yaml
3. Curriculum and data generation parameters for the model training are set in src/curriculum.py
(i)  The flag self.control sets whether the dynamical system consists of control inputs
(ii) The flag self.discard sets whether to drop system statistics from context. self.discard_mode decides what statistics are dropped. self.discard_mode='AllEM'  removes all parameter information so that transformer is only provided with measurements. self.discard_mode='All' removes noise covariances, control matrix, and state transtion matrix from the context. self.discard_mode='Noise' removes only noise covariance. 
(iii) The flag self.Non_Linear sets whether the dynamical system is non-linear, the flag is True

non_lin_mode = 1
State update:
w_t = tanh(w_{t-1}) + noise + B u_t

Description:
• Pure elementwise tanh nonlinearity
• No linear mixing matrix F
• Strong saturation behavior
• Linear observation model

non_lin_mode = 2

State update:
w_t = a * tanh(b * w_{t-1}) + noise + B u_t

Parameters:
a ~ Uniform(-p1, p1)
b ~ Uniform(-p2, p2)

Description:
• Scaled tanh nonlinearity
• Adjustable amplitude and slope
• Random parameters per sequence
• Linear observation model

non_lin_mode = 4

State update:
w_t = a * sin(b * w_{t-1}) + noise + B u_t

Description:
• Periodic nonlinear dynamics
• Random amplitude and frequency
• Linear observation model

non_lin_mode = 5

State update:
w_t = a * sin(b * w_{t-1})
+ sigmoid(w_{t-1})
+ noise + B u_t

Description:
• Combination of oscillation and saturation
• More complex nonlinear behavior

non_lin_mode = 6

State update:
w_t = F * tanh(2 w_{t-1}) + noise + B u_t

Description:
• Linear mixing matrix F
• Tanh applied before mixing
• Coupled nonlinear dynamics

non_lin_mode = 7

State update:
w_t = F * sin(2 w_{t-1}) + noise + B u_t

Description:
• Oscillatory nonlinear dynamics
• Coupled through matrix F

non_lin_mode = 8

State update:
w_t = F * sigmoid(2 w_{t-1}) + noise + B u_t

Description:
• Smooth bounded nonlinearity
• Coupled through matrix F

non_lin_mode = 9

State update:
w_t = 0.5 * a * tanh(b * w_{t-1})
+ 0.5 * sigmoid(w_{t-1})
+ noise + B u_t

Description:
• Hybrid tanh + sigmoid
• Mixed nonlinear behavior

non_lin_mode = 10

State update:
w_t = a * tanh(b * w_{t-1})
+ (2/9) * exp(-(w_{t-1}^2))
+ noise + B u_t

Description:
• Tanh plus Gaussian bump
• Non-monotonic dynamics

non_lin_mode = 11

State update:
w_t = F * tanh(2 w_{t-1})
+ (2/9) * exp(-(w_{t-1}^2))
+ noise + B u_t

Description:
• Linear mixing + tanh + Gaussian bump
• More expressive coupled nonlinearity

non_lin_mode = 3 (SPECIAL CASE)

This is a Coordinated Turn (CT) model.

State:
w = [x, vx, y, vy, omega]

State transition:
Nonlinear trigonometric rotation dynamics
based on angular velocity omega.

Measurement:
y = [ sqrt(x^2 + y^2),
atan2(y, x) ] + noise

Description:
• Radar-style range and bearing measurement
• Uses Jacobians for CLRB computation
• Strongly nonlinear both in dynamics and measurement
• Most physically structured mode

#################### Evaluating the Model ###############
See Eval_ICL_Dyn_Sys.py; You have to specify model directory along with the simulation parameters that affect the model architecture (e.g observation dimensions etc)