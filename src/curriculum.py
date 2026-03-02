import math
import numpy

class Curriculum:
    def __init__(self, args):
        # args.dims and args.points each contain start, end, inc, interval attributes
        # inc denotes the change in n_dims,
        # this change is done every interval,
        # and start/end are the limits of the parameter
        self.n_dims_truncated = args.dims.start
        self.n_points = args.points.start
        self.n_dims_schedule = args.dims
        self.n_points_schedule = args.points
        self.step_count = 0
        self.F_alpha=0.0
        self.Q_alpha=0.0
        self.R_alpha=0.0
        self.F_alpha_eye_prob=0.5;
        self.F_stay_eye_till=0
        self.F_steps_to_one=50000;
        self.Q_stay_zero_till=0;
        self.Q_steps_to_max=100000;
        self.R_stay_zero_till = 0;
        self.R_steps_to_max = 100000;
        self.Q_alpha_max=0.0125; 
        self.R_alpha_max=0.025;
        self.set_alpha_prob_to_zero_after=500000
        self.y_dim=1
        self.discard=False
        self.discard_mode='AllEM' # 'All' or 'Noise' or 'AllEM
        self.option=1 # F option 1 for strategy 1 or 3 strategy 2
        self.control=False
        self.Non_Linear=False
        self.non_lin_mode=11;
        self.non_lin_params=[1,1]
        self.gpu=4;
        self.state_est=False

    def update(self):
        self.step_count += 1
        if self.step_count>self.set_alpha_prob_to_zero_after:
            self.F_alpha_eye_prob=0.0;
        if self.step_count>self.F_stay_eye_till:
            toss=numpy.random.rand()
            if toss>self.F_alpha_eye_prob:
                self.F_alpha=min(float((self.step_count-self.F_stay_eye_till)/self.F_steps_to_one), 1.0)
            else:
                self.F_alpha=0.0

        if self.step_count>self.Q_stay_zero_till:
            frac=(self.step_count-self.Q_stay_zero_till)/(self.Q_steps_to_max)
            self.Q_alpha=min(float(frac*self.Q_alpha_max),self.Q_alpha_max)

        if self.step_count>self.R_stay_zero_till:
            frac=(self.step_count-self.R_stay_zero_till)/(self.R_steps_to_max)
            self.R_alpha=min(float(frac*self.R_alpha_max),self.R_alpha_max)


        self.n_dims_truncated = self.update_var(
            self.n_dims_truncated, self.n_dims_schedule
        )
        self.n_points = self.update_var(self.n_points, self.n_points_schedule)

    def update_var(self, var, schedule):
        if self.step_count % schedule.interval == 0:
            var += schedule.inc

        return min(var, schedule.end)


# returns the final value of var after applying curriculum.
def get_final_var(init_var, total_steps, inc, n_steps, lim):
    final_var = init_var + math.floor((total_steps) / n_steps) * inc

    return min(final_var, lim)
