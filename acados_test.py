import os
import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, acados_model


class DI2DOptimizer:
    def __init__(self, model ,t_horizon=1, n_nodes = 20,
                 q_cost=None, q_mask=None, B_x=None,
                 model_name="double_integrator_mpc", solver_options=None):

        # Weighted squared error loss function q = (x, v)
        if q_cost is None:
            q_cost = np.array([10, 0.05])

        # Time horizon and the number of control nodes within horizon
        self.T = t_horizon
        self.N = n_nodes

        self.model = model

        # Declare model variables
        self.p = cs.MX.sym('p', 1)
        self.v = cs.MX.sym('v', 1)

        # Full state variables
        self.s = cs.vertcat(self.p, self.v)
        self.state_dim = 2

        # Control input constraints
        self.max_u = 10
        self.min_u = -self.max_u

        # Control input
        self.u = cs.MX.sym('u')


    def acados_setup_model(self, nominal, model_name):

        def fill_in_acados_model(s, p, u, dynamics, name):
            s_dot = cs.MX.sym('s_dot', dynamics.shape)
            f_impl = s_dot - dynamics

            # Dynamics model
            model = AcadosModel()
            model.f_expl_expr = dynamics
            model.f_impl_expr = f_impl
            model.s = s
            model.sdot = s_dot
            model.u = u
            model.p = p
            model.name = name

            return model

        acados_models = {}
        dynamics_equations = {}

        dynamics_equations[0] = nominal

        s_ = self.s
        dynamics_ = nominal

        acados_models[0] = fill_in_acados_model(s=s_, u=self.u, p=[],dynamics=dynamics_,name=model_name)

        return acados_models, dynamics_equations

    def DI_dynamics(self):
        """
        Symbolic dynamics of Double Integrator model
        :return:
        """

        s_dot = cs.vertcat(self.p_dynamics(), self.v_dynamics())
        return cs.Function('s_dot',[self.s, self.u],[s_dot],['s','u'],['s_dot'])

    def p_dynamics(self):
        return self.v

    def v_dynamics(self):
        return self.u



if __name__ == "__main__":
    print("Testing acados models")
    J = np.ones(5)