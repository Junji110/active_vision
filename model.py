import numpy as np


class MarkovEyeMovementModel(object):
    def __init__(self, transprobs, errorprob, switchprob):
        self.set_transprobs(transprobs)
        self.set_errorprob(errorprob)
        self.set_switchprob(switchprob)
        self.sactypes = None
        self.fixtypes = None
        self.sactype_ratios = None

    def set_transprobs(self, transprobs):
        self.p_intra_early = transprobs[0][0]
        self.p_trans_early = transprobs[0][1]
        self.p_intra_late = transprobs[1][0]
        self.p_trans_late = transprobs[1][1]

    def set_errorprob(self, errorprob):
        self.p_error = errorprob

    def set_switchprob(self, switchprob):
        self.p_switch = switchprob

    def get_probs(self):
        return ((self.p_intra_early, self.p_trans_early),
                (self.p_intra_late, self.p_trans_late)), self.p_error, self.p_switch

    # methods for simulating the model
    def _init_fixtypes(self, initial_fixations):
        if len(initial_fixations) != 2:
            raise ValueError("`initial_fixations` must be a 2-tuple.")
        self.fixtypes = np.empty(self.num_fix, np.int)
        self.fixtypes[0:2] = initial_fixations

    def _gen_fixtypes(self, initial_fixations):
        self._init_fixtypes(initial_fixations)
        p_intra = self.p_intra_early
        p_trans = self.p_trans_early
        switched = False
        for i, randnum in enumerate(np.random.rand(self.num_fix-2)):
            if not switched and np.random.rand() < self.p_switch:
                p_intra = self.p_intra_late
                p_trans = self.p_trans_late
                switched = True

            if self.fixtypes[i] == self.fixtypes[i+1]:
                # case: previous saccade is intra-obj
                self.fixtypes[i+2] = self.fixtypes[i+1] if randnum < p_intra\
                    else 1 - self.fixtypes[i+1]
            else:
                # case: previous saccade is trans-obj
                self.fixtypes[i+2] = 1 - self.fixtypes[i+1] if randnum < p_trans\
                    else self.fixtypes[i+1]

    def _introduce_fix_errors(self):
        # no error for the first fixation
        idx_error = np.where(np.random.rand(self.num_fix - 1) < self.p_error)[0] + 1
        self.fixtypes[idx_error] = -1

    def _gen_sactypes(self):
        sactypes = []
        for fix_pre, fix_post in zip(self.fixtypes[:-1], self.fixtypes[1:]):
            if fix_pre == -1 and fix_post == -1:
                # case: BG-to-BG saccade
                sactypes.append(4)
            elif fix_pre == -1:
                # case: BG-to-obj saccade
                sactypes.append(3)
            elif fix_post == -1:
                # case: obj-to-BG saccade
                sactypes.append(2)
            elif fix_pre != fix_post:
                # case: trans-obj saccade
                sactypes.append(1)
            else:
                # case: intra-obj saccade
                sactypes.append(0)
        self.sactypes = np.array(sactypes)

    def run_trial(self, initial_fixations, num_fix):
        self.num_fix = num_fix
        self._gen_fixtypes(initial_fixations)
        self._introduce_fix_errors()
        self._gen_sactypes()

    def get_sactypes(self):
        if self.sactypes is None:
            raise ValueError("Saccade type sequence is not generated yet.")
        return self.sactypes

    def get_fixtypes(self):
        if self.fixtypes is None:
            raise ValueError("Fixation type sequence is not generated yet.")
        return self.fixtypes

    # methods for solving the model
    def _gen_transmat(self):
        p0, p1, p2, p3 = self.p_intra_early, self.p_trans_early,\
                         self.p_intra_late, self.p_trans_late
        q0, q1, q2, q3 = 1 - p0, 1 - p1, 1 - p2, 1 - p3
        p_sw = self.p_switch
        q_sw = 1 - p_sw
        transmat = np.zeros((4, 4))
        transmat[0, :] = (q_sw * p0, q_sw * q1, 0, 0)
        transmat[1, :] = (q_sw * q0, q_sw * p1, 0, 0)
        transmat[2, :] = (p_sw * p2, p_sw * q3, p2, q3)
        transmat[3, :] = (p_sw * q2, p_sw * p3, q2, p3)
        self.transmat = transmat

    def _gen_hidden_state(self, initial_fixations, num_fix):
        # initialize ratios
        # rs[0, :] : ratio of intra-obj saccades in the early mode
        # rs[1, :] : ... trans-obj ... early mode
        # rs[2, :] : ... intra-obj ... late mode
        # rs[3, :] : ... trans-obj ... late mode
        rs = np.zeros((4, num_fix - 1), np.float)
        if initial_fixations in [(0, 1), (1, 0), (-1, 0), (-1, 1)]:
            # # start from the early mode and a trans-obj sac
            # rs[1, 0] = 1.0

            # # start from the early mode and the ratios according to p_intra
            # rs[0, 0] = self.p_intra_early
            # rs[1, 0] = 1 - self.p_intra_early

            # start from the early mode and the ratios according to the equilibrium ratio
            if self.p_intra_early + self.p_trans_early == 2:
                rs[0, 0] = 0.5
                rs[1, 0] = 0.5
            else:
                rs[0, 0] = (1 - self.p_trans_early) / (2 - self.p_intra_early - self.p_trans_early)
                rs[1, 0] = 1 - rs[0, 0]
        else:
            raise ValueError(
                "Specified initial fixation types are not supported.")

        # solve ratios
        for i in range(num_fix - 2):
            rs[:, i+1] = self.transmat.dot(rs[:, i])

        self.rs_hidden = rs

    def _gen_observable(self, initial_fixations):
        rs_hidden = self.rs_hidden
        p_err = self.p_error
        q_err = 1 - p_err

        # initialize the ratios of observed saccade types
        # rs[0, :] : ratio of intra-obj saccades
        # rs[1, :] : ... trans-obj saccades
        # rs[2, :] : ... object-to-background saccades
        # rs[3, :] : ... background-to-object saccades
        # rs[4, :] : ... background-to-background saccades
        rs = np.zeros((5, rs_hidden.shape[1]), np.float)
        if initial_fixations[0] in [0, 1]:
            rs[0, 0] = (rs_hidden[0, 0] + rs_hidden[2, 0]) * q_err
            rs[1, 0] = (rs_hidden[1, 0] + rs_hidden[3, 0]) * q_err
            rs[2, 0] = rs_hidden.sum(0)[0] * p_err
            rs[3, 0] = 0
            rs[4, 0] = 0
        elif initial_fixations[0] == -1:
            rs[0, 0] = 0
            rs[1, 0] = 0
            rs[2, 0] = 0
            rs[3, 0] = rs_hidden.sum(0)[0] * q_err
            rs[4, 0] = rs_hidden.sum(0)[0] * p_err

        rs[0, 1:] = (rs_hidden[0, 1:] + rs_hidden[2, 1:]) * q_err**2
        rs[1, 1:] = (rs_hidden[1, 1:] + rs_hidden[3, 1:]) * q_err**2
        rs[2, 1:] = rs_hidden.sum(0)[1:] * p_err * q_err
        rs[3, 1:] = rs_hidden.sum(0)[1:] * q_err * p_err
        rs[4, 1:] = rs_hidden.sum(0)[1:] * p_err**2

        self.sactype_ratios = rs

    def _gen_transprobs_obs(self):
        transmat = self.transmat
        rs = self.rs_hidden
        p_intra_obs = ((transmat[0, 0] + transmat[2, 0])*rs[0, :] + transmat[2, 2]*rs[2, :]) / (rs[0, :] + rs[2, :])
        p_trans_obs = ((transmat[1, 1] + transmat[3, 1])*rs[1, :] + transmat[3, 3]*rs[3, :]) / (rs[1, :] + rs[3, :])
        self.transprobs_obs = np.array((p_intra_obs, p_trans_obs))

    # old version of solving methods
    # def _solve_transprobs(self, num_fix):
    #     i = np.arange(1, num_fix - 1)
    #     q_switch = 1.0 - self.p_switch
    #     self.p_intra_sol = q_switch**i * self.p_intra_early + (1.0 - q_switch ** i) * self.p_intra_late
    #     self.p_trans_sol = q_switch**i * self.p_trans_early + (1.0 - q_switch ** i) * self.p_trans_late
    #
    # def _solve_sactype_ratios(self, initial_fixations, num_fix):
    #     p_error = self.p_error
    #     q_error = 1.0 - p_error
    #
    #     # initialize ratios
    #     rs = np.zeros((5, num_fix - 1), np.float)
    #     if initial_fixations in [(0, 1), (1, 0)]:
    #         rs[1, 0] = q_error
    #         rs[2, 0] = p_error
    #     elif initial_fixations in [(-1, 0), (-1, 1)]:
    #         rs[3, 0] = q_error
    #         rs[4, 0] = p_error
    #     else:
    #         raise ValueError("Specified initial fixation types are not supported.")
    #
    #     for i in range(num_fix - 2):
    #         if rs[0, i] + rs[1, i] == 0:
    #             r0_eff = rs[0, i]
    #             r1_eff = rs[1, i] + rs[3, i]
    #         else:
    #             r0_eff = rs[0, i] + rs[3, i] * rs[0, i] / (rs[0, i] + rs[1, i])
    #             r1_eff = rs[1, i] + rs[3, i] * rs[1, i] / (rs[0, i] + rs[1, i])
    #         rs[0, i+1] = (self.p_intra_sol[i] * r0_eff + (1.0 - self.p_trans_sol[i]) * r1_eff) * q_error
    #         rs[1, i+1] = (self.p_trans_sol[i] * r1_eff + (1.0 - self.p_intra_sol[i]) * r0_eff) * q_error
    #         rs[2, i+1] = (rs[0, i] + rs[1, i] + rs[3, i]) * p_error
    #         rs[3, i+1] = (rs[2, i] + rs[4, i]) * q_error
    #         rs[4, i+1] = (rs[2, i] + rs[4, i]) * p_error
    #     self.sactype_ratios = rs

    def solve(self, initial_fixations, num_fix):
        self._gen_transmat()
        self._gen_hidden_state(initial_fixations, num_fix)
        self._gen_observable(initial_fixations)

        # # old varsion
        # self.num_fix = num_fix
        # self._solve_transprobs(num_fix)
        # self._solve_sactype_ratios(initial_fixations, num_fix)

    def get_sactype_ratios(self):
        if self.sactype_ratios is None:
            raise ValueError("Saccade type ratios are not generated yet.")
        return self.sactype_ratios

    def get_transprobs_sol(self):
        self._gen_transprobs_obs()
        return self.transprobs_obs[:, :-1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # model parameters
    probs = [(0.1, 0.9), (0.6, 0.4)]  # p_intra, p_trans
    p_switch = 0.2
    p_error = 0.2

    # first_fixations = (0, 1)  # 1st fixation is on object and 1st saccade is trans-obj (monkey H)
    first_fixations = (-1, 1)  # 1st fixation is on background and 1st saccade is trans-obj (monkey S)

    # simulation parameters
    num_trial = 10000
    num_fix_per_trial = 25

    model = MarkovEyeMovementModel(probs, p_error, p_switch)

    # simulate the model
    sactypes = []
    for i_trial in range(num_trial):
        model.run_trial(first_fixations, num_fix_per_trial)
        sactypes_trial = model.get_sactypes()
        sactypes.extend(sactypes_trial)
    sactypes = np.array(sactypes)

    # solve the model
    model.solve(first_fixations, num_fix_per_trial)
    ratio_sactype_anal = model.get_sactype_ratios()

    # calculate values for plots
    sactypes = sactypes.reshape((num_trial, -1))
    sactypeIDs = [0, 1, 2, 3, 4]
    num_sactype = {sactypeID: [] for sactypeID in sactypeIDs}
    for sactypeID in sactypeIDs:
        for i in range(sactypes.shape[1]):
            mask_sactype = sactypes[:, i] == sactypeID
            num_sactype[sactypeID].append(mask_sactype.sum())
        num_sactype[sactypeID] = np.array(num_sactype[sactypeID])

    # plot saccade type counts
    sactype_labels = ['Intra-obj', 'Trans-obj', 'Obj-to-BG', 'BG-to-obj', 'BG-to-BG']
    sactype_colors = [(.9, .25, .2), (.35, .70, .9), (0, .6, .5), (.9, .6, 0), (.5, .5, .5)]
    fig = plt.figure(1, figsize=(7.5, 4.5))
    fig.subplots_adjust(left=0.12, right=0.95, bottom=0.15, top=0.9)
    ax_sim = fig.add_subplot(111)
    # ax_anal = fig.add_subplot(212)
    for sactypeID in sactypeIDs:
        # ax_sim.plot(num_sactype[sactypeID]/float(num_trial), label=sactype_labels[sactypeID], c=sactype_colors[sactypeID])
        ax_sim.plot(ratio_sactype_anal[sactypeID], label=sactype_labels[sactypeID], c=sactype_colors[sactypeID], lw=1, alpha=0.5)
    ax_sim.set_xlim(0, 20)
    ax_sim.set_ylim(0, 1)
    ax_sim.set_ylabel("Ratio")
    ax_sim.set_xlabel("Saccade order")
    ax_sim.grid(c='gray')
    # ax_anal.set_xlim(0, 20)
    # ax_anal.set_ylim(0, 1)
    # ax_anal.grid(c='gray')

    plt.show()
