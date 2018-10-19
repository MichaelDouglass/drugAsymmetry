import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib as mpl
import os

def statesFromBits(bit, N):
    """Creates binary strings from number input

    Args:
        bit (int): length of array
        N (int): Zero padding returned
    Returns:
        states (list): list of binary strings from 0 to bit-1
    """
    states = []
    for i in range(bit):
        b = format(i, 'b')
        bB = b.zfill(N)
        states.append(bB)
    return states


def hamDist(Si, Sj):
    """ Return the Hamming distance between two bits

    Args:
        Si (string): binary value
        Sj (string): binary value
    Returns:
        Hamming Distance
    """
    return sum(ei != ej for ei, ej in zip(Si, Sj))

def whatsInf(T):
    c = 0
    P = np.ones(np.size(T))
    while not np.array_equal(P, T):
        P = T
        T = T*T
        c = c + 1
    return [P, c]

def csvExplore(Drug1, Drug2, Drug3, f0=None, f1=None, finf=None, cols=None):
    """Used to show all data from given csv file_extension

    Shows 3x3 plot of steering drug inputs
    Columns are the amount of times the third drug is Applied
    (0 Times, 1 Time, Infinite Times)
    Rows are lowest fitness achieved, rho, and Theta

    Args:
        Drug1 (drug class): First steering drug
        Drug2 (drug class): Second steering drug
        Drug3 (drug class): Final drug
    Returns:
    """
    nameD1 = Drug1.name
    nameD2 = Drug2.name
    nameD3 = Drug3.name

    if cols == None:
        col_names=['Final Drug', 'Steering Drug 1', 'Steering Drug 1 Amount', 'Steering Drug 2', 'Steering Drug 2 Amount',
                   'Drug1:Drug2', 'Drug1+Drug2', 'Fitness of infinite apps', 'Lowest Fitness Achieved',
                   'Maximum Simulated Drug Application', 'R-Value', 'epsilon', 'rho', 'theta']
    else:
        col_names = cols

    if f0 == None:
        filename = 'R0-10_AllDrugs_OnlyCountedWithin95perc_app10_eps_NoThirdInf.csv'
    else:
        filename = f0

    df0 = pd.read_csv(filename, names=col_names)
    df0 = df0[df0['Final Drug'] == nameD3]
    df0 = df0[(df0['Steering Drug 1'] == nameD1) & (df0['Steering Drug 2'] == nameD2)]

    if f1 == None:
        filename = 'R0-10_AllDrugs_OnlyCountedWithin95perc_app10_eps_ThirdOnceInf.csv'
    else:
        filename = f1
    df1 = pd.read_csv(filename, names=col_names)
    df1 = df1[df1['Final Drug'] == nameD3]
    df1 = df1[(df1['Steering Drug 1'] == nameD1) & (df1['Steering Drug 2'] == nameD2)]

    if finf == None:
        filename = 'R0-10_AllDrugs_OnlyCountedWithin95perc_app10_eps_ThirdInf.csv'
    else:
        filename = finf
    dfInf = pd.read_csv(filename, names=col_names)
    dfInf = dfInf[dfInf['Final Drug'] == nameD3]
    dfInf = dfInf[(dfInf['Steering Drug 1'] == nameD1) & (dfInf['Steering Drug 2'] == nameD2)]

    dataFrames = [df0, df1, dfInf]
    fig = plt.figure(figsize=(12,12))
    count = 331

    try:
        yminF = min([min(df0['Lowest Fitness Achieved']), min(df1['Lowest Fitness Achieved']), min(dfInf['Lowest Fitness Achieved'])])
    except:
        yminF = 0

    try:
        ymaxF = max([max(df0['Lowest Fitness Achieved']), max(df1['Lowest Fitness Achieved']), max(dfInf['Lowest Fitness Achieved'])])
    except:
        ymaxF = 3


    for df in dataFrames:
        ax = fig.add_subplot(count)
        ax.plot(df['R-Value'], df['Lowest Fitness Achieved'])
        if count == 331:
            ax.set_title('Third Drug 0 Applications')
            ax.set_ylabel('Lowest Fitness Achieved')
        if count == 332:
            ax.set_title('Third Drug 1 Applications')
        if count == 333:
            ax.set_title('Third Drug Inf Applications')
        ax.set_ylim([yminF, ymaxF])

        spot2 = count + 3

        ax = fig.add_subplot(spot2)
        ax.plot(df['R-Value'], df['rho'])
        if spot2 == 334:
            ax.set_ylabel('Rho')
        ax.set_ylim([0, 10])

        spot3 = count + 6

        ax = fig.add_subplot(spot3)
        ax.plot(df['R-Value'], df['theta']/(np.pi/180))
        if spot3 == 337:
            ax.set_ylabel('Theta (deg)')
        if spot3 == 338:
            ax.set_xlabel('R-Value')
        ax.set_ylim([0, 180])

        count = count + 1
        plt.suptitle(nameD1+'$\Rightarrow$'+nameD2+'$\Rightarrow$'+nameD3, fontweight='bold', fontsize='x-large')
    return [df0, df1, dfInf, yminF, ymaxF]


def matrixPlotter(DRUG1, DRUG2, DRUG, yminF=0, ymaxF=3.4, mod=None, Ap=10, S0=None, states=None):
    """Used to show plot fitness values as a matrix when drug sequence applied

    Args:
        DRUG1 (drug class): First steering drug
        DRUG2 (drug class): Second steering drug
        DRUG (drug class): Final drug
    Returns:
    """
    if states == None:
        states = statesFromBits(16, 4)
    else:
        states = states

    if S0 == None:
        S0 = np.array(np.ones(len(states))/2**4)
    else:
        S0 = S0

    cmap = plt.cm.inferno
    norm = mpl.colors.BoundaryNorm(np.arange(yminF,ymaxF,0.01), cmap.N)

    avgMat = np.zeros([Ap+1, Ap+1])
    if mod == None:
        for NA in range(0,Ap+1):
            for NB in range(0,Ap+1):
                SN = np.array(S0) * DRUG1.tMat**NA * DRUG2.tMat**NB * DRUG.tMat**100
                AvgFit = np.dot(np.array(SN[0,:]), DRUG.Fit)
                avgMat[NA, NB] = AvgFit
    if mod == -1:
        for NA in range(0,Ap+1):
            for NB in range(0,Ap+1):
                if NA == 0 and NB == 0:
                    SN = np.array(S0) * DRUG.tMat**100
                elif NA == 0:
                    SN = np.array(S0) * DRUG2.tMat**NB * DRUG.tMat**100
                elif NB == 0:
                    SN = np.array(S0) * DRUG1.tMat**NA * DRUG.tMat**100
                else:
                    SN = np.array(S0) * DRUG1.tMat**NA * DRUG2.tMat**NB * DRUG.tMat**100
                AvgFit = np.dot(np.array(SN[0,:]), DRUG.Fit)
                avgMat[NA, NB] = AvgFit
    if mod == 1:
        for NA in range(0,Ap+1):
            for NB in range(0,Ap+1):
                SN = np.array(S0) * DRUG1.tMat**NA * DRUG2.tMat**NB * DRUG.tMat**1
                AvgFit = np.dot(np.array(SN[0,:]), DRUG.Fit)
                avgMat[NA, NB] = AvgFit
    if mod == 0:
        for NA in range(0,Ap+1):
            for NB in range(0,Ap+1):
                SN = np.array(S0) * DRUG1.tMat**NA * DRUG2.tMat**NB
                AvgFit = np.dot(np.array(SN[0,:]), DRUG.Fit)
                avgMat[NA, NB] = AvgFit

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    Z = ax.matshow(avgMat, cmap=cmap, norm=norm)
    ax.set_xlabel(DRUG1.name + ' Applied Second N Times')
    ax.set_ylabel(DRUG2.name + ' Applied First N Times')
    ax.set_title('Average Fitness <f> for ' + DRUG.name + ' at R = '+str(DRUG.R))
    plt.colorbar(Z)
    plt.tight_layout()


def avgFitAnimator(DRUG1, DRUG2, DRUG, R, yminF=0, ymaxF=3.4, mod=None, Ap=10, S0=None, states=None):
    """
    """
    if not os.path.exists('animations'):
        os.mkdir('animations')

    if states == None:
        states = statesFromBits(16, 4)
    else:
        states = states

    if S0 == None:
        S0 = np.array(np.ones(len(states))/2**4)
    else:
        S0 = S0

    cmap = plt.cm.inferno
    norm = mpl.colors.BoundaryNorm(np.arange(yminF,ymaxF,0.01), cmap.N)

    avgMat = np.zeros([Ap+1, Ap+1])

    figs = plt.figure(num=0)
    ims = []
    for r in R:
        DRUG1.R = r
        DRUG1.redoMat()
        DRUG2.R = r
        DRUG2.redoMat()
        DRUG.R = r
        DRUG.redoMat()

        if mod == None:
            for NA in range(0,Ap+1):
                for NB in range(0,Ap+1):
                    SN = np.array(S0) * DRUG1.tMat**NA * DRUG2.tMat**NB * DRUG.infMat
                    AvgFit = np.dot(np.array(SN[0,:]), DRUG.Fit)
                    avgMat[NA, NB] = AvgFit
        if mod == 1:
            for NA in range(0,Ap+1):
                for NB in range(0,Ap+1):
                    SN = np.array(S0) * DRUG1.tMat**NA * DRUG2.tMat**NB * DRUG.tMat**1
                    AvgFit = np.dot(np.array(SN[0,:]), DRUG.Fit)
                    avgMat[NA, NB] = AvgFit
        if mod == 0:
            for NA in range(0,Ap+1):
                for NB in range(0,Ap+1):
                    SN = np.array(S0) * DRUG1.tMat**NA * DRUG2.tMat**NB
                    AvgFit = np.dot(np.array(SN[0,:]), DRUG.Fit)
                    avgMat[NA, NB] = AvgFit

        mat = plt.matshow(avgMat, cmap=cmap, norm=norm, fignum=0, animated=True)
        # plt.set_xlabel(DRUG1.name + ' Applied Second N Times')
        # plt.set_ylabel(DRUG2.name + ' Applied First N Times')
        plt.suptitle('Average Fitness <f> for ' + DRUG.name + 'R-Value = '+str(R[0])+' to '+str(r))
        # plt.colorbar(mat)
        ims.append([mat])
    ani = animation.ArtistAnimation(figs, ims, interval=1000, blit=True,
                                    repeat_delay=5000)
    ani.save('animations/test.gif', writer='imagemagick')
    plt.close()

class Drug:
    """ Drug class main takes fitness landscape and returns probability matrix.

    Args:
        name (string): Full name of the Drug
        Fit (list or array): Fitness values for drug corresponding to the
                             states given (binary)
        States (list): binary value strings must be same length as Fitness
        LFP (string): binary genotype corresponding to lowest fitness
                      peak.
        R (float): biasing variable described in Markov model
    Attributes:
        LFPf (float): fitness value of LPF states
        tMat(matrix): probability matrix from markov model of dimensions
                      [states x states]
    """
    def __init__(self, name, Fit, states, LFP, R=0):
        self.name = name
        self.Fit = np.array(Fit)
        self.states = states
        self.LFP = LFP
        self.R = R

        # The following code is to calculate the transition matrix
        P = np.zeros([len(self.states),len(self.states)])

        for i in range(len(self.states)):
            for j in range(len(self.states)):
                if self.Fit[j] >= self.Fit[i] and hamDist(self.states[i], self.states[j]) == 1:
                    P[i, j] = (self.Fit[j] - self.Fit[i])**R
                    d = 0
                    for s in range(len(self.states)):
                        if hamDist(self.states[i], self.states[s]) == 1 and self.Fit[s] > self.Fit[i]:
                            d = d + (self.Fit[s] - self.Fit[i])**R
                    P[i, j] = P[i, j]/d


        # if there is no probability to reach any other state,
        # there is 100% probability to remain in the initial state.
        for p in range(len(P)):
            spots = np.where(P[p] != 0)
            if len(spots[0])==0:
                P[p, p]=1

        self.tMat = np.matrix(P)
        self.LFPf = self.Fit[int(self.LFP, 2)]
        PC = whatsInf(self.tMat)
        self.infMat = PC[0]
        self.infCount = PC[1]
    def redoMat(self):
        R = self.R
        P = np.zeros([len(self.states),len(self.states)])

        for i in range(len(self.states)):
            for j in range(len(self.states)):
                if self.Fit[j] >= self.Fit[i] and hamDist(self.states[i], self.states[j]) == 1:
                    P[i, j] = (self.Fit[j] - self.Fit[i])**R
                    d = 0
                    for s in range(len(self.states)):
                        if hamDist(self.states[i], self.states[s]) == 1 and self.Fit[s] > self.Fit[i]:
                            d = d + (self.Fit[s] - self.Fit[i])**R
                    P[i, j] = P[i, j]/d


        # if there is no probability to reach any other state,
        # there is 100% probability to remain in the initial state.
        for p in range(len(P)):
            spots = np.where(P[p] != 0)
            if len(spots[0])==0:
                P[p, p]=1

        self.tMat = np.matrix(P)
