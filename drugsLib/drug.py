import numpy as np

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
        LPFf (float): fitness value of LPF states
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
        self.LPFf = self.Fit[int(self.LFP, 2)]
