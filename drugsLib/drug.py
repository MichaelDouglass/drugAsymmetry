# drug class for initializing drugs

import numpy as np

def statesFromBits(bit, N):
    states = []
    for i in range(16):
        b = format(i, 'b')
        bB = b.zfill(N)
        states.append(bB)
    return states

# Define the Hamming Distance
def hamDist(Si, Sj):
    return sum(ei != ej for ei, ej in zip(Si, Sj))

# Make a drug class
class Drug:
    # This will make it easy to call certain qualities
    # Arguments:
    # name is a string
    # Fit is an array of the fitness values
    # States is the binary genotype that corresponds to the bins
    # R is the biasing variable that we set to 0
    def __init__(self, name, Fit, states, LFP, R=0):
        # @par
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
