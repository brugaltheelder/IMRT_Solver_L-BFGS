__author__ = 'troy'

from scipy import io
import scipy.sparse as sps
import scipy.optimize as spo
import numpy as np

class imrt_data(object):
    def __init__(self, workingDir, dataDir):
        #read in inputs, set others
        self.workingDir, self.dataDir = workingDir, dataDir
        inputs = [int(l) for l in open(self.workingDir+"inputs.txt","r").read().split()]
        self.nVox, self.nStructs, self.nBeams = inputs[0], inputs[1], inputs[2]
        self.beams = [i for i in range(self.nBeams)]
        self.nBPB = [int(l) for l in open(self.workingDir+"nBPB.txt","r").read().split()]
        self.nBPBcum = [0]+[i for i in np.cumsum(self.nBPB)]
        self.nDijPB = [int(l) for l in open(self.workingDir+"nDijs.txt","r").read().split()]
        self.nBeamlets = sum(self.nBPB)
        print 'Voxels: ', self.nVox, 'Structs: ', self.nStructs,'Beams: ', self.nBeams

        # read in dij matrices and weights
        self.dij = []
        for b in range(self.nBeams):
            Dij = np.fromfile(self.dataDir+'voxel'+str(self.beams[b])+'.bin',dtype=np.float32).reshape((self.nDijPB[b],3))
            self.dij.append(sps.csc_matrix((Dij[:][:,2],(Dij[:][:,0]-1,Dij[:][:,1]-1)),shape=(self.nBPB[b],self.nVox)).transpose())

        self.alpha = np.array([float(l) for l in open(self.workingDir+"alpha.txt","r").read().split()])
        self.beta = np.array([float(l) for l in open(self.workingDir+"beta.txt","r").read().split()])
        self.thresh = np.array([float(l) for l in open(self.workingDir+"thresh.txt","r").read().split()])

class imrt_model(object):
    def __init__(self, workingDir, dataDir):
        self.data = imrt_data( workingDir, dataDir)
        self.x0 = np.zeros(self.data.nBeamlets)

    def solve(self):
        self.fluence, self.obj, self.outDict = spo.fmin_l_bfgs_b(self.calcObjGrad, x0=self.x0.copy(), bounds = [(0,None) for i in xrange(self.data.nBeamlets)], disp=5)

    def calcDose(self, fluence):
        dose = np.zeros(self.data.nVox)
        for b in range(self.data.nBeams):
            dose += self.data.dij[b].dot(fluence[self.data.nBPBcum[b]:self.data.nBPBcum[b+1]])
        return dose

    def calcObjGrad(self,fluence):
        dose = self.calcDose(fluence)
        oDose, uDose = np.array(dose-self.data.thresh), np.array(dose-self.data.thresh)
        #obj = float(self.data.alpha.dot(oDose.clip(0).flatten()**2)+self.data.beta.dot(uDose.clip(-1e10,0).flatten()**2))
        obj = float(self.data.alpha.dot(oDose.clip(0)**2)+self.data.beta.dot(uDose.clip(-1e10,0)**2))
        grad = np.zeros(self.data.nBeamlets)
        zhat = np.multiply(self.data.alpha, oDose.clip(0)) + np.multiply(self.data.beta, uDose.clip(-1e10,0))
        for b in range(self.data.nBeams):
            grad[self.data.nBPBcum[b]:self.data.nBPBcum[b+1]] = self.data.dij[b].transpose().dot(zhat)

        return obj,grad

    def outputDose(self):
        dose = self.calcDose(self.fluence)
        io.savemat(self.data.workingDir+'out.mat',{'obj':self.obj,'fluence':self.fluence,'dose':dose})



#workingDir,dataDir = '','' # This is to run in current directory
#workingDir,dataDir = 'Prostate6/','Prostate6/'# 6 beam case
#workingDir,dataDir = 'Prostate180/','/media/troy/datadrive/Data/DataProject/Prostate/data/'# 6 beam case
workingDir,dataDir = 'HN394/','/media/troy/datadrive/Data/DataProject/HN/Dij/NCP/'# 6 beam case
model = imrt_model(workingDir,dataDir)
model.solve()
model.outputDose()