import numpy as np

class MinimizationProblem:
    NumObjCall = 0

    def __init__(self, num_variable, bounds, penaltyFactor = 1e6):
        self.num_variable = num_variable
        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds)
        self.VariableBounds = bounds
        self.PenaltyFactor = penaltyFactor
    
    def DefaultPopluationInit(self, N):
        return self.Repair(np.random.rand(N, len(self.VariableBounds)) * (self.VariableBounds[:,1] - self.VariableBounds[:,0]) + self.VariableBounds[:,0])
    
    def ObjectiveFunction(self, x) -> float:
        raise NotImplementedError
    
    def Constraints(self, x):
        raise NotImplementedError
    
    def FullObjFunction(self, x):
        raise NotImplementedError
    
    def Repair(self, x):
        x = np.maximum(x, self.VariableBounds[:,0])
        x = np.minimum(x, self.VariableBounds[:,1])
        return x
    
    def Check(self, x):
        assert len(x) == self.num_variable \
                and (isinstance(self.PenaltyFactor, float) or isinstance(self.PenaltyFactor, int) or len(self.PenaltyFactor) == self.num_variable)

class PressureVesselProblem(MinimizationProblem):
    def __init__(self, penaltyFactor = 1e6):
        '''
            x1: discrete (multiple of 0.0625)
                Ts (thickness of the shell) 
                Range: [1.1, 99]
                
            x2: discrete (multiple of 0.0625)
                Th (thickness of the head)
                Range: [0.6, 99]

            x3: continuous
                R (inner radius)
                Range: [50, 70]

            x4: continuous
                L (length of the cylindrical section of the vessel, not including the head)
                Range: [30, 50]
        '''
        # super().__init__(4, [[1.125,99],[0.625,99],[50,70],[30,50]], penaltyFactor)
        super().__init__(4, [[0.0625,99*0.0625],[0.0625,99*0.0625],[10,200],[10,200]], penaltyFactor)

    def ObjectiveFunction(self, x) -> float:
        assert len(x) == self.num_variable \
                and (isinstance(self.PenaltyFactor, float) or isinstance(self.PenaltyFactor, int) or len(self.PenaltyFactor) == self.num_variable)
        # obj = 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]*x[2] + 3.1661*x[0]*x[0]*x[3] + 19.8621*x[0]*x[0]*x[2]
        PressureVesselProblem.NumObjCall += 1
        obj = 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]*x[2] + 3.1661*x[0]*x[0]*x[3] + 19.84*x[0]*x[0]*x[2]
        constraint = self.Constraints(x)
        return obj + self.PenaltyFactor * np.abs(constraint[constraint > 0]).sum()
    
    def Constraints(self, x):
        return np.array([-x[0]+0.0193*x[2], -x[1]+0.00954*x[2], -np.pi*(x[2]*x[2]*x[3]+4/3*x[2]*x[2]*x[2])+1296000, x[3]-240])

    def Repair(self, x):
        x[[0,1]] = 0.0625 * np.trunc(x[[0,1]]/0.0625)
        return super().Repair(x)

    def FullObjFunction(self, x):
        assert len(x) == self.num_variable \
                and (isinstance(self.PenaltyFactor, float) or isinstance(self.PenaltyFactor, int) or len(self.PenaltyFactor) == self.num_variable)
        # obj = 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]*x[2] + 3.1661*x[0]*x[0]*x[3] + 19.8621*x[0]*x[0]*x[2]
        obj = 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]*x[2] + 3.1661*x[0]*x[0]*x[3] + 19.84*x[0]*x[0]*x[2]
        constraint = self.Constraints(x)
        return obj, constraint

class SpringDesignProblem(MinimizationProblem):
    def __init__(self, penaltyFactor = 1e6):
        '''
        x1: contiunuous
            d (wire diameter)
            Range: [0.05, 0.2]

        x2: continuous
            D (mean coil diameter)
            Range: [0.25, 1.3]

        x3: continuous
            N (number of active coils)
            Range: [2, 15]
        
        '''
        super().__init__(3, [[0.05, 0.2], [0.25, 1.3], [2, 15]] , penaltyFactor)
    
    def ObjectiveFunction(self, x) -> float:
        assert len(x) == self.num_variable \
                and (isinstance(self.PenaltyFactor, float) or isinstance(self.PenaltyFactor, int) or len(self.PenaltyFactor) == self.num_variable)
        SpringDesignProblem.NumObjCall += 1
        obj = (x[2]+2)*x[1]*x[0]*x[0]
        constraint = self.Constraints(x)
        return obj + self.PenaltyFactor * np.abs(constraint[constraint > 0]).sum()

    def Constraints(self, x):
        return np.array([
                            1-x[1]*x[1]*x[1]*x[2]/71875/x[0]**4, # Deflection constraint
                            (4*x[1]*x[1]-x[0]*x[1])/(4000*np.pi*(x[1]*x[0]**3-x[0]**4))+0.615/np.pi/1000/x[0]/x[0]-1, # Shear stress
                            1 - np.sqrt(1.15*1e11/14.76684)/200/np.pi*x[0]/x[1]/x[1]/x[2], # Frequency of surge waves
                            (x[1]+x[0])/1.5-1 # Diameter constraint
                        ])
    
    def FullObjFunction(self, x):
        assert len(x) == self.num_variable \
                and (isinstance(self.PenaltyFactor, float) or isinstance(self.PenaltyFactor, int) or len(self.PenaltyFactor) == self.num_variable)
        # obj = 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]*x[2] + 3.1661*x[0]*x[0]*x[3] + 19.8621*x[0]*x[0]*x[2]
        obj = (x[2]+2)*x[1]*x[0]*x[0]
        constraint = self.Constraints(x)
        return obj, constraint

class GearTrainDesignProblem(MinimizationProblem):
    def __init__(self, penaltyFactor=1000000):
        '''
            x1~x4: discrete (integer)
                   teeth number
                   Range: [12, 60]
        '''
        super().__init__(4, [[12, 60]]*4 , penaltyFactor)
    
    def ObjectiveFunction(self, x) -> float:
        assert len(x) == self.num_variable \
                and (isinstance(self.PenaltyFactor, float) or isinstance(self.PenaltyFactor, int) or len(self.PenaltyFactor) == self.num_variable)
        GearTrainDesignProblem.NumObjCall += 1
        obj = (1/6.931-x[0]*x[1]/x[2]/x[3])
        return obj*obj
    
    def FullObjFunction(self, x):
        assert len(x) == self.num_variable \
                and (isinstance(self.PenaltyFactor, float) or isinstance(self.PenaltyFactor, int) or len(self.PenaltyFactor) == self.num_variable)
        return self.ObjectiveFunction(x), None

    def Repair(self, x):
        return super().Repair(np.fix(x))

class WeldedBeamDesignProblem(MinimizationProblem):
    def __init__(self, penaltyFactor=1000000):
        '''
            x1: continuous
                h (thickness of weld)
                Range: [0.1, 2]

            x2: continuous
                l (length of the clamped bar)
                Rnage: [0, 10]
            
            x3: continuous
                t (height of the bar)
                Range: [0.1, 10]

            x4: continuous
                b (thickness of the bar)
                Range: [0.125, 2]
        '''
        super().__init__(4, [[0.125, 2], [0, 10], [0, 10], [0.125, 2]], penaltyFactor)
    
    def ObjectiveFunction(self, x) -> float:
        self.Check(x)
        obj = (1+0.37*0.283)*x[0]*x[0]*x[1] + (0.17*0.283)*x[2]*x[3]*(14+x[1])
        constraint = self.Constraints(x)
        return obj + self.PenaltyFactor * np.abs(constraint[constraint > 0]).sum()
    
    def Constraints(self, x):
        P = 6000
        L = 14
        dmax = 0.25
        E = 30*1e6
        G = 12*1e6
        tmax = 13600
        sigmax = 30000
        M = P*(L+x[1]/2)
        R = np.sqrt(x[1]*x[1]/4+(x[0]+x[2])**2/4)
        J = np.sqrt(2)*x[0]*x[1]*(x[1]*x[1]/12+((x[0]+x[2])/2)**2)
        sig = 6*P*L/x[3]/x[2]/x[2]
        I = 1/12*x[2]*x[3]**3
        alpha = 1/3*G*x[2]*x[3]**3
        Pc = 4.013*np.sqrt(E*I*alpha)/L/L*(1-x[2]/2/L*np.sqrt(E*I/alpha))
        tau_ = P/np.sqrt(2)/x[0]/x[1]
        tau__ = M*R/J
        tau = np.sqrt(tau_*tau_+2*tau_*tau__*x[1]/2/R+tau__*tau__)
        DEL = 4*P*L**3/(E*x[2]**3*x[3])

        return np.array([tau-tmax, sig-sigmax, x[0]-x[3], P - Pc, DEL - dmax])
    
    def FullObjFunction(self, x):
        self.Check(x)
        obj = (1+0.37*0.283)*x[0]*x[0]*x[1] + (0.17*0.283)*x[2]*x[3]*(14+x[1])
        constraint = self.Constraints(x)
        return obj, constraint