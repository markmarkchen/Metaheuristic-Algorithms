import numpy as np
import pip
from Logger import logger
import time
import datetime
from Solver import ALO, ABC, BA, CS, DE, FA, FPA, GSA, GWO, HS, SFLA, SCA, TLBO, WOA, SA, PSO, GA, ICA, MA

from Benchmark import GearTrainDesignProblem, PressureVesselProblem, SpringDesignProblem, WeldedBeamDesignProblem
if __name__ == "__main__":
    # k = 0
    # F = lambda x: np.sum((x+k)**2-10*np.cos(2*np.pi*(x+k))+10)
    # Bounds = np.array([[-5.12+k,5.12+k] for _ in range(200)])
    # F = lambda x: np.sum((x+k)**2)
    # Bounds = np.array([[-100,100] for _ in range(30)])
    # F = lambda x:-20*np.exp(-0.2*np.sqrt(sum((x+k)**2)/len(x)))-np.exp(sum(np.cos(2*np.pi*(x+k)))/len(x))+20+np.e
    # Bounds = np.array([[-32,32] for _ in range(200)])

    # WeldedBeamDesign = WeldedBeamDesignProblem(100)
    # Population = WeldedBeamDesign.DefaultPopluationInit(100)
    # Population30 = WeldedBeamDesign.DefaultPopluationInit(30)
    # Population150_sqrt = WeldedBeamDesign.DefaultPopluationInit(int(np.sqrt(150)))
    # Bounds = WeldedBeamDesign.VariableBounds
    # F = WeldedBeamDesign.ObjectiveFunction
    # R = WeldedBeamDesign.Repair
    # E = WeldedBeamDesign.FullObjFunction

    # print(E([0.205396, 3.484293, 9.037426, 0.206276]))


    from scheduler import Pipeline
    import pickle

    # PressureVessel = PressureVesselProblem(1e6)
    # Population = PressureVessel.DefaultPopluationInit(100)
    # Population30 = PressureVessel.DefaultPopluationInit(30)
    # Population150_sqrt = PressureVessel.DefaultPopluationInit(int(np.sqrt(150)))
    # Bounds = PressureVessel.VariableBounds
    # F = PressureVessel.ObjectiveFunction
    # R = PressureVessel.Repair
    # E = PressureVessel.FullObjFunction

    SpringDesign = SpringDesignProblem(100)
    Population = SpringDesign.DefaultPopluationInit(100)
    Population30 = SpringDesign.DefaultPopluationInit(30)
    Population150_sqrt = SpringDesign.DefaultPopluationInit(int(np.sqrt(150)))
    Bounds = SpringDesign.VariableBounds
    F = SpringDesign.ObjectiveFunction
    R = SpringDesign.Repair
    E = SpringDesign.FullObjFunction
    print(E([0.05169,0.35673,11.2885]))

    # GearTrainDesign = GearTrainDesignProblem(100)
    # Population = GearTrainDesign.DefaultPopluationInit(100)
    # Population30 = GearTrainDesign.DefaultPopluationInit(30)
    # Population150_sqrt = GearTrainDesign.DefaultPopluationInit(int(np.sqrt(150)))
    # Bounds = GearTrainDesign.VariableBounds
    # F = GearTrainDesign.ObjectiveFunction
    # R = GearTrainDesign.Repair
    # E = GearTrainDesign.FullObjFunction

    ALO = ALO(NumberOfIteration = 2000, AntRatio = 3)
    ABC = ABC(NumberOfIteration = 2000, Limits = 100, OnlookerRatio = 10)
    BA = BA(NumberOfIteration = 2000)
    CS = CS(NumberOfIteration = 2000, Alpha = 0.001)
    DE = DE(NumberOfIteration = 2000, variant="DE/rand/1/exp")
    FA = FA(NumberOfIteration = 2000)
    FPA = FPA(NumberOfIteration = 2000)
    GSA = GSA(NumberOfIteration = 2000)
    GWO = GWO(NumberOfIteration = 2000)
    HS = HS(NumberOfIteration = 2000)
    SFLA = SFLA(NumberOfIteration = 10000)
    SCA = SCA(NumberOfIteration = 2000)
    TLBO = TLBO(NumberOfIteration = 2000)
    WOA = WOA(NumberOfIteration = 2000)
    SA = SA(NumberOfIteration = 2000)
    PSO = PSO(NumberOfIteration = 2000, Cognitive = 1.5, Social = 1.5)
    GA = GA(NumberOfIteration = 2000, Crossover= "2pt", Selection = "3Tournament")
    ICA = ICA(NumberOfIteration = 2000, NumberOfEmpires = 5)
    MA = MA(NumberOfIteration = 1000, LocalSearchMaxIter = 5, Crossover= "2pt", Selection = "3Tournament")
    # pipe = Pipeline(SpringDesign, [Population30], Solver=[GA,DE], Repeat=30)
    
    Solver = [ALO, ABC, BA, CS, DE, FA, FPA, GSA, GWO, HS, SFLA, SCA, TLBO, WOA, SA, PSO, GA, ICA, MA]

    PressureVessel = PressureVesselProblem(1e6)
    Population30 = PressureVessel.DefaultPopluationInit(30)
    pipe = Pipeline(PressureVessel, [Population30], Solver=Solver, Repeat=30)
    pipe.Run()
    pipe.Save()

    SpringDesign = SpringDesignProblem(100)
    Population30 = SpringDesign.DefaultPopluationInit(30)
    pipe = Pipeline(SpringDesign, [Population30], Solver=Solver, Repeat=30)
    pipe.Run()
    pipe.Save()

    GearTrainDesign = GearTrainDesignProblem(100)
    Population30 = GearTrainDesign.DefaultPopluationInit(30)
    pipe = Pipeline(GearTrainDesign, [Population30], Solver=Solver, Repeat=30)
    pipe.Run()
    pipe.Save()


    exit()

    

    

    # Population = DefaultUniformInitialization(1000, Bounds)

    np.seterr(all='raise')
    
    ALO = ALO(NumberOfIteration = 10000, ObjectiveFunction = F, AntRatio = 3, ExplicitObjFunc = E, Repair = R)
    ALO.VariableBounds = Bounds
    ALO.Population = np.copy(Population)
    ALO.Run()

    ABC = ABC(NumberOfIteration = 2000, ObjectiveFunction = F, ExplicitObjFunc = E, Limits = 100, OnlookerRatio = 10, Repair = R)
    ABC.VariableBounds = Bounds
    ABC.Population = np.copy(Population)
    ABC.Run()

    BA = BA(NumberOfIteration = 2000, ObjectiveFunction = F, ExplicitObjFunc = E, Repair = R)
    BA.VariableBounds = Bounds
    BA.Population = np.copy(Population30)
    BA.Run()

    CS = CS(NumberOfIteration = 10000, ObjectiveFunction = F, Repair = R, ExplicitObjectiveFunction = E, Alpha = 0.001)
    CS.VariableBounds = Bounds
    CS.Population = np.copy(Population)
    CS.Run()

    DE = DE(NumberOfIteration = 500, ObjectiveFunction = F, variant="DE/rand/1/exp", Repair = R, ExplicitObjFunc = E)
    DE.VariableBounds = Bounds
    DE.Population = np.copy(Population)
    DE.Run()

    FA = FA(NumberOfIteration = 2000, ObjectiveFunction = F, ExplicitObjFunc = E, Repair = R)
    FA.VariableBounds = Bounds
    FA.Population = np.copy(Population150_sqrt)
    FA.Run()

    FPA = FPA(NumberOfIteration = 2000, ObjectiveFunction = F, ExplicitObjFunc = E, Repair = R)
    FPA.VariableBounds = Bounds
    FPA.Population = np.copy(Population)
    FPA.Run()

    GSA = GSA(NumberOfIteration = 2000, ObjectiveFunction = F, ExplicitObjFunc = E, Repair = R)
    GSA.VariableBounds = Bounds
    GSA.Population = np.copy(Population150_sqrt)
    GSA.Run()

    GWO = GWO(NumberOfIteration = 10000, ObjectiveFunction = F, Repair = R, ExplicitObjFunc = E)
    GWO.VariableBounds = Bounds
    GWO.Population = np.copy(Population)
    GWO.Run()

    HS = HS(NumberOfIteration = 2000, ObjectiveFunction = F, ExplicitObjFunc = E, Repair = R)
    HS.VariableBounds = Bounds
    HS.Population = np.copy(Population)
    HS.Run()

    SFLA = SFLA(NumberOfIteration = 10000, ObjectiveFunction = F, ExplicitObjFunc = E, Repair = R)
    SFLA.VariableBounds = Bounds
    SFLA.Population = np.copy(Population)
    SFLA.Run()

    SCA = SCA(NumberOfIteration = 2000, ObjectiveFunction = F, ExplicitObjFunc = E, Repair = R)
    SCA.VariableBounds = Bounds
    SCA.Population = np.copy(Population)
    SCA.Run()
    
    TLBO = TLBO(NumberOfIteration = 2000, ObjectiveFunction = F, ExplicitObjFunc = E, Repair = R)
    TLBO.VariableBounds = Bounds
    TLBO.Population = np.copy(Population)
    TLBO.Run()
    
    WOA = WOA(NumberOfIteration = 10000, ObjectiveFunction = F, ExplicitObjFunc = E, Repair = R)
    WOA.VariableBounds = Bounds
    WOA.Population = np.copy(Population)
    WOA.Run()
    
    SA = SA(NumberOfIteration = 10000, ObjectiveFunction = F, ExplicitObjFunc = E, Repair = R)
    SA.VariableBounds = Bounds
    SA.Population = np.copy(Population)
    SA.Run()

    PSO = PSO(NumberOfIteration = 10000, ObjectiveFunction = F, Cognitive = 1.5, Social = 1.5, ExplicitObjFunc = E, Repair = R)
    PSO.VariableBounds = Bounds
    PSO.Population = np.copy(Population)
    PSO.Run()

    GA = GA(NumberOfIteration = 1500, ObjectiveFunction = F, Crossover= "2pt", Selection = "3Tournament", ExplicitObjFunc = E, Repair = R)
    GA.VariableBounds = Bounds
    GA.Population = np.copy(Population)
    GA.Run()

    ICA = ICA(NumberOfIteration = 1500, ObjectiveFunction = F, NumberOfEmpires = 100, ExplicitObjFunc = E, Repair = R)
    ICA.VariableBounds = Bounds
    ICA.Population = np.copy(Population)
    ICA.Run()

    MA = MA(NumberOfIteration = 1500, ObjectiveFunction = F,LocalSearchMaxIter = 20, Crossover= "2pt", Selection = "3Tournament", ExplicitObjFunc = E, Repair = R)
    MA.VariableBounds = Bounds
    MA.Population = np.copy(Population)
    MA.Run()