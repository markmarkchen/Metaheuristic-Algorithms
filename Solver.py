import numpy as np
import pip
from Logger import logger
import time
import datetime

def DefaultUniformInitialization(N, Bounds):
        return np.random.rand(N, len(Bounds)) * (Bounds[:,1] - Bounds[:,0]) + Bounds[:,0]

class GenericMA:
    def __init__(self, **kwargs) -> None:
        '''
        Attributes
        ----------
        NumberOfIteration   -> int : the maximum iteration for terminal condition.
        VariableBounds      -> list in the shape (D, 2) : the lower and upper bound for the D design variables
        Population          -> list in the shape (N, D)
        Objectivevalurs     -> list in the shape (N, 1)
        BestSol             -> list in the shape (D,)
        BestObj             -> double
        Logger              -> Logger
        Log                 -> list

        Methods
        -------
        ObjectiveFunction : objective function for the minimization problem
            -Input: solution in the shape (D,)
            -Output: double
        ExplicitObjFunction: return the objective value and violation amounts of each constraints
            -Input: solution in the shape (D,)
            -Output: double, (D*double,)
            
        Copy : deep copy of a solution
            -Input: solution in the shape (D,)
            -Output: solution in the shape (D,)

        Repair : repair a solution
            -Input: solution in the shape (D,)
            -Output: solution in the shape (D,)
        '''
        self.NumberOfIteration = kwargs.get("NumberOfIteration", 100) 
        self.ObjectiveFunction = kwargs.get("ObjectiveFunction", lambda x:sum(y**2 for y in x)) 
        self.ExplicitObjFunc = kwargs.get("ExplicitObjFunc", lambda x:(sum(y**2 for y in x), None)) 
        self.VariableBounds = kwargs.get("VarialbeBounds", []) 

        self.Population = kwargs.get("Population", None) 
        self.ObjectiveValues = []

        self.BestSol = []
        self.BestObj = np.inf
        self._IterBestSol = []
        self._IterBestObj = np.inf
        
        self.Logger = kwargs.get("Logger", logger())
        self.Log = []

        self.Initialization = kwargs.get("Initialization", self.DefaultInitialization)
        self.Reset = kwargs.get("Reset", self.DefaultReset)
        self.Copy = kwargs.get("Copy", np.copy)
        self.Repair = kwargs.get("Repair", self.DefaultRepair)

        self._iteration = 0

    def DefaultReset(self):
        self._iteration = 0
        self.Population = []
        self.ObjectiveValues = []

        self.BestSol = []
        self.BestObj = np.inf
        self._IterBestSol = []
        self._IterBestObj = np.inf

        self.Log = []

    def DefaultInitialization(self):
        self.ObjectiveValues = np.array(list(map(self.ObjectiveFunction,self.Population)))
        index = np.argmin(self.ObjectiveValues)
        self.BestObj = self.ObjectiveValues[index]
        self.BestSol = self.Copy(self.Population[int(index)])

    def Iteration(self):
        pass

    def Run(self) -> None:
        # iter 0
        self.Initialization()
        log, _ = self.Logger.Log(self.Copy, self.BestSol, self.BestObj
                ,self._IterBestSol, self._IterBestObj, self.Population, self.ObjectiveValues)
        self.Log.append(log)
        length = 0
        # iter 1 to NumberOfIteration
        self._iteration = 1
        now = time.time()
        print(f"{type(self).__name__} starts at {datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')}".center(100,"="))
        try:
            while self._iteration <= self.NumberOfIteration:
                self.Iteration()
                log, logstr = self.Logger.Log(self.Copy, self.BestSol, self.BestObj
                            , self._IterBestSol, self._IterBestObj, self.Population, self.ObjectiveValues)
                self.Log.append(log)
                
                logstr = f"Iter: {self._iteration}|    {logstr}"

                if len(logstr) != length:
                    print("\b"*length*2, end="\r")
                length = len(logstr)
                print(logstr, end="\r")
                self._iteration += 1
        except:
            pass
        print()
        now = time.time()
        obj, vio = self.ExplicitObjFunc(self.BestSol)
        print()
        print(f"{type(self).__name__} RESULTS:".center(100,"-"))
        print(f"# of Iterations: {self.NumberOfIteration}")
        print(f"Best Solution: {self.BestSol}")
        print(f"Objective Value: {obj}")
        print(f"Constraint: {vio}")
        print()
        print(f"{type(self).__name__} finished at {datetime.datetime.fromtimestamp(now).strftime('%Y-%m-%d %H:%M:%S')}".center(100,"="))
        print()

    def DefaultRepair(self,sol):
        sol = np.maximum(sol,self.VariableBounds[:,0])
        sol = np.minimum(sol,self.VariableBounds[:,1])
        return sol

    def RouletteWheel(self, FitnessValues, N, start = 0, end = None):
        if end == None:
                end = len(FitnessValues)
        # pSum = np.sum(FitnessValues[start:end])
        # P = np.full_like(FitnessValues[start:end], 1/)
        # P = FitnessValues[start:end] * P
        P = np.cumsum(FitnessValues[start:end])
        Output = []
        n = np.random.rand(N) * P[-1]
        for i in range(N):
            for index, p in enumerate(P):
                if p > n[i]:
                    Output.append(index+start)
                    break
        assert len(Output) == N
        return Output
    
    def SetIteration(self, i):
        self._iteration = i

    @property
    def PopulationSize(self):
        return None if len(self.Population)== 0 else len(self.Population)
    @property
    def D(self):
        return len(self.Population[0]) if self.Population is not None else None

class ALO(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.AntPopulation = []
        self.AntObjectiveValues = []
        self.RandomWalk = kwargs.get("RandomWalk", self.DefaultRandomWalk)
        self.AntRatio = 1

    def DefaultInitialization(self):
        super().DefaultInitialization()
        self.AntPopulation = np.zeros((self.PopulationSize*self.AntRatio, self.D))
        self.AntObjectiveValues = np.zeros(self.PopulationSize*self.AntRatio)
        self.VariableBounds = np.array(self.VariableBounds)
        self.SortedAntlionFitness = 1/np.sort(self.ObjectiveValues)
        if not all(self.SortedAntlionFitness > 0):
            self.SortedAntlionFitness = self.SortedAntlionFitness - np.min(self.SortedAntlionFitness) + 100

    def DefaultRandomWalk(self, antlion):
        # scaled search radius
        I = 1
        if self._iteration>self.NumberOfIteration*(0.95):
            I=1+1000000*(self._iteration/self.NumberOfIteration)
        elif self._iteration>self.NumberOfIteration*(0.9):
            I=1+100000*(self._iteration/self.NumberOfIteration)
        elif self._iteration>self.NumberOfIteration*(3/4):
            I=1+10000*(self._iteration/self.NumberOfIteration)
        elif self._iteration>self.NumberOfIteration/2:
            I=1+1000*(self._iteration/self.NumberOfIteration)
        elif self._iteration>self.NumberOfIteration/10:
            I=1+100*(self._iteration/self.NumberOfIteration)
        bounds = self.VariableBounds/I
        
        # move the scaled search radius around the selected antlion (search range)
        if np.random.rand() < 0.5:
            bounds[:,0] = bounds[:,0] + antlion
        else:
            bounds[:,0] = -bounds[:,0] + antlion
        if np.random.rand() < 0.5:
            bounds[:,1] = bounds[:,1] + antlion
        else:
            bounds[:,1] = -bounds[:,1] + antlion
        
        ### mathematically equivalent
        sol = np.random.normal(size=(len(antlion,)))
        # 10 -> 4 std
        sol = sol * np.abs(bounds[:,1]-bounds[:,0])/(14*(1-self._iteration/self.NumberOfIteration)+4) + np.sum(bounds, axis=1)/2

        # ## original
        # sol = np.copy(bounds[:,0])
        # for i in range(len(sol)):
        #     X = np.cumsum(2*(np.random.rand(self.NumberOfIteration,1)>0.5)-1)
        #     a, b = X.min(), X.max()
        #     c, d = bounds[i,0], bounds[i,1]
        #     sol[i] = sol[i] + (X[self._iteration-1]-a)/(b-a)*(d-c)

        return sol

    def Iteration(self):
        # n = self.RouletteWheel(1/(self.ObjectiveValues-self.BestObj+np.finfo(float).eps), self.PopulationSize*self.AntRatio)
        n = self.RouletteWheel(self.SortedAntlionFitness, self.PopulationSize*self.AntRatio)
        
        for i in range(len(n)):
            RA = self.RandomWalk(self.Population[n[i]])
            RE = self.RandomWalk(self.BestSol)
            self.AntPopulation[i,:] = ((RA+RE)/2)

        if self.Repair is not None:
            self.AntPopulation = self.Repair(self.AntPopulation)

        for i in range(len(n)):
            self.AntObjectiveValues[i] = self.ObjectiveFunction(self.AntPopulation[i,:])
        
        self.Population = np.concatenate((self.Population, self.AntPopulation), axis = 0)
        self.ObjectiveValues = np.concatenate((self.ObjectiveValues, self.AntObjectiveValues), axis=0)
        indices = np.argsort(self.ObjectiveValues)
        self.Population = self.Population[indices[:self.PopulationSize//(self.AntRatio+1)],:]
        self.ObjectiveValues = self.ObjectiveValues[indices[:self.PopulationSize]]

        self._IterBestObj = self.ObjectiveValues[0]
        self._IterBestSol = self.Copy(self.Population[0])
        
        if self.ObjectiveValues[0] < self.BestObj:
            self.BestObj = self.ObjectiveValues[0]
            self.BestSol = self.Copy(self.Population[0])
        self.ObjectiveValues[0] = self.BestObj
        self.Population[0] = self.Copy(self.BestSol)

class ABC(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.Limits = kwargs.get("Limits", 50)
        self.Ages = []
        self.OnlookerRatio = kwargs.get("OnlookerRatio", 10)
    
    def DefaultInitialization(self):
        super().DefaultInitialization()
        self.Ages = np.zeros(self.PopulationSize)
        self.InitFitness = 1/np.sort(self.ObjectiveValues)

    def Iteration(self):
        n = np.random.randint(0, self.PopulationSize, self.PopulationSize)
        for i in range(self.PopulationSize):
            temp = self.Population[i] + (1-self.Ages[i]/self.Limits)*(2*np.random.rand(len(self.Population[i]))-1) * (self.Population[i]-self.Population[n[i]])
            if self.Repair != None:
                temp = self.Repair(temp)
            tempObj = self.ObjectiveFunction(temp)
            if tempObj < self.ObjectiveValues[i]:
                self.ObjectiveValues[i] = tempObj
                self.Population[i] = self.Copy(temp)
                self.Ages[i] = 0
                if self.ObjectiveValues[i] < self.BestObj:
                    self.BestObj = self.ObjectiveValues[i]
                    self.BestSol = self.Copy(self.Population[i])
                if self.ObjectiveValues[i] < self._IterBestObj:
                    self._IterBestObj = self.ObjectiveValues[i]
                    self._IterBestSol = self.Copy(self.Population[i])
            else:
                self.Ages[i] += 1
            if self.ObjectiveValues[i] == self.BestObj:
                self.Ages[i] = 0

        # foodSource = self.RouletteWheel(1/(self.ObjectiveValues-np.min(self.ObjectiveValues)+np.finfo(float).eps), self.PopulationSize)
        foodSource = self.RouletteWheel(self.InitFitness, self.PopulationSize)
        # n = np.random.randint(0, self.PopulationSize, self.PopulationSize*self.OnlookerRatio)
        n = self.RouletteWheel(self.InitFitness, self.PopulationSize*self.OnlookerRatio)

        for i in foodSource:
            for j in range(self.OnlookerRatio):
                temp = self.Population[i] + (2*np.random.rand(len(self.Population[i]))-1) * (self.Population[i]-self.Population[n[i*self.OnlookerRatio+j]])
                # if i < self.PopulationSize-1:
                #     temp = self.Population[i] + (2*np.random.rand(len(self.Population[i]))-1) * (self.Population[i]-self.Population[i+1])
                # else:
                #     temp = self.Population[i] + (2*np.random.rand(len(self.Population[i]))-1) * (self.Population[i]-self.Population[i-1])
                if self.Repair != None:
                    temp = self.Repair(temp)
                tempObj = self.ObjectiveFunction(temp)
                self.Ages[i] += 1
                if tempObj < self.ObjectiveValues[i]:
                    self.ObjectiveValues[i] = tempObj
                    self.Population[i] = self.Copy(temp)
                    self.Ages[i] = 0
                else:
                    if self.Ages[i] > self.Limits:
                        self.Population[i] = DefaultUniformInitialization(1,self.VariableBounds)
                        self.ObjectiveValues[i] = self.ObjectiveFunction(self.Population[i])
                        self.Ages[i] = 0
                
                if self.ObjectiveValues[i] < self.BestObj:
                    self.BestObj = self.ObjectiveValues[i]
                    self.BestSol = self.Copy(self.Population[i])
                if self.ObjectiveValues[i] < self._IterBestObj:
                    self._IterBestObj = self.ObjectiveValues[i]
                    self._IterBestSol = self.Copy(self.Population[i])
        indices = np.argsort(self.ObjectiveValues)
        self.Population = self.Population[indices]
        self.ObjectiveValues = self.ObjectiveValues[indices]

class BA(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.freqMin = kwargs.get("FreqMin", 0)
        self.freqMax = kwargs.get("FreqMax", 2)
        self.Velocity = []
        self.Loudness = 1
        self.PulseRate = 2
        self.Alpha = 0.97
        self.Gamma = 0.1
        self.r0 = 0.5
    
    def DefaultInitialization(self):
        super().DefaultInitialization()
        self.Velocity = np.zeros_like(self.Population)
        self.Loudness = 1
        self.PulseRate = 2
    
    def Iteration(self):
        self.PulseRate = self.r0*(1-np.exp(-self.Gamma*self._iteration))
        # self.PulseRate = 0.5
        self.Loudness = self.Alpha*self.Loudness
        # self.Loudness = (1-self._iteration/self.NumberOfIteration)*self.Loudness
        for i in range(self.PopulationSize):
            tempSol = []
            if np.random.rand() > self.PulseRate:
                self.Velocity[i] = self.Velocity[i] + (self.BestSol-self.Population[i]) * (self.freqMin + (self.freqMax-self.freqMin)*np.random.rand())
                tempSol = self.Population[i] + self.Velocity[i]
            else:
                tempSol = self.BestSol + 0.1 * np.random.normal(size=(len(self.Population[i],)))*self.Loudness

            if self.Repair != None:
                tempSol = self.Repair(tempSol)

            tempObj = self.ObjectiveFunction(tempSol)
            
            if (tempObj < self.ObjectiveValues[i]) and (np.random.rand() > self.Loudness):
                self.Population[i] = self.Copy(tempSol)
                self.ObjectiveValues[i] = tempObj

            if tempObj < self.BestObj:
                self.BestObj = tempObj
                self.BestSol = self.Copy(tempSol)

            if self.ObjectiveValues[i] < self._IterBestObj:
                self._IterBestObj = self.ObjectiveValues[i]
                self._IterBestSol = self.Copy(self.Population[i]) 

from scipy.stats import levy
class CS(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.Pa = kwargs.get("Pa", 0.3)
        self.Alpha = kwargs.get("Alpha", 0.1)

    def DefaultInitialization(self):
        super().DefaultInitialization()

    def Iteration(self):
        n = np.random.randint(0,self.PopulationSize, self.PopulationSize)
        for i in range(self.PopulationSize):
            tempSol = self.Population[i] + (2*np.random.random()-1)*(1-self._iteration/self.NumberOfIteration)*self.Alpha * levy.rvs(scale = 0.5, size=len(self.Population[i]))
            if self.Repair != None:
                tempSol = self.Repair(tempSol)
            tempObj = self.ObjectiveFunction(tempSol)
            if tempObj < self.ObjectiveValues[n[i]]:
                self.ObjectiveValues[n[i]] = tempObj
                self.Population[n[i]] = self.Copy(tempSol)
            if tempObj < self.BestObj:
                self.BestObj = tempObj
                self.BestSol = self.Copy(tempSol)
            if tempObj < self._IterBestObj:
                self._IterBestObj = tempObj
                self._IterBestSol = self.Copy(tempSol)

        indices = np.argsort(self.ObjectiveValues)[-1::-1]
        randInit = DefaultUniformInitialization(int(self.PopulationSize*self.Pa), self.VariableBounds)
        if self.Repair is not None:
            randInit = self.Repair(randInit)
        for i, init in enumerate(randInit):
            self.Population[indices[i]] = self.Copy(init)
            self.ObjectiveValues[indices[i]] = self.ObjectiveFunction(init)
class DE(GenericMA):
    def __init__(self, variant = "DE/rand/1/exp", **kwargs) -> None:
        super().__init__(**kwargs)
        '''
        _variant : list
            The variant representation should follow the below convention. 
            (Please refer to https://arxiv.org/ftp/arxiv/papers/1105/1105.1901.pdf for more detailed descriptions.)

                /*---------------------------------------------------------------------------------
                DE/(mutation method)/(# of solution agents involved in mutation)/(crossover method)
                ---------------------------------------------------------------------------------*/
            
            (mutation method):  -rand: assign the randomly selected agent as the base agent.
                                -best: assign the best solution so far as the base agent.
                                -current-to-best: assign the current agent as the base agent and the best solution so far as the secondary agent.
                                -current-to-rand: assign the current agent as the base agent and a ramdomly selected agent as the secondary agent.
                                -rand-to-best: assign a ramdoly selected agent as the base agent and the best solution so far as the secondary agent.
            
            (crossover method): -bin: binomial crossover
                                -exp: exponential crossover
        '''
        self._variant = [x.lower() for x in variant.split("/")]
        assert len(self._variant) == 4
        self._variant[2] = int(self._variant[2])

        self.Cr = kwargs.get("Cr", 0.7) # Crossover rate
        self.F = kwargs.get("F", 1)     # Scaling factor for differences between ramdoly selected agents.
        self.K = kwargs.get("K", 1)     # Scaling factor for the difference between the best solution and another given agent.
        
        # Customized Methods
        self.Mutation = kwargs.get("Mutation", self.DefaultMutation)
        self.CrossOver = kwargs.get("Cossover", self.DefaultCrossover)
        
        self._defaultMuatationMethods = {"rand":self.DefaultRandMutation,
                                        "best":self.DefaultBestMutation,
                                        "current-to-best":self.DefaultCurrent2BestMutation,
                                        "current-to-rand":self.DefaultCurrent2RandMutation}
        self._defaultCrossoverMethods = {"bin":self.DefaultBinCrossover,
                                        "exp":self.DefaultExpCrossover}

    def DefaultMutation(self, index = None, **kwargs):
        assert len(self._variant) == 4
        return self._defaultMuatationMethods[self._variant[1]](index)
        
    def DefaultRandMutation(self, index = None):
        tempSol = []
        n = np.random.choice(self.PopulationSize, 1+2*self._variant[2], replace=False)
        for i in range(self._variant[2]):
            if tempSol == []:
                tempSol = np.subtract(self.Population[n[1+i*2]],self.Population[n[2+i*2]])
            else:
                tempSol = np.add(tempSol, np.subtract(self.Population[n[1+i*2]],self.Population[n[2+i*2]]))
        return np.add(self.Population[n[0]], self.F * tempSol)
    
    def DefaultBestMutation(self, index = None):
        tempSol = []
        n = np.random.choice(self.PopulationSize, 2*self._variant[2], replace=False)
        for i in range(self._variant[2]):
            if tempSol == []:
                tempSol = np.subtract(self.Population[n[i*2]],self.Population[n[1+i*2]])
            else:
                tempSol = np.add(tempSol, np.subtract(self.Population[n[i*2]],self.Population[n[1+i*2]]))
        return np.add(self.BestSol, self.F * tempSol)
    
    def DefaultCurrent2BestMutation(self, index = None):
        tempSol = []
        n = np.random.choice(self.PopulationSize, 2*self._variant[2], replace=False)
        for i in range(self._variant[2]):
            if tempSol == []:
                tempSol = np.subtract(self.Population[n[i*2]],self.Population[n[1+i*2]])
            else:
                tempSol = np.add(tempSol, np.subtract(self.Population[n[i*2]],self.Population[n[1+i*2]]))
        return np.add(self.Population[index], np.add(self.F * tempSol, 
                                                     self.K * np.subtract(self.BestSol, self.Population[index])))
    
    def DefaultCurrent2RandMutation(self, index = None):
        tempSol = []
        n = np.random.choice(self.PopulationSize, 2*self._variant[2]+1, replace=False)
        for i in range(self._variant[2]):
            if tempSol == []:
                tempSol = np.subtract(self.Population[n[1+i*2]],self.Population[n[2+i*2]])
            else:
                tempSol = np.add(tempSol, np.subtract(self.Population[n[1+i*2]],self.Population[n[2+i*2]]))
        return np.add(self.Population[index], np.add(self.F * tempSol, 
                                                     self.K * np.subtract(self.Population[n[0]], self.Population[index])))
    
    def DefaultRand2BestMutation(self, index = None):
        tempSol = []
        n = np.random.choice(self.PopulationSize, 2*self._variant[2]+1, replace=False)
        for i in range(self._variant[2]):
            if tempSol == []:
                tempSol = np.subtract(self.Population[n[1+i*2]],self.Population[n[2+i*2]])
            else:
                tempSol = np.add(tempSol, np.subtract(self.Population[n[1+i*2]],self.Population[n[2+i*2]]))
        return np.add(self.Population[index], np.add(self.F * tempSol, 
                                                     self.K * np.subtract(self.Population[n[0]], self.Population[index])))
    
    def DefaultCrossover(self, p1, p2):
        return self.DefaultExpCrossover(p1,p2)

    def DefaultBinCrossover(self, p1, p2):
        d = np.random.randint(0,len(p2))
        sol = self.Copy(p2)
        for i in range(len(p2)):
            if np.random.random() <= self.Cr or i == d:
                sol[i] = p1[i]
        return sol
    
    def DefaultExpCrossover(self, p1, p2):
        d = np.random.randint(0,len(p2))
        sol = self.Copy(p2)
        for i in range(len(p2)):
            k = (d + i)%len(p2)
            if np.random.random() <= self.Cr:
                sol[k] = p1[k]
            else:
                break
        return sol
    
    def DefaultInitialization(self):
        self.ObjectiveValues = list(map(self.ObjectiveFunction,self.Population))
        index = np.argmin(self.ObjectiveValues)
        self.BestObj = self.ObjectiveValues[index]
        self.BestSol = self.Copy(self.Population[int(index)])
    
    def Iteration(self):
        self._IterBestSol = None
        self._IterBestObj = np.inf

        for i in range(self.PopulationSize):
            mutant = self.Mutation(i, **self.__dict__)
            if self.Repair != None:
                mutant = self.Repair(mutant)
            newSol = self.CrossOver(mutant, self.Population[i])
            if self.Repair != None:
                newSol = self.Repair(newSol)
            newObj = self.ObjectiveFunction(newSol)
            if newObj < self.ObjectiveValues[i]:
                self.Population[i] = newSol
                self.ObjectiveValues[i] = newObj
                if newObj < self._IterBestObj:
                    self._IterBestObj = newObj
                    self._IterBestSol = self.Copy(newSol)
                if newObj < self.BestObj:
                    self.BestObj = newObj
                    self.BestSol = self.Copy(newSol)

class FA(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.Absorption = kwargs.get("Absorption", 0.002)
        self.Alpha = kwargs.get("Alpha", 0.01)
        self.B0 = kwargs.get("B0", 1)
    
    def DefaultInitialization(self):
        super().DefaultInitialization()

    def Iteration(self):
        for i in range(self.PopulationSize):
            for j in range(self.PopulationSize):
                if i == j:
                    continue
                if self.ObjectiveValues[j] < self.ObjectiveValues[i]:
                    r = self.Population[j]-self.Population[i]
                
                    self.Population[i] = self.Population[i] + self.B0 * np.exp(-self.Absorption * np.abs(sum(r))) * r + self._iteration/self.NumberOfIteration * (np.random.rand(len(self.Population[i]))-0.5)
                    if self.Repair != None:
                        self.Population[i] = self.Repair(self.Population[i])
                    self.ObjectiveValues[i] = self.ObjectiveFunction(self.Population[i])
                    if self.ObjectiveValues[i] < self.BestObj:
                        self.BestObj = self.ObjectiveValues[i]
                        self.BestSol = self.Copy(self.Population[i])
                    if self.ObjectiveValues[i] < self._IterBestObj:
                        self._IterBestObj = self.ObjectiveValues[i]
                        self._IterBestSol = self.Copy(self.Population[i])
class FPA(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.P = kwargs.get("P", 0.3)

    def Iteration(self):
        rnd = np.random.rand(self.PopulationSize)
        for i in range(self.PopulationSize):
            tempSol = []
            if rnd[i] < self.P:
                tempSol = self.Population[i] + levy.rvs(scale=0.01)* (self.BestSol-self.Population[i])
            else:
                n = np.random.choice(self.PopulationSize, 2, replace=False)
                tempSol = self.Population[i] + np.random.rand()*(self.Population[n[0]]-self.Population[n[1]])
            if self.Repair != None:
                tempSol = self.Repair(tempSol)
            tempObj = self.ObjectiveFunction(tempSol)
            if tempObj < self.ObjectiveValues[i]:
                self.ObjectiveValues[i] = tempObj
                self.Population[i] = self.Copy(tempSol)
            if tempObj < self.BestObj:
                self.BestObj = tempObj
                self.BestSol = self.Copy(tempSol)
            if tempObj < self._IterBestObj:
                self._IterBestObj = tempObj
                self._IterBestSol = self.Copy(tempSol)

class GSA(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.G0 = kwargs.get("G0", 100)
        self.Alpha = kwargs.get("Alpha", 20)
    def DefaultInitialization(self):
        super().DefaultInitialization()
        self.Kbest = self.PopulationSize
        self.Velocity = np.zeros_like(self.Population)

    def Iteration(self):
        a = np.zeros_like(self.Population)
        G = self.G0 * np.exp(-self.Alpha * self._iteration/self.NumberOfIteration)
        worst = np.max(self.ObjectiveValues)
        M = (self.ObjectiveValues - worst)/(np.sum(self.ObjectiveValues)-self.PopulationSize*worst)
        self.Kbest = int(self.PopulationSize*(1-0.98*self._iteration/self.NumberOfIteration))
        indices = np.argsort(self.ObjectiveValues)
        
        for i in range(self.PopulationSize):
            for j in range(self.Kbest):
                if i == indices[j]:
                    continue
                r = self.Population[indices[j]]-self.Population[i]
                a[i] = a[i] + G*M[indices[j]]/(np.abs(np.sum(r))+1E-5)*r
        self.Velocity = np.random.rand(self.PopulationSize,1)*self.Velocity+a
        self.Population = self.Population + self.Velocity
        if self.Repair != None:
            self.Population = self.Repair(self.Population)
        for i in range(self.PopulationSize):
            self.ObjectiveValues[i] = self.ObjectiveFunction(self.Population[i])
            if self.ObjectiveValues[i] < self.BestObj:
                self.BestObj = self.ObjectiveValues[i]
                self.BestSol = self.Copy(self.Population[i])
            if self.ObjectiveValues[i] < self._IterBestObj:
                self._IterBestObj = self.ObjectiveValues[i]
                self._IterBestSol = self.Copy(self.Population[i])

class GWO(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def DefaultInitialization(self):
        self.ObjectiveValues = np.array(list(map(self.ObjectiveFunction,self.Population)))
        indices = np.argsort(self.ObjectiveValues)
        self.BestObj = self.ObjectiveValues[indices[0]]
        self.BestSol = self.Copy(self.Population[indices[0]])
        self.k = 3
        self.TopWolves = np.copy(self.Population[indices[:self.k]])
        self.TopObjs = self.ObjectiveValues[indices[:self.k]]

    def Iteration(self):
        a = 2 - 2 * self._iteration/self.NumberOfIteration
        r = np.random.rand(2*self.k*self.PopulationSize, self.D)
        A = 2 * a * r[:self.k*self.PopulationSize] - a
        C = 2 * r[self.k*self.PopulationSize:]
        # self.Population = (np.tile(self.TopWolves, (self.PopulationSize,1)) - A * np.abs(C*np.tile(self.TopWolves,(self.PopulationSize,1)) - np.repeat(self.Population, self.k, axis = 0))).reshape(self.PopulationSize,-1,self.D).sum(axis=1)/self.k
        self.Population = (np.tile(self.TopWolves, (self.PopulationSize,1)) - A * np.abs(C* np.tile(self.TopWolves,(self.PopulationSize,1)) - np.repeat(self.Population, self.k, axis = 0))).reshape(self.PopulationSize,-1,self.D).sum(axis=1)/self.k
        
        if self.Repair != None:
            self.Population = self.Repair(self.Population)
        for i in range(self.PopulationSize):
            self.ObjectiveValues[i] = self.ObjectiveFunction(self.Population[i])
            
            if self.ObjectiveValues[i] < self.BestObj:
                self.BestObj = self.ObjectiveValues[i]
                self.BestSol = self.Copy(self.Population[i])
            
            if self.ObjectiveValues[i] < self._IterBestObj:
                self._IterBestObj = self.ObjectiveValues[i]
                self._IterBestSol = self.Copy(self.Population[i])
            
            for k in range(len(self.TopObjs)):
                if self.ObjectiveValues[i] < self.TopObjs[k]:
                    self.TopObjs[k] = self.ObjectiveValues[i]
                    self.TopWolves[k] = self.Copy(self.Population[i])
                    break
            
class HS(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.PitchAdjustRate = kwargs.get("PitchAdjustRate", 0.25)
        self.AcceptRate = kwargs.get("AcceptRate", 0.7)
        self.AdjustBandwidth = kwargs.get("AdjustBandwidth", 0.1)
    
    def DefaultInitialization(self):
        self.ObjectiveValues = np.array(list(map(self.ObjectiveFunction,self.Population)))
        indices = np.argsort(self.ObjectiveValues)
        self.Population = self.Population[indices]
        self.ObjectiveValues = self.ObjectiveValues[indices]
        self.BestObj = self.ObjectiveValues[0]
        self.BestSol = self.Copy(self.Population[0])
    
    def Iteration(self):
        temp = []
        tempObjs = []
        self._IterBestObj = self.ObjectiveValues[0]
        self._IterBestSol = self.Copy(self.Population[0])
        for i in range(self.PopulationSize):
            tempSol = np.zeros_like(self.Population[i])
            for j in range(self.D):
                if np.random.rand() > self.AcceptRate:
                    tempSol[j] = self.Population[np.random.randint(0,self.PopulationSize),j]
                elif np.random.rand() > self.AcceptRate:
                    tempSol[j] = self.Population[np.random.randint(0,self.PopulationSize),j] \
                            + self.AdjustBandwidth*(2*np.random.rand()-1)
                else:
                    tempSol[j] = self.VariableBounds[j,0] + np.random.rand()*(self.VariableBounds[j,1]-self.VariableBounds[j,0])
            if self.Repair is not None:
                tempSol = self.Repair(tempSol)
            tempObj = self.ObjectiveFunction(tempSol)
            if tempObj < self.ObjectiveValues[-1]:
                temp.append(self.Copy(tempSol))
                tempObjs.append(tempObj)
            
                if tempObj < self.BestObj:
                    self.BestObj = tempObj
                    self.BestSol = self.Copy(tempSol)
                    
                if tempObj < self._IterBestObj:
                    self._IterBestObj = tempObj
                    self._IterBestSol = self.Copy(tempSol)
        if len(tempObjs) > 0:
            self.ObjectiveValues = np.concatenate((self.ObjectiveValues, tempObjs))
            indices = np.argsort(self.ObjectiveValues)[:self.PopulationSize]
            self.Population = np.vstack((self.Population,temp))[indices]
            self.ObjectiveValues = self.ObjectiveValues[indices]
            
class SFLA(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.NumberOfMemeplex = kwargs.get("NumberOfMemeplex", 10)
        self.ShuffleIteration = 300
    
    def DefaultInitialization(self):
        self.ObjectiveValues = np.array(list(map(self.ObjectiveFunction,self.Population)))
        indices = np.argsort(self.ObjectiveValues)
        self._k = self.PopulationSize//self.NumberOfMemeplex
        self._MemeIndices = [m*self._k+i for i in range(self._k) for m in range(self.NumberOfMemeplex)]
        self.Population = self.Population[indices[self._MemeIndices]]
        self.ObjectiveValues = self.ObjectiveValues[indices[self._MemeIndices]]

        self.BestObj = self.ObjectiveValues[0]
        self.BestSol = self.Copy(self.Population[0])

    def Iteration(self):
        if self._iteration > 0 and self._iteration % self.ShuffleIteration == 0:
            indices = np.argsort(self.ObjectiveValues)
            self.Population = self.Population[indices[self._MemeIndices]]
            self.ObjectiveValues = self.ObjectiveValues[indices[self._MemeIndices]]
        worst = [(m+1)*self._k-1 for m in range(self.NumberOfMemeplex)]
        best = [m*self._k for m in range(self.NumberOfMemeplex)]
        temp = self.Population[worst] + np.random.rand(self.NumberOfMemeplex, self.D)*(self.Population[best]-self.Population[worst])
        for m in range(self.NumberOfMemeplex):
            if self.Repair is not None:
                temp[m] = self.Repair(temp[m])
            tempObj = self.ObjectiveFunction(temp[m])
            if tempObj > self.ObjectiveValues[(m+1)*self._k-1]:
                temp[m] = self.Population[(m+1)*self._k-1] + np.random.rand(self.D)*(self.BestSol-self.Population[(m+1)*self._k-1])
                if self.Repair is not None:
                    temp[m] = self.Repair(temp[m])
                tempObj = self.ObjectiveFunction(temp[m])
            
            if tempObj > self.ObjectiveValues[(m+1)*self._k-1]:
                temp[m] = DefaultUniformInitialization(1,self.VariableBounds)
                if self.Repair is not None:
                    temp[m] = self.Repair(temp[m])
                tempObj = self.ObjectiveFunction(temp[m])

            self.ObjectiveValues[(m+1)*self._k-1] = tempObj
            self.Population[(m+1)*self._k-1] = self.Copy(temp[m])
            
            if self.ObjectiveValues[(m+1)*self._k-1] < self.BestObj:
                self.BestObj = self.ObjectiveValues[(m+1)*self._k-1]
                self.BestSol = self.Copy(temp[m])
                
            if self.ObjectiveValues[(m+1)*self._k-1] < self._IterBestObj:
                self._IterBestObj = tempObj
                self._IterBestSol = self.Copy(temp[m])

            indices = np.argsort(self.ObjectiveValues[m*self._k:(m+1)*self._k])
            self.Population[m*self._k:(m+1)*self._k] = self.Population[m*self._k+indices]
            self.ObjectiveValues[m*self._k:(m+1)*self._k] = self.ObjectiveValues[m*self._k+indices]

class SCA(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def Iteration(self):
        r1 = 2 - 2 * self._iteration/self.NumberOfIteration
        r2 = 2*np.pi*np.random.rand(self.PopulationSize,self.D)
        r3 = 2*np.random.rand(self.PopulationSize,self.D)
        r4 = np.random.rand(self.PopulationSize,self.D)
        
        sin = self.Population + r1 * np.sin(r2) * np.abs(r3*self.BestSol-self.Population)
        cos = self.Population + r1 * np.cos(r2) * np.abs(r3*self.BestSol-self.Population)
        r = r4 > 0.5
        self.Population = r*sin+(~r)*cos

        for i in range(self.PopulationSize):
            if self.Repair is not None:
                self.Population[i] = self.Repair(self.Population[i])
            self.ObjectiveValues[i] = self.ObjectiveFunction(self.Population[i])

            if self.ObjectiveValues[i] < self.BestObj:
                self.BestObj = self.ObjectiveValues[i]
                self.BestSol = self.Copy(self.Population[i])
            
            if self.ObjectiveValues[i] < self._IterBestObj:
                self._IterBestObj = self.ObjectiveValues[i]
                self._IterBestSol = self.Copy(self.Population[i])
            
class TLBO(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
    
    def DefaultInitialization(self):
        super().DefaultInitialization()
        self.Teacher = self.Copy(self.BestSol)
        self.TeacherObj = self.BestObj

    def Iteration(self):
        Mean = np.mean(self.Population, axis=0)[np.newaxis,:]
        TF = (np.random.rand(self.PopulationSize)>0.5)[:,np.newaxis]+1
        temp = self.Population + np.random.rand(self.PopulationSize,self.D)*(self.Teacher-TF*Mean)
        for i in range(self.PopulationSize):
            if self.Repair is not None:
                temp[i] = self.Repair(temp[i])
            tempObj = self.ObjectiveFunction(temp[i])
            if tempObj < self.ObjectiveValues[i]:
                self.ObjectiveValues[i] = tempObj
                self.Population[i] = self.Copy(temp[i])
                if tempObj < self.TeacherObj:
                    self.TeacherObj = tempObj
                    self.Teacher = self.Copy(temp[i])
                if tempObj < self.BestObj:
                    self.BestObj = tempObj
                    self.BestSol = self.Copy(temp[i])
                if tempObj < self._IterBestObj:
                    self._IterBestObj = tempObj
                    self._IterBestSol = self.Copy(temp[i])
        
        for i in range(self.PopulationSize):
            n = i
            while n == i:
                n = np.random.randint(0,self.PopulationSize)
            step = self.Population[i] - self.Population[n]
            if self.ObjectiveValues[n] < self.ObjectiveValues[i]:
                step = -step
            
            temp = self.Population[i] + np.random.rand(self.D)*step
            if self.Repair is not None:
                temp = self.Repair(temp)
            tempObj = self.ObjectiveFunction(temp)
            if tempObj < self.ObjectiveValues[i]:
                self.ObjectiveValues[i] = tempObj
                self.Population[i] = self.Copy(temp)
                if tempObj < self.TeacherObj:
                    self.TeacherObj = tempObj
                    self.Teacher = self.Copy(temp)
                if tempObj < self.BestObj:
                    self.BestObj = tempObj
                    self.BestSol = self.Copy(temp)
                if tempObj < self._IterBestObj:
                    self._IterBestObj = tempObj
                    self._IterBestSol = self.Copy(temp)

class WOA(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def DefaultInitialization(self):
        super().DefaultInitialization()
        self.Population = self.Population - self.BestSol

    def Iteration(self):
        a = 2 - 2 * self._iteration/self.NumberOfIteration
        a2 = -1 - self._iteration/self.NumberOfIteration
        b = 1
        
        for i in range(self.PopulationSize):
            temp = self.Copy(self.Population[i])
            A = 2 * a * np.random.rand() - a
            C = 2 * np.random.rand()
            P = np.random.rand()
            l=(a2-1)*np.random.rand()+1
            for j in range(len(temp)):
                if P < 0.5:
                    if np.abs(A) >= 1:
                        X_rand = self.Population[np.random.randint(0, self.PopulationSize), j]
                        self.Population[i,j]  = X_rand - A * np.abs(C*X_rand-self.Population[i,j])
                    else:
                        self.Population[i,j] = self.BestSol[j] - A * np.abs(C*self.BestSol[j]-self.Population[i,j])
                else:
                    dist = np.abs(self.BestSol[j]-self.Population[i,j])
                    self.Population[i,j] = dist * np.exp(b*l)*np.cos(l*2*np.pi) + self.BestSol[j]

            
            if self.Repair is not None:
                self.Population[i] = self.Repair(self.Population[i])
            
            self.ObjectiveValues[i] = self.ObjectiveFunction(self.Population[i])
            if self.ObjectiveValues[i] < self.BestObj:
                self.BestObj = self.ObjectiveValues[i]
                self.BestSol = self.Copy(self.Population[i])
            
            if self.ObjectiveValues[i] < self._IterBestObj:
                self._IterBestObj = self.ObjectiveValues[i]
                self._IterBestSol = self.Copy(self.Population[i])

class SA(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.CoolingFactor = kwargs.get("CoolingFactor", None)
        self.T0 = kwargs.get("T0", 100)
        self.Tf = kwargs.get("Tf", 10)
        self.Scale = kwargs.get("Scale", 20)
    def DefaultInitialization(self):
        super().DefaultInitialization()
        self.T = self.T0
        self.Diff = (self.VariableBounds[:,1] - self.VariableBounds[:,0])/self.Scale
        # self.Mid = (self.VariableBounds[:,1] + self.VariableBounds[:,0])/2

    def Iteration(self):
        temp = self.Population +  self.Diff * np.random.randn(self.PopulationSize,self.D)
        if self.Repair is not None:
            tempObj = np.array(list(map(lambda x:self.ObjectiveFunction(self.Repair(x)),temp)))
        else:
            tempObj = np.array(list(map(self.ObjectiveFunction,temp)))
        for i in range(self.PopulationSize):
            if tempObj[i] < self.ObjectiveValues[i]:
                self.ObjectiveValues[i] = tempObj[i]
                self.Population[i] = self.Copy(temp[i])
            else:
                delta = tempObj[i] - self.ObjectiveValues[i]
                if np.exp(-delta/self.T) > np.random.rand():
                    self.ObjectiveValues[i] = tempObj[i]
                    self.Population[i] = self.Copy(temp[i])

            if self.ObjectiveValues[i] < self.BestObj:
                self.BestObj = self.ObjectiveValues[i]
                self.BestSol = self.Copy(self.Population[i])
            
            if self.ObjectiveValues[i] < self._IterBestObj:
                self._IterBestObj = self.ObjectiveValues[i]
                self._IterBestSol = self.Copy(self.Population[i])

        if self.CoolingFactor is not None:
            self.T = (self.T-self.Tf)*self.CoolingFactor + self.Tf
        else:
            self.T = (self.T-self.Tf) * (1-self._iteration/self.NumberOfIteration) + self.Tf

class PSO(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.Cognitive = kwargs.get("Cognitive", 2)
        self.Social = kwargs.get("Social", 2)
        self.Inertia = kwargs.get("Inertia", 0.9)

    def DefaultInitialization(self):
        super().DefaultInitialization()
        self.ParticleBest = self.Copy(self.Population)
        self.ParticleBestObj = self.Copy(self.ObjectiveValues)
        self.V = np.zeros((self.PopulationSize,self.D))

    def Iteration(self):
        self.V = self.Inertia * self.V + self.Social * np.random.rand(self.PopulationSize,1) * (self.BestSol-self.Population) \
                + self.Cognitive * np.random.rand(self.PopulationSize,1) * (self.ParticleBest-self.Population)
        self.Population = self.Population + self.V
        for i in range(self.PopulationSize):
            if self.Repair is not None:
                self.Population[i] = self.Repair(self.Population[i])
            self.ObjectiveValues[i] = self.ObjectiveFunction(self.Population[i])

            if self.ObjectiveValues[i] < self.ParticleBestObj[i]:
                self.ParticleBestObj[i] = self.ObjectiveValues[i]
                self.ParticleBest = self.Copy(self.Population[i])
            
            if self.ObjectiveValues[i] < self.BestObj:
                self.BestObj = self.ObjectiveValues[i]
                self.BestSol = self.Copy(self.Population[i])
            
            if self.ObjectiveValues[i] < self._IterBestObj:
                self._IterBestObj = self.ObjectiveValues[i]
                self._IterBestSol = self.Copy(self.Population[i])

class GA(GenericMA):
    def __init__(self, Crossover = "2pt", Selection = "RouletteWheel", **kwargs) -> None:
        super().__init__(**kwargs)
        self.CR = kwargs.get("CR", 0.7)
        self.MR = kwargs.get("MR", 0.1)
        
        self.Crossover = kwargs.get("Crossover", self.DefaultCrossoverMethods(Crossover))
        self.Selection = kwargs.get("Selection", self.DefaultSelectionMethods(Selection))
        self.Mutation = kwargs.get("Mutation", self.DefaultMutation)

    def DefaultSelectionMethods(self, method):
        assert type(method) is str
        method = method.lower()
        if method == "roulettewheel":
            return self.RouletteWheelSelection
        elif method == "linearrank":
            return self.LinearRankSelection
        elif method.endswith("tournament"):
            return self.NTournamentSelection( int(method[0:-len("tournament")]))
        
    def RouletteWheelSelection(self, parentObjValues
                                   , offspringObjValues
                                   , mutantObjValues):
        x = np.concatenate((parentObjValues, offspringObjValues[:self.N_cr], mutantObjValues[:self.N_mu]))
        allFitnessValues = np.max(x) - x
        # allFitnessValues = 1/(1+np.exp(x - np.max(x)))
        indices = self.RouletteWheel(allFitnessValues, len(parentObjValues))
        
        parents = []
        offsprings = []
        mutants = []
        for i in indices:
            if i >= len(parentObjValues) + self.N_cr:
                mutants.append(i - len(parentObjValues) - self.N_cr)
            elif i >= len(parentObjValues):
                offsprings.append(i - len(parentObjValues))
            else:
                parents.append(i)
        return parents, offsprings, mutants


    def LinearRankSelection(self, parentObjValues
                                , offspringObjValues
                                , mutantObjValues):
        indices = np.argsort(np.concatenate((parentObjValues, offspringObjValues[:self.N_cr], mutantObjValues[:self.N_mu])))
        parents = []
        offsprings = []
        mutants = []
        
        for i in indices[:self.PopulationSize]:
            if i >= len(parentObjValues) + self.N_cr:
                mutants.append(i - len(parentObjValues) - self.N_cr)
            elif i >= len(parentObjValues):
                offsprings.append(i - len(parentObjValues))
            else:
                parents.append(i)
        return parents, offsprings, mutants

    def NTournamentSelection(self, N):
        def TournamentSelection(parentObjValues
                              , offspringObjValues
                              , mutantObjValues):      
            randIndices = np.random.randint(0, len(parentObjValues)+self.N_cr+self.N_mu, size=(len(parentObjValues), N))
            parents = []
            offsprings = []
            mutants = []
            for i in range(len(parentObjValues)):
                minObj = float("inf")
                minId = -1
                for j in randIndices[i]:
                    tempObj = -1
                    if j >= len(parentObjValues) + self.N_cr:
                        tempObj = mutantObjValues[j - len(parentObjValues) - self.N_cr]
                    elif j >= len(parentObjValues):
                        tempObj = offspringObjValues[j - len(parentObjValues)]
                    else:
                        tempObj = parentObjValues[j]
                    if tempObj < minObj:
                        minObj = tempObj
                        minId = j
                if minId >= len(parentObjValues) + self.N_cr:
                    mutants.append(minId - len(parentObjValues) - self.N_cr)
                elif minId >= len(parentObjValues):
                    offsprings.append(minId - len(parentObjValues))
                else:
                    parents.append(minId)
            return parents, offsprings, mutants
        return TournamentSelection

    def DefaultCrossoverMethods(self, method):
        assert type(method) is str
        method = method.lower()

        if method.endswith("pt"):
            return self.NPtCrossover(int(method[:-2]))
        elif method == "segmented":
            return self.SegmentedCrossover
        elif method == "uniform":
            return self.UniformCrossover
        elif method == "shuffle":
            return self.ShuffleCrossover

    
    def NPtCrossover(self, N):
        def PtCrossover(population):
            def ptcr(x, cutpt):
                i, j = x
                assert len(population[i]) == len(population[j])
                P1 = self.Copy(population[i])
                P2 = self.Copy(population[j])
                for pt in range(len(cutpt)//2):
                    P1[cutpt[2*pt]:cutpt[2*pt+1]+1] = P2[cutpt[2*pt]:cutpt[2*pt+1]+1]
                if len(cutpt)%2 != 0:
                    P1[cutpt[-1]:] = P2[cutpt[-1]:]
                if self.Repair is not None:
                    P1 = self.Repair(P1)
                return P1

            indices = np.random.randint(0, len(population), size=(self.N_cr,2))
            cutpts = np.sort(np.random.randint(0, len(population[0]), size=(self.N_cr, N)), axis=1)
            return np.array(list(map(ptcr, indices, cutpts)))
        return PtCrossover

    def SegmentedCrossover(self):
        pass
    def UniformCrossover(self):
        pass
    def ShuffleCrossover(self):
        pass
    def DefaultMutation(self, population):
        # indices = np.random.randint(0, len(population), size = self.N_mu)
        mut = (np.random.rand(len(population), population.shape[1])<self.MR)
        muts = population*~mut+np.random.randn(len(population), population.shape[1])*mut
        # muts = population*~mut+DefaultUniformInitialization(len(population), self.VariableBounds)*mut
        if self.Repair is not None:
            muts = np.array(list(map(self.Repair, muts)))
        return muts

    def DefaultInitialization(self):
        super().DefaultInitialization()
        self.Offsprings = np.zeros_like(self.Population)
        self.mutants = np.zeros_like(self.Population)
        self.OffspringsObj = np.empty_like(self.ObjectiveValues)
        self.mutantsObj = np.empty_like(self.ObjectiveValues)
        self.N_cr = int(self.CR * self.PopulationSize)
        self.N_mu = int(self.MR * self.PopulationSize)

    def Iteration(self):
        self.Offsprings = self.Crossover(self.Population)
        self.OffspringsObj = np.array(list(map(self.ObjectiveFunction, self.Offsprings)))
        self.mutants = self.Mutation(self.Population)
        self.mutantsObj = np.array(list(map(self.ObjectiveFunction, self.mutants)))
        id = np.argmin(self.OffspringsObj)
        if self.OffspringsObj[id] < self._IterBestObj:
            self._IterBestObj = self.OffspringsObj[id]
            self._IterBestSol = self.Copy(self.Offsprings[id])
        id = np.argmin(self.mutantsObj)
        if self.mutantsObj[id] < self._IterBestObj:
            self._IterBestObj = self.mutantsObj[id]
            self._IterBestSol = self.Copy(self.mutants[id])
        if self._IterBestObj < self.BestObj:
            self.BestObj = self._IterBestObj
            self.BestSol = self.Copy(self._IterBestSol)
        p, off, mut = self.Selection(self.ObjectiveValues, self.OffspringsObj, self.mutantsObj)
        
        self.Population = np.concatenate((self.Population[p],self.Offsprings[off], self.mutants[mut]))
        self.ObjectiveValues = np.concatenate((self.ObjectiveValues[p], self.OffspringsObj[off], self.mutantsObj[mut]))

class MA(GA):
    def __init__(self, LocalSearchMaxIter = 10, Crossover="2pt", Selection="RouletteWheel", **kwargs) -> None:
        super().__init__(Crossover, Selection, **kwargs)
        self.LocalSearchMaxIter = LocalSearchMaxIter
    
    def LocalSearch(self, pop):
        objs = np.zeros(len(pop))
        for j in range(len(pop)):
            x = self.Copy(pop[j])
            xObj = self.ObjectiveFunction(x)
            dir = 0.0001 * np.random.randn(len(x))
            y = x + dir
            if self.Repair is not None:
                y = self.Repair(y)
            yObj = self.ObjectiveFunction(y)
            dir = dir * (yObj < xObj * 2 -1)
            step = 1
            
            for i in range(self.LocalSearchMaxIter):
                y = x + step * dir
                if self.Repair is not None:
                    y = self.Repair(y)
                yObj = self.ObjectiveFunction(y)
                if yObj < xObj:
                    x = self.Copy(y)
                    xObj = yObj
                    step *= 2
                else:
                    step *= 0.5
                
            pop[j] = x
            objs[j] = xObj
        return objs
            
    def Iteration(self):
        self.Offsprings = self.Crossover(self.Population)
        self.OffspringsObj = self.LocalSearch(self.Offsprings)

        self.mutants = self.Mutation(self.Population)
        self.mutantsObj = self.LocalSearch(self.mutants)
        id = np.argmin(self.OffspringsObj)
        if self.OffspringsObj[id] < self._IterBestObj:
            self._IterBestObj = self.OffspringsObj[id]
            self._IterBestSol = self.Copy(self.Offsprings[id])
        id = np.argmin(self.mutantsObj)
        if self.mutantsObj[id] < self._IterBestObj:
            self._IterBestObj = self.mutantsObj[id]
            self._IterBestSol = self.Copy(self.mutants[id])
        if self._IterBestObj < self.BestObj:
            self.BestObj = self._IterBestObj
            self.BestSol = self.Copy(self._IterBestSol)
        p, off, mut = self.Selection(self.ObjectiveValues, self.OffspringsObj, self.mutantsObj)
        
        self.Population = np.concatenate((self.Population[p],self.Offsprings[off], self.mutants[mut]))
        self.ObjectiveValues = np.concatenate((self.ObjectiveValues[p], self.OffspringsObj[off], self.mutantsObj[mut]))
        return super().Iteration()

class ICA(GenericMA):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.NumberOfEmpires = kwargs.get("NumberOfEmpires", 10)
    
    def DefaultInitialization(self):
        self.ObjectiveValues = np.array(list(map(self.ObjectiveFunction,self.Population)))
        indices = np.argsort(self.ObjectiveValues)
        self.BestObj = self.ObjectiveValues[indices[0]]
        self.BestSol = self.Copy(self.Population[indices[0]])
        self.Empires = self.Copy(self.Population[:self.NumberOfEmpires])
        self.EmpiresObj = self.Copy(self.ObjectiveValues[:self.NumberOfEmpires])
        self.Colonies = [[] for _ in range(self.NumberOfEmpires)]
        
        # indices = np.random.permutation(self.PopulationSize-self.NumberOfEmpires)
        indices = np.arange(self.PopulationSize-self.NumberOfEmpires)
        C = self.EmpiresObj - np.max(self.EmpiresObj)
        colonies = np.rint(np.cumsum(np.abs(C/np.sum(C)) * (self.PopulationSize-self.NumberOfEmpires))).astype('int32')
        
        s = 0
        for i in range(self.NumberOfEmpires):
            self.Colonies[i] = (indices[s:colonies[i]]+self.NumberOfEmpires).tolist()
            self.Colonies[i].append(i)
            s = colonies[i]
    
    def Iteration(self):
        pi_6 = np.pi / 6
        TC = np.zeros(len(self.Empires))
        for emp in range(len(self.Empires)):
            for col in self.Colonies[emp]:
                dev = 2 * np.random.rand(len(self.Empires[emp])) - 1
                if np.array_equal(self.Empires[emp], self.Population[col]):
                    self.Population[col] = self.Population[col] \
                                        + (1-self._iteration / self.NumberOfIteration) * dev
                else:
                    dir = 2 * np.random.rand() * (self.Empires[emp] - self.Population[col])
                    dev = dev - dev * dir / np.linalg.norm(dir)
                    dev /= np.linalg.norm(dev)
                    self.Population[col] = self.Population[col] \
                                        + dir \
                                        + np.linalg.norm(dir) * (np.random.rand() * pi_6) * dev
               
                if self.Repair is not None:
                    self.Population[col] = self.Repair(self.Population[col])
                self.ObjectiveValues[col] = self.ObjectiveFunction(self.Population[col])

                if self.ObjectiveValues[col] < self.BestObj:
                    self.BestSol = self.Copy(self.Population[col])
                    self.BestObj = self.ObjectiveValues[col]
                if self.ObjectiveValues[col] < self._IterBestObj:
                    self._IterBestSol = self.Copy(self.Population[col])
                    self._IterBestObj = self.ObjectiveValues[col]
                
                if self.ObjectiveValues[col] < self.EmpiresObj[emp]:
                    self.Empires[emp], self.Population[col] = self.Copy(self.Population[col]), self.Copy(self.Empires[emp])
                    self.EmpiresObj[emp], self.ObjectiveValues[col] = self.ObjectiveValues[col], self.EmpiresObj[emp]
            TC[emp] = self.EmpiresObj[emp] + np.mean(self.ObjectiveValues[self.Colonies[emp]])
        if len(TC) > 1:
            weakEmp = np.argmax(self.EmpiresObj)
            weakCols = self.Colonies[weakEmp]
            weakCol = weakCols[np.argmax(self.ObjectiveValues[weakCols])]
            TC = TC - np.max(TC)
            bestEmp = np.argmax(np.abs(TC/np.sum(TC)) - np.random.rand(len(TC)))
            self.Colonies[bestEmp].append(weakCol)
            self.Colonies[weakEmp].remove(weakCol)
            
            if len(self.Colonies[weakEmp]) == 0:
                self.Empires = np.delete(self.Empires, weakEmp, 0)
                self.EmpiresObj = np.delete(self.EmpiresObj, weakEmp)
                del self.Colonies[weakEmp]

            