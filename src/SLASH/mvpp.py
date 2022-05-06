import itertools
import math
import os.path
import re
import sys
import time

import clingo
import numpy as np
import torch

class MVPP(object):
    def __init__(self, program, k=1, eps=0.000001):
        self.k = k
        self.eps = eps

        # each element in self.pc is a list of atoms (one list for one prob choice rule)
        self.pc = []
        # each element in self.parameters is a list of probabilities
        self.parameters = []
        # each element in self.learnable is a list of Boolean values
        self.learnable = []
        # self.asp is the ASP part of the LPMLN program
        self.asp = ""
        # self.pi_prime is the ASP program \Pi' defined for the semantics
        self.pi_prime = ""
        # self.remain_probs is a list of probs, each denotes a remaining prob given those non-learnable probs
        self.remain_probs = []

        self.pc, self.parameters, self.learnable, self.asp, self.pi_prime, self.remain_probs = self.parse(program)
        self.normalize_probs()
        
        
    def debug_mvpp(self):
        print("pc",self.pc)
        print("params", self.parameters)
        print("learnable", self.learnable)
        print("asp", self.asp)
        print("pi_prime", self.pi_prime)
        
        

    def parse(self, program):

        pc = []
        parameters = []
        learnable = []
        asp = ""
        pi_prime = ""
        remain_probs = []
        npp_choices = {}

        lines = []
        # if program is a file
        if os.path.isfile(program):
            with open(program, 'r') as program:
                lines = program.readlines()
        # if program is a string containing all rules of an LPMLN program
        elif type(program) is str and re.sub(r'\n%[^\n]*', '\n', program).strip().endswith(('.', ']')):
            lines = program.split('\n')
        else:
            print("Error! The MVPP program {} is not valid.".format(program))
            sys.exit()

        #iterate over all lines
        # 1. build the choice rules 1{...}1
        # 2. store which npp values are learnable
        for line in lines:
            if re.match(r".*[0-1]\.?[0-9]*\s.*;.*", line):                
                out = re.search(r'@[0-9]*\.[0-9]*([a-z][a-zA-Z0-9_]*)\(([0-9]*),([0-9]),([a-z]*[a-zA-Z0-9]*)', line.replace(" ",""), flags=0)
                npp_name = out.group(1)
                npp_e = out.group(2)
                #out.group(3) is the identifier for the inference type e.g. 1,2,3,4 
                npp_input = out.group(4)

                list_of_atoms = []
                list_of_probs = []
                list_of_bools = []
                choices = line.strip()[:-1].split(";")
                for choice in choices:
                    prob, atom = choice.strip().split(" ", maxsplit=1)
                    # Note that we remove all spaces in atom since clingo output does not contain space in atom
                    list_of_atoms.append(atom.replace(" ", ""))
                    if prob.startswith("@"):
                        list_of_probs.append(float(prob[1:]))
                        list_of_bools.append(True)
                    else:
                        list_of_probs.append(float(prob))
                        list_of_bools.append(False)
                pc.append(list_of_atoms)
                parameters.append(list_of_probs)
                learnable.append(list_of_bools)

                if (npp_name, npp_input, npp_e) not in npp_choices:
                    npp_choices[(npp_name, npp_input,npp_e)]= [] 
                npp_choices[(npp_name, npp_input,npp_e)] += list_of_atoms
                
            else:
                asp += (line.strip()+"\n")

        #create choice rules for npp atoms
        for atom in npp_choices.keys():
            pi_prime += "1{"+"; ".join(npp_choices[atom])+"}1.\n"
        pi_prime += asp

        for ruleIdx, list_of_bools in enumerate(learnable):
            remain_prob = 1
            for atomIdx, b in enumerate(list_of_bools):
                if b == False:
                    remain_prob -= parameters[ruleIdx][atomIdx]
            remain_probs.append(remain_prob)


        return pc, parameters, learnable, asp, pi_prime, remain_probs

    def normalize_probs(self):
        for ruleIdx, list_of_bools in enumerate(self.learnable):
            summation = 0
            # 1st, we turn each probability into [0+eps,1-eps]
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    if self.parameters[ruleIdx][atomIdx] >=1 :
                        self.parameters[ruleIdx][atomIdx] = 1- self.eps
                    elif self.parameters[ruleIdx][atomIdx] <=0:
                        self.parameters[ruleIdx][atomIdx] = self.eps

            # 2nd, we normalize the probabilities
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    summation += self.parameters[ruleIdx][atomIdx]
            for atomIdx, b in enumerate(list_of_bools):
                if b == True:
                    self.parameters[ruleIdx][atomIdx] = self.parameters[ruleIdx][atomIdx] / summation * self.remain_probs[ruleIdx]
        return True

    def prob_of_interpretation(self, I):
        prob = 1.0
        # I must be a list of atoms, where each atom is a string
        while not isinstance(I[0], str):
            I = I[0]
        for ruleIdx,list_of_atoms in enumerate(self.pc):
            for atomIdx, atom in enumerate(list_of_atoms):
                if atom in I:
                    prob = prob * self.parameters[ruleIdx][atomIdx] #multiply the probabilities
        return prob

    # we assume query is a string containing a valid Clingo program, 
    # and each query is written in constraint form
    def find_one_SM_under_query(self, query):
        program = self.pi_prime + query
        clingo_control = clingo.Control(["--warn=none"])
        models = []
        clingo_control.add("base", [], program)
        clingo_control.ground([("base", [])])
        clingo_control.solve([], lambda model: models.append(model.symbols(atoms=True)))
        models = [[str(atom) for atom in model] for model in models]
        return models

    # we assume query is a string containing a valid Clingo program, 
    # and each query is written in constraint form
    def find_all_SM_under_query(self, query):
        program = self.pi_prime + query
        clingo_control = clingo.Control(["0", "--warn=none"])
        models = []
        try:
            clingo_control.add("base", [], program)
        except:
            print("\nPi': \n{}".format(program))
        clingo_control.ground([("base", [])])
        clingo_control.solve([], lambda model: models.append(model.symbols(atoms=True)))
        models = [[str(atom) for atom in model] for model in models]
        return models

    # k = 0 means to find all stable models
    def find_k_SM_under_query(self, query, k=3):

        program = self.pi_prime + query
        clingo_control = clingo.Control(["--warn=none", str(int(k))])
        models = []

        try:
            clingo_control.add("base", [], program)
        except:
            print("\nPi': \n{}".format(program))
            
        
        clingo_control.ground([("base", [])])
        clingo_control.solve([], lambda model: models.append(model.symbols(atoms=True)))
        models = [[str(atom) for atom in model] for model in models]

        return models

    # there might be some duplications in SMs when optimization option is used
    # and the duplications are removed by this method
    def remove_duplicate_SM(self, models):
        models.sort()
        return list(models for models,_ in itertools.groupby(models))

    # Note that the MVPP program cannot contain weak constraints
    def find_all_most_probable_SM_under_query_noWC(self, query):
        """Return a list of stable models, each is a list of strings
        @param query: a string of a set of constraints/facts
        """
        program = self.pi_prime + query + '\n'
        # for each probabilistic rule with n atoms, add n weak constraints
        for ruleIdx, atoms in enumerate(self.pc):
            for atomIdx, atom in enumerate(atoms):
                if self.parameters[ruleIdx][atomIdx] < 0.00674:
                    penalty = -1000 * -5
                else:
                    penalty = int(-1000 * math.log(self.parameters[ruleIdx][atomIdx]))
                program += ':~ {}. [{}, {}, {}]\n'.format(atom, penalty, ruleIdx, atomIdx)

        clingo_control = clingo.Control(['--warn=none', '--opt-mode=optN', '0', '-t', '8'])
        models = []
        clingo_control.add("base", [], program)
        clingo_control.ground([("base", [])])
        clingo_control.solve([], lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
        models = [[str(atom) for atom in model] for model in models]
        return self.remove_duplicate_SM(models)

    def find_one_most_probable_SM_under_query_noWC(self, query=''):
        """Return a list of a single stable model, which is a list of strings
        @param query: a string of a set of constraints/facts
        """
        
        program = self.pi_prime + query + '\n'
        # for each probabilistic rule with n atoms, add n weak constraints
        for ruleIdx, atoms in enumerate(self.pc):
            for atomIdx, atom in enumerate(atoms):
                if self.parameters[ruleIdx][atomIdx] < 0.00674:
                    penalty = -1000 * -5
                else:
                    penalty = int(-1000 * math.log(self.parameters[ruleIdx][atomIdx]))
                program += ':~ {}. [{}, {}, {}]\n'.format(atom, penalty, ruleIdx, atomIdx)
                
        clingo_control = clingo.Control(['--warn=none', '-t', '8'])#8 parallel mode param
        models = []
        #print(program)
        clingo_control.add("base", [], program)
        clingo_control.ground([("base", [])])
        clingo_control.solve([], lambda model: models.append(model.symbols(atoms=True)))
        models = [[str(atom) for atom in model] for model in models]
        return [models[-1]]

    def find_all_opt_SM_under_query_WC(self, query):
        """ Return a list of stable models, each is a list of strings
        @param query: a string of a set of constraints/facts

        """
        program = self.pi_prime + query
        clingo_control = clingo.Control(['--warn=none', '--opt-mode=optN', '0'])
        models = []
        try:
            clingo_control.add("base", [], program)
        except:
            print('\nSyntax Error in Program: Pi\': \n{}'.format(program))
            sys.exit()
        clingo_control.ground([("base", [])])
        clingo_control.solve([], lambda model: models.append(model.symbols(atoms=True)) if model.optimality_proven else None)
        models = [[str(atom) for atom in model] for model in models]
        return self.remove_duplicate_SM(models)

    # compute P(Q)
    def inference_query_exact(self, query):
        prob = 0
        models = self.find_all_SM_under_query(query)
        for I in models:
            prob += self.prob_of_interpretation(I)
        return prob
    
    # computes P(Q) given a list of stable models satisfying Q
    def sum_probability_for_stable_models(self, models):
        prob = 0
        for I in models:
            prob += self.prob_of_interpretation(I)
        return prob

    def gradient(self, ruleIdx, atomIdx, query):
        # we will compute P(I)/p_i where I satisfies query and c=v_i
        p_query_i = 0
        # we will compute P(I)/p_j where I satisfies query and c=v_j for i!=j
        p_query_j = 0
        # we will compute P(I) where I satisfies query
        p_query = 0

        # 1st, we generate all I that satisfies query
        models = self.find_k_SM_under_query(query, k=3)
        # 2nd, we iterate over each model I, and check if I satisfies c=v_i
        c_equal_vi = self.pc[ruleIdx][atomIdx]
        p_i = self.parameters[ruleIdx][atomIdx]
        for I in models:
            p_I = self.prob_of_interpretation(I)
            p_query += p_I
            if c_equal_vi in I:
                p_query_i += p_I/p_i
            else:
                for atomIdx2, p_j in enumerate(self.parameters[ruleIdx]):
                    c_equal_vj = self.pc[ruleIdx][atomIdx2]
                    if c_equal_vj in I:
                        p_query_j += p_I/p_j

        # 3rd, we compute gradient
        gradient = (p_query_i-p_query_j)/p_query
        return gradient

    def mvppLearnRule(self, ruleIdx, models, probs):
        """Return a np array denoting the gradients for the probabilities in rule ruleIdx

        @param ruleIdx: an integer denoting a rule index
        @param models: the list of models that satisfy an underlined query Q, each model is a list of string
        @param probs: a list of probabilities, one for each model
        """
        
        gradients = []
        # if there is only 1 stable model, we learn from complete interpretation
        if len(models) == 1:
            model = models[0]
            # we compute the gradient for each p_i in the ruleIdx-th rule
            p = 0
            one_in = False
            for i, cEqualsVi in enumerate(self.pc[ruleIdx]):
                if cEqualsVi in model:
                    gradients.append(1.0)
                    p = self.parameters[ruleIdx][i]
                    one_in = True
                else:
                    gradients.append(-1.0)
            for i, cEqualsVi in enumerate(self.pc[ruleIdx]):
                if p != 0:     
                    if not isinstance(p,torch.Tensor) :   
                        p = torch.Tensor([p])         
                    gradients[i] = gradients[i]/torch.min(torch.Tensor([1]),p+self.eps)
                    #gradients[i] = gradients[i]/p
            #not one c=v -> this means that the rule is an artifact of the +-operator and should not be learned
            if not one_in:
                gradients = np.zeros(len(gradients))

        # if there are more than 1 stable models, we use the equation in the proposition in the NeurASP paper
        else:
            denominator = sum(probs)
            # we compute the gradient for each p_i in the ruleIdx-th rule
            for i, cEqualsVi in enumerate(self.pc[ruleIdx]): 
                #i is the index for one network output and cEqualsVi contains an atom of the form digit(0,i1,0) which corresponds to i
                numerator = 0
                # we accumulate the numerator by looking at each model I that satisfies O
                for modelIdx, model in enumerate(models):
                    # if I satisfies cEqualsVi
                    if cEqualsVi in model:
                        if self.parameters[ruleIdx][i] != 0:
                            numerator += probs[modelIdx] / self.parameters[ruleIdx][i]
                        else:
                            numerator += probs[modelIdx] / (self.parameters[ruleIdx][i] + self.eps)


                    # if I does not satisfy cEqualsVi
                    else:
                        for atomIdx, atom in enumerate(self.pc[ruleIdx]):
                            if atom in model:
                                if self.parameters[ruleIdx][atomIdx]!=0:
                                    numerator -= probs[modelIdx] / self.parameters[ruleIdx][atomIdx]
                                else:
                                    numerator -= probs[modelIdx] / (self.parameters[ruleIdx][atomIdx]+self.eps)
                #if denominator == 0 :
                if denominator < self.eps:
                    gradients.append(0)
                else:
                    gradients.append(numerator / denominator)
                    #Note: When we have an NPP that is part of a rule with multiple querytypes then sometimes one part of the npp is "unused"
                    #This means that after computing the SM there never will be any of these npp instances
                    #The nominator will then be zero making all gradients zero. With this SLASH can filter out these unused npps to not include them in the backward pass  

        return np.array(gradients, dtype=object)

    def mvppLearn(self, models):
        probs = [self.prob_of_interpretation(model) for model in models]
        gradients = np.array([[0.0 for item in l] for l in self.parameters], dtype=object)
        if len(models) != 0:
            # we compute the gradients w.r.t. the probs in each rule
            for ruleIdx,list_of_bools in enumerate(self.learnable): #for mnist self.learnable contains two lists with 10 Booleans each with value True ->ruleIdx is then 0 or 1
                gradients[ruleIdx] = self.mvppLearnRule(ruleIdx, models, probs)
                for atomIdx, b in enumerate(list_of_bools):
                    if b == False:
                        gradients[ruleIdx][atomIdx] = 0


        return gradients

    # gradients are stored in numpy array instead of list
    # query is a string
    def gradients_one_query(self, query, opt=False):
        """Return an np-array denoting the gradients
        @param query: a string for query
        @param opt: a Boolean denoting whether we use optimal stable models instead of stable models
        """
        if opt:
            models = self.find_all_opt_SM_under_query_WC(query)
        else:
            models = self.find_k_SM_under_query(query, k=0)
        return self.mvppLearn(models), models

    # gradients are stored in numpy array instead of list
    def gradients_multi_query(self, list_of_query):
        gradients = [[0.0 for item in l] for l in self.parameters]
        for query in list_of_query:
            gradients = [[c+d for c,d in zip(i,j)] for i,j in zip(gradients,self.gradients_one_query(query))]
        return gradients

    # list_of_query is either a list of strings or a file containing queries separated by "#evidence"
    def learn_exact(self, list_of_query, lr=0.01, thres=0.0001, max_iter=None):
        # if list_of_query is an evidence file, we need to first turn it into a list of strings
        if type(list_of_query) is str and os.path.isfile(list_of_query):
            with open(list_of_query, 'r') as f:
                list_of_query = f.read().strip().strip("#evidence").split("#evidence")
        print("Start learning by exact computation with {} queries...\n\nInitial parameters: {}".format(len(list_of_query), self.parameters))
        time_init = time.time()
        check_continue = True
        iteration = 1
        while check_continue:
            old_parameters = self.parameters
            print("\n#### Iteration {} ####\n".format(iteration))
            check_continue = False
            dif = [[lr*grad for grad in l] for l in self.gradients_multi_query(list_of_query)]

            for ruleIdx, list_of_bools in enumerate(self.learnable):
            # 1st, we turn each gradient into [-0.2, 0.2]
                for atomIdx, b in enumerate(list_of_bools):
                    if b == True:
                        if dif[ruleIdx][atomIdx] > 0.2 :
                            dif[ruleIdx][atomIdx] = 0.2
                        elif dif[ruleIdx][atomIdx] < -0.2:
                            dif[ruleIdx][atomIdx] = -0.2

            self.parameters = [[c+d for c,d in zip(i,j)] for i,j in zip(dif,self.parameters)]
            self.normalize_probs()

            # we termintate if the change of the parameters is lower than thres
            dif = [[abs(c-d) for c,d in zip(i,j)] for i,j in zip(old_parameters,self.parameters)]
            print("After {} seconds of training (in total)".format(time.time()-time_init))
            print("Current parameters: {}".format(self.parameters))
            maxdif = max([max(l) for l in dif])
            print("Max change on probabilities: {}".format(maxdif))
            iteration += 1
            if maxdif > thres:
                check_continue = True
            if max_iter is not None:
                if iteration > max_iter:
                    check_continue = False
        print("\nFinal parameters: {}".format(self.parameters))

    ##############################
    ####### Sampling Method ######
    ##############################

    # it will generate k sample stable models for a k-coherent program under a specific total choice
    def k_sample(self):
        asp_with_facts = self.asp
        clingo_control = clingo.Control(["0", "--warn=none"])
        models = []
        for ruleIdx,list_of_atoms in enumerate(self.pc):
            tmp = np.random.choice(list_of_atoms, 1, p=self.parameters[ruleIdx])
            asp_with_facts += tmp[0]+".\n"
        clingo_control.add("base", [], asp_with_facts)
        clingo_control.ground([("base", [])])
        result = clingo_control.solve([], lambda model: models.append(model.symbols(shown=True)))
        models = [[str(atom) for atom in model] for model in models]
        return models

    # it will generate k*num sample stable models
    def sample(self, num=1):
        models = []
        for i in range(num):
            models = models + self.k_sample()
        return models

    # it will generate at least num of samples that satisfy query
    def sample_query(self, query, num=50):
        count = 0
        models = []
        while count < num:
            asp_with_facts = self.asp
            asp_with_facts += query
            clingo_control = clingo.Control(["0", "--warn=none"])
            models_tmp = []
            for ruleIdx,list_of_atoms in enumerate(self.pc):
                tmp = np.random.choice(list_of_atoms, 1, p=self.parameters[ruleIdx])
                asp_with_facts += tmp[0]+".\n"
            clingo_control.add("base", [], asp_with_facts)
            clingo_control.ground([("base", [])])
            result = clingo_control.solve([], lambda model: models_tmp.append(model.symbols(shown=True)))
            if str(result) == "SAT":
                models_tmp = [[str(atom) for atom in model] for model in models_tmp]
                count += len(models_tmp)
                models = models + models_tmp
            elif str(result) == "UNSAT":
                pass
            else:
                print("Error! The result of a clingo call is not SAT nor UNSAT!")
        return models

    # it will generate at least num of samples that satisfy query
    def sample_query2(self, query, num=50):
        count = 0
        models = []
        candidate_sm = []
        # we first find out all stable models that satisfy query
        program = self.pi_prime + query
        clingo_control = clingo.Control(['0', '--warn=none'])
        clingo_control.add('base', [], program)
        clingo_control.ground([('base', [])])
        clingo_control.solve([], lambda model: candidate_sm.append(model.symbols(shown=True)))
        candidate_sm = [[str(atom) for atom in model] for model in candidate_sm]
        probs = [self.prob_of_interpretation(model) for model in candidate_sm]

        while count < num:
            asp_with_facts = self.pi_prime
            asp_with_facts += query
            clingo_control = clingo.Control(["0", "--warn=none"])
            models_tmp = []
            for ruleIdx,list_of_atoms in enumerate(self.pc):
                tmp = np.random.choice(list_of_atoms, 1, p=self.parameters[ruleIdx])
                asp_with_facts += tmp[0]+".\n"
            clingo_control.add("base", [], asp_with_facts)
            clingo_control.ground([("base", [])])
            result = clingo_control.solve([], lambda model: models_tmp.append(model.symbols(shown=True)))
            if str(result) == "SAT":
                models_tmp = [[str(atom) for atom in model] for model in models_tmp]
                count += len(models_tmp)
                models = models + models_tmp
            elif str(result) == "UNSAT":
                pass
            else:
                print("Error! The result of a clingo call is not SAT nor UNSAT!")
        return models

    # we compute the gradients (numpy array) w.r.t. all probs in the ruleIdx-th rule
    # given models that satisfy query
    def gradient_given_models(self, ruleIdx, models):
        arity = len(self.parameters[ruleIdx])

        # we will compute N(O) and N(O,c=v_i)/p_i for each i
        n_O = 0
        n_i = [0]*arity

        # 1st, we compute N(O)
        n_O = len(models)

        # 2nd, we compute N(O,c=v_i)/p_i for each i
        for model in models:
            for atomIdx, atom in enumerate(self.pc[ruleIdx]):
                if atom in model:
                    n_i[atomIdx] += 1
        for atomIdx, p_i in enumerate(self.parameters[ruleIdx]):
            n_i[atomIdx] = n_i[atomIdx]/p_i
        
        # 3rd, we compute the derivative of L'(O) w.r.t. p_i for each i
        tmp = np.array(n_i) * (-1)
        summation = np.sum(tmp)
        gradients = np.array([summation]*arity)
        for atomIdx, p_i in enumerate(self.parameters[ruleIdx]):
            gradients[atomIdx] = gradients[atomIdx] + 2* n_i[atomIdx]
        gradients = gradients / n_O
        return gradients


    # gradients are stored in numpy array instead of list
    # query is a string
    def gradients_one_query_by_sampling(self, query, num=50):
        gradients = np.array([[0.0 for item in l] for l in self.parameters])
        # 1st, we generate at least num of stable models that satisfy query
        models = self.sample_query(query=query, num=num)

        # 2nd, we compute the gradients w.r.t. the probs in each rule
        for ruleIdx,list_of_bools in enumerate(self.learnable):
            gradients[ruleIdx] = self.gradient_given_models(ruleIdx, models)
            for atomIdx, b in enumerate(list_of_bools):
                if b == False:
                    gradients[ruleIdx][atomIdx] = 0
        return gradients

    # we compute the gradients (numpy array) w.r.t. all probs given list_of_query
    def gradients_multi_query_by_sampling(self, list_of_query, num=50):
        gradients = np.array([[0.0 for item in l] for l in self.parameters])

        # we itereate over all query
        for query in list_of_query:
            # 1st, we generate at least num of stable models that satisfy query
            models = self.sample_query(query=query, num=num) 

            # 2nd, we accumulate the gradients w.r.t. the probs in each rule
            for ruleIdx,list_of_bools in enumerate(self.learnable):
                gradients[ruleIdx] += self.gradient_given_models(ruleIdx, models)
                for atomIdx, b in enumerate(list_of_bools):
                    if b == False:
                        gradients[ruleIdx][atomIdx] = 0
        return gradients

    # we compute the gradients (numpy array) w.r.t. all probs given list_of_query
    # while we generate at least one sample without considering probability distribution
    def gradients_multi_query_by_one_sample(self, list_of_query):
        gradients = np.array([[0.0 for item in l] for l in self.parameters])

        # we itereate over all query
        for query in list_of_query:
            # 1st, we generate one stable model that satisfy query
            models = self.find_one_SM_under_query(query=query)

            # 2nd, we accumulate the gradients w.r.t. the probs in each rule
            for ruleIdx,list_of_bools in enumerate(self.learnable):
                gradients[ruleIdx] += self.gradient_given_models(ruleIdx, models)
                for atomIdx, b in enumerate(list_of_bools):
                    if b == False:
                        gradients[ruleIdx][atomIdx] = 0
        return gradients

    # list_of_query is either a list of strings or a file containing queries separated by "#evidence"
    def learn_by_sampling(self, list_of_query, num_of_samples=50, lr=0.01, thres=0.0001, max_iter=None, num_pretrain=1):
        # Step 0: Evidence Preprocessing: if list_of_query is an evidence file, 
        # we need to first turn it into a list of strings
        if type(list_of_query) is str and os.path.isfile(list_of_query):
            with open(list_of_query, 'r') as f:
                list_of_query = f.read().strip().strip("#evidence").split("#evidence")

        print("Start learning by sampling with {} queries...\n\nInitial parameters: {}".format(len(list_of_queries), self.parameters))
        time_init = time.time()

        # Step 1: Parameter Pre-training: we pretrain the parameters 
        # so that it's easier to generate sample stable models
        assert type(num_pretrain) is int
        if num_pretrain >= 1:
            print("\n#######################################################\nParameter Pre-training for {} iterations...\n#######################################################".format(num_pretrain))
            for iteration in range(num_pretrain):
                print("\n#### Iteration {} for Pre-Training ####\nGenerating 1 stable model for each query...\n".format(iteration+1))
                dif = lr * self.gradients_multi_query_by_one_sample(list_of_query)
                self.parameters = (np.array(self.parameters) + dif).tolist()
                self.normalize_probs()

                print("After {} seconds of training (in total)".format(time.time()-time_init))
                print("Current parameters: {}".format(self.parameters))

        # Step 2: Parameter Training: we train the parameters using "list_of_query until"
        # (i) the max change on probabilities is lower than "thres", or
        # (ii) the number of iterations is more than "max_iter"
        print("\n#######################################################\nParameter Training for {} iterations or until converge...\n#######################################################".format(max_iter))
        check_continue = True
        iteration = 1
        while check_continue:
            print("\n#### Iteration {} ####".format(iteration))
            old_parameters = np.array(self.parameters)            
            check_continue = False

            print("Generating {} stable model(s) for each query...\n".format(num_of_samples))
            dif = lr * self.gradients_multi_query_by_sampling(list_of_query, num=num_of_samples)

            self.parameters = (np.array(self.parameters) + dif).tolist()
            self.normalize_probs()
            
            print("After {} seconds of training (in total)".format(time.time()-time_init))
            print("Current parameters: {}".format(self.parameters))

            # we termintate if the change of the parameters is lower than thres
            dif = np.array(self.parameters) - old_parameters
            dif = abs(max(dif.min(), dif.max(), key=abs))
            print("Max change on probabilities: {}".format(dif))

            iteration += 1
            if dif > thres:
                check_continue = True
            if max_iter is not None:
                if iteration > max_iter:
                    check_continue = False

        print("\nFinal parameters: {}".format(self.parameters))