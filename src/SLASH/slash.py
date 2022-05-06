import pickle
import re
import sys
import time
import os

import clingo
import torch
from torch.autograd import grad
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch.nn as nn
import time
import utils

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../SLASH/')


from mvpp import MVPP
from sklearn.metrics import confusion_matrix

from tqdm import tqdm
from joblib import Parallel, delayed


def pad_3d_tensor(target, framework, bs, ruleNum, max_classes):

    if framework == 'numpy':
        padded = torch.tensor([np.hstack((np.asarray(row, dtype=np.float32),  [0] * (max_classes - len(row))) ) for batch in target for row in batch]).type(torch.FloatTensor).view(bs, ruleNum, max_classes)    
    
    if framework == 'torch':
        padded = torch.stack([torch.hstack((row,  torch.tensor([0] * (max_classes - len(row)), device="cuda" ) ) ) for batch in target for row in batch ]).view(ruleNum, bs, max_classes)
    return padded


def replace_plus_minus_occurences(pi_prime):
    pat_pm = r'(\s*[a-z][a-zA-Z0-9_]*)\((\s*[A-Z]*[a-zA-Z0-9_]*\s*),\s*\+([A-Z]*[a-zA-Z0-9_]*\s*),\s*\-([A-Z]*[a-zA-Z0-9_]*\s*)\)' #1 +- p(c|x) posterior
    pat_mp = r'(\s*[a-z][a-zA-Z0-9_]*)\((\s*[A-Z]*[a-zA-Z0-9_]*\s*),\s*\-([A-Z]*[a-zA-Z0-9_]*\s*),\s*\+([A-Z]*[a-zA-Z0-9_]*\s*)\)' #2 -+ p(x|c) likelihood
    pat_mm = r'(\s*[a-z][a-zA-Z0-9_]*)\((\s*[A-Z]*[a-zA-Z0-9_]*\s*),\s*\-([A-Z]*[a-zA-Z0-9_]*\s*),\s*\-([A-Z]*[a-zA-Z0-9_]*\s*)\)' #3 -- p(x,c) joint
    pat_pp = r'(\s*[a-z][a-zA-Z0-9_]*)\((\s*[A-Z]*[a-zA-Z0-9_]*\s*),\s*\+([A-Z]*[a-zA-Z0-9_]*\s*),\s*\+([A-Z]*[a-zA-Z0-9_]*\s*)\)' #3 -- p(c) prior

    #track which operator(+-,-+,--) was used for which npp
    npp_operators= {}
    for match in re.findall(pat_pm, pi_prime):
        if match[0] not in npp_operators:
            npp_operators[match[0]] = {}
        npp_operators[match[0]][1] = True
    for match in re.findall(pat_mp, pi_prime):
        if match[0] not in npp_operators:
            npp_operators[match[0]] = {}
        npp_operators[match[0]][2] = True
    for match in re.findall(pat_mm, pi_prime):
        if match[0] not in npp_operators:
            npp_operators[match[0]] = {}
        npp_operators[match[0]][3] = True
    for match in re.findall(pat_pp, pi_prime):
        if match[0] not in npp_operators:
            npp_operators[match[0]] = {}
        npp_operators[match[0]][4] = True
    
    #replace found matches with asp compatible form for npp occuring in rules
    #example: digit(0,+A,-N1) -> digit(0,1,A,N1)
    pi_prime = re.sub( pat_pm, r'\1(\2,1,\3,\4)', pi_prime)
    pi_prime = re.sub( pat_mp, r'\1(\2,2,\3,\4)', pi_prime)
    pi_prime = re.sub( pat_mm, r'\1(\2,3,\3,\4)', pi_prime)
    pi_prime = re.sub( pat_pp, r'\1(\2,4,\3,\4)', pi_prime)

    return pi_prime, npp_operators



def compute_gradients_splitwise(networkOutput_split, query_batch_split, mvpp, n, normalProbs, dmvpp, method, opt):
        """
        Computes the gradients, stable models and P(Q) for part of the batch.
        
        @param networkOutput_split:
        @param query_batch_split:
        @param mvpp:
        @param n:
        @param normalProbs:
        @param dmvpp:
        @param method:
        @param opt:
        :return:returns the gradients, the stable models and the probability P(Q)
        """
        
        
        query_batch_split = query_batch_split.tolist()
        
        #create a list to store the gradients into
        gradient_batch_list_split = []
        
        #create a list to store the stable models into
        model_batch_list_split = []
        
        #create a list to store p(Q) into
        prob_q_batch_list_split = []
        
        #iterate over all queries
        for bidx, query in enumerate(query_batch_split):
            
            # Step 2: we compute the semantic gradients 

            # Step 2.1: replace the parameters in the MVPP program with network outputs
            #iterate over all rules

            for ruleIdx in range(mvpp['networkPrRuleNum']):
                #for (m, i, inf_type, t, j) in mvpp['networkProb'][ruleIdx]:
                #    print("ruleIdx", ruleIdx)
                #    print("M:",m,"i:", i,"inf_type:", inf_type,"t:", t,"j:", j,"n[m]",n[m], "access",i*n[m]+j)

                #get the network outputs for the current element in the batch and put it into the correct rule
                dmvpp.parameters[ruleIdx] = [networkOutput_split[m][inf_type][t][bidx][i*n[m]+j] for (m, i, inf_type, t, j) in mvpp['networkProb'][ruleIdx]]
                if len(dmvpp.parameters[ruleIdx]) == 1:
                    dmvpp.parameters[ruleIdx] =  [dmvpp.parameters[ruleIdx][0][0],1-dmvpp.parameters[ruleIdx][0][0]]

            # Step 2.2: replace the parameters for normal prob. rules in the MVPP program with updated probabilities
            if normalProbs:
                for ruleIdx, probs in enumerate(normalProbs):
                    dmvpp.parameters[mvpp['networkPrRuleNum']+ruleIdx] = probs


            #dmvpp.normalize_probs()

            check = False

            query, _ = replace_plus_minus_occurences(query)
            

            if method == 'exact': #default exact
                gradients, models = dmvpp.gradients_one_query(query, opt=opt)
            elif method == 'slot':
                models = dmvpp.find_one_most_probable_SM_under_query_noWC(query)
                gradients = dmvpp.mvppLearn(models)
            elif method == 'sampling':
                models = dmvpp.sample_query(query, num=10)
                gradients = dmvpp.mvppLearn(models)
            elif method == 'network_prediction':
                models = dmvpp.find_one_most_probable_SM_under_query_noWC()
                check = SLASH.satisfy(models[0], mvpp['program_asp'] + query)
                gradients = dmvpp.mvppLearn(models) if check else -dmvpp.mvppLearn(models)
                if check:
                    continue
            elif method == 'penalty':
                models = dmvpp.find_all_SM_under_query()
                models_noSM = [model for model in models if not SLASH.satisfy(model, mvpp['program_asp'] + query)]
                gradients = - dmvpp.mvppLearn(models_noSM)
            else:
                print('Error: the method \'%s\' should be either \'exact\' or \'sampling\'', method)
            
            prob_q = dmvpp.sum_probability_for_stable_models(models)
            
            model_batch_list_split.append(models)
            gradient_batch_list_split.append(gradients)
            prob_q_batch_list_split.append(prob_q)
        
        return gradient_batch_list_split, model_batch_list_split, prob_q_batch_list_split





class SLASH(object):
    def __init__(self, dprogram, networkMapping, optimizers, gpu=True):

        """
        @param dprogram: a string for a NeurASP program
        @param networkMapping: a dictionary maps network names to neural networks modules
        @param optimizers: a dictionary maps network names to their optimizers
        
        @param gpu: a Boolean denoting whether the user wants to use GPU for training and testing
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

        self.dprogram = dprogram
        self.const = {} # the mapping from c to v for rule #const c=v.
        self.n = {} # the mapping from network name to an integer n denoting the domain size; n would be 1 or N (>=3); note that n=2 in theory is implemented as n=1
        self.max_n = 2 # integer denoting biggest domain of all npps
        self.e = {} # the mapping from network name to an integer e
        self.domain = {} # the mapping from network name to the domain of the predicate in that network atom
        self.normalProbs = None # record the probabilities from normal prob rules
        self.networkOutputs = {}
        self.networkGradients = {}
        self.networkTypes = {}
        if gpu==True:
            self.networkMapping = {key : nn.DataParallel(networkMapping[key].to(self.device)) for key in networkMapping}
        else:
            self.networkMapping = networkMapping
        self.optimizers = optimizers
        # self.mvpp is a dictionary consisting of 4 keys: 
        # 1. 'program': a string denoting an MVPP program where the probabilistic rules generated from network are followed by other rules;
        # 2. 'networkProb': a list of lists of tuples, each tuple is of the form (model, i ,term, j)
        # 3. 'atom': a list of list of atoms, where each list of atoms is corresponding to a prob. rule
        # 4. 'networkPrRuleNum': an integer denoting the number of probabilistic rules generated from network
        self.mvpp = {'networkProb': [], 'atom': [], 'networkPrRuleNum': 0, 'program': '','networkProbSinglePred':{}}
        self.mvpp['program'], self.mvpp['program_pr'], self.mvpp['program_asp'], self.npp_operators = self.parse(query='')
        self.stableModels = [] # a list of stable models, where each stable model is a list
        self.prob_q = [] # a list of probabilites for each query in the batch


    def constReplacement(self, t):
        """ Return a string obtained from t by replacing all c with v if '#const c=v.' is present

        @param t: a string, which is a term representing an input to a neural network
        """
        t = t.split(',')
        t = [self.const[i.strip()] if i.strip() in self.const else i.strip() for i in t]
        return ','.join(t)

    def networkAtom2MVPPrules(self, networkAtom, npp_operators):
        """
        @param networkAtom: a string of a neural atom
        @param countIdx: a Boolean value denoting whether we count the index for the value of m(t, i)[j]
        """
    
        print("networkAtom2MVPP")
        print(networkAtom,"\n")

        # STEP 1: obtain all information
        regex = '^(npp)\((.+)\((.+)\),\((.+)\)\)$'
        out = re.search(regex, networkAtom)        
        
        network_type = out.group(1)
        m = out.group(2)
        e, inf_type, t = out.group(3).split(',', 2) # in case t is of the form t1,...,tk, we only split on the second comma
        domain = out.group(4).split(',')
        inf_type =int(inf_type)

        #TODO how do we get the network type?
        self.networkTypes[m] = network_type

        t = self.constReplacement(t)
        # check the value of e
        e = e.strip()
        e = int(self.constReplacement(e))
        n = len(domain)
        if n == 2:
            n = 1
        self.n[m] = n
        if self.max_n <= n:
            self.max_n = n
        self.e[m] = e

        self.domain[m] = domain
        if m not in self.networkOutputs:
            self.networkOutputs[m] = {}

        for o in npp_operators[m]:
            if o not in self.networkOutputs[m]:
                self.networkOutputs[m][o] = {}
            self.networkOutputs[m][o][t]= None    
        
        # STEP 2: generate MVPP rules
        mvppRules = []

        # we have different translations when n = 2 (i.e., n = 1 in implementation) or when n > 2
        if n == 1:
            print("n is 1")
            for i in range(e):
                rule = '@0.0 {}({}, {}, {}, {}); @0.0 {}({},{}, {}, {}).'.format(m, i, inf_type, t, domain[0], m, i, inf_type, t, domain[1])
                prob = [tuple((m, i, inf_type, t, 0))]
                atoms = ['{}({}, {}, {}, {})'.format(m, i,inf_type, t, domain[0]), '{}({},{},{}, {})'.format(m, i, inf_type,  t, domain[1])]
                mvppRules.append(rule)
                self.mvpp['networkProb'].append(prob)
                self.mvpp['atom'].append(atoms)
                self.mvpp['networkPrRuleNum'] += 1
            self.mvpp['networkProbSinglePred'][m] = True
        elif n > 2:
            for i in range(e):
                rule = ''
                prob = []
                atoms = []
                for j in range(n):
                    atom = '{}({},{}, {}, {})'.format(m,  i, inf_type, t, domain[j])
                    rule += '@0.0 {}({},{}, {}, {}); '.format(m, i,inf_type, t, domain[j])
                    prob.append(tuple((m, i, inf_type, t, j)))
                    atoms.append(atom)
                mvppRules.append(rule[:-2]+'.')
                self.mvpp['networkProb'].append(prob)
                self.mvpp['atom'].append(atoms)
                self.mvpp['networkPrRuleNum'] += 1

            
        else:
            print('Error: the number of element in the domain %s is less than 2' % domain)
        return mvppRules



    def parse(self, query=''):
        dprogram = self.dprogram + query
        # 1. Obtain all const definitions c for each rule #const c=v.
        regex = '#const\s+(.+)=(.+).'
        out = re.search(regex, dprogram)
        if out:
            self.const[out.group(1).strip()] = out.group(2).strip()
            
        # 2. Generate prob. rules for grounded network atoms
        clingo_control = clingo.Control(["--warn=none"])
        
        # 2.1 remove weak constraints and comments
        program = re.sub(r'\n:~ .+\.[ \t]*\[.+\]', '\n', dprogram)
        program = re.sub(r'\n%[^\n]*', '\n', program)
        
        # 2.2 replace [] with ()
        program = program.replace('[', '(').replace(']', ')')
        

        # 2.3 use MVPP package to parse prob. rules and obtain ASP counter-part
        mvpp = MVPP(program)
        if mvpp.parameters and not self.normalProbs:
            self.normalProbs = mvpp.parameters
        pi_prime = mvpp.pi_prime


        #print("PI PRIME 1")
        #print(pi_prime)

        #2.4 parse +-Notation and add a const to flag the operation in the npp call
        pi_prime = pi_prime.replace(' ','').replace('#const','#const ')
        
        pi_prime, npp_operators = replace_plus_minus_occurences(pi_prime)

        #print("PI PRIME 2")
        #print(pi_prime)

        #extend npps definitions with the operators found
        #example: npp(digit(1,X),(0,1,2,3,4,5,6,7,8,9)):-img(X). with a +- and -- call in the program 
        # -> npp(digit(1,3,X),(0,1,2,3,4,5,6,7,8,9)):-img(X). npp(digit(1,1,X),(0,1,2,3,4,5,6,7,8,9)):-img(X).
        pat_npp = r'(npp\()([a-z]*[a-zA-Z0-9_]*)(\([0-9]*,)([A-Z]*[a-zA-Z0-9_]*\),\(([a-z0-9]*,)*[a-z0-9]*\)\))(:-[a-z][a-zA-Z0-9_]*\([A-Z][a-zA-Z0-9_]*\))?.'
        pat_npp = r'(npp\()([a-z]*[a-zA-Z0-9_]*)(\([0-9]*,)([A-Z]*[a-zA-Z0-9_]*\),\((?:[a-z0-9]*,)*[a-z0-9]*\)\))(:-[a-z][a-zA-Z0-9_]*\([A-Z][a-zA-Z0-9_]*\))?.'


        def npp_sub(match):
            npp_extended =""
            for o in npp_operators[match.group(2)]:
                    if match.group(5) is None:
                        body = ""
                    else: 
                        body = match.group(5)
                    npp_extended = '{}{}{}{},{}{}.\n'.format(match.group(1), match.group(2), match.group(3),o,match.group(4), body )+ npp_extended 
            return npp_extended

        pi_prime = re.sub(pat_npp, npp_sub, pi_prime)


        #print("PI PRIME 3")
        #print(pi_prime)

        # 2.5 use clingo to generate all grounded network atoms and turn them into prob. rules
        clingo_control.add("base", [], pi_prime)
        clingo_control.ground([("base", [])])
        #symbolic mappings map the constants to functions in the ASP program
        symbols = [atom.symbol for atom in clingo_control.symbolic_atoms]
        

        #iterate over all NPP atoms and extract information for the MVPP program
        mvppRules = [self.networkAtom2MVPPrules(str(atom),npp_operators) for atom in symbols if (atom.name == 'npp')]
        mvppRules = [rule for rules in mvppRules for rule in rules]
        
        # 3. obtain the ASP part in the original NeurASP program after +- replacements
        lines = [line.strip() for line in pi_prime.split('\n') if line and not re.match("^\s*npp\(", line)]

        return '\n'.join(mvppRules + lines), '\n'.join(mvppRules), '\n'.join(lines), npp_operators


    @staticmethod
    def satisfy(model, asp):
        """
        Return True if model satisfies the asp program; False otherwise
        @param model: a stable model in the form of a list of atoms, where each atom is a string
        @param asp: an ASP program (constraints) in the form of a string
        """
        asp_with_facts = asp + '\n'
        for atom in model:
            asp_with_facts += atom + '.\n'
        clingo_control = clingo.Control(['--warn=none'])
        clingo_control.add('base', [], asp_with_facts)
        clingo_control.ground([('base', [])])
        result = clingo_control.solve()
        if str(result) == 'SAT':
            return True
        return False

        
    def infer(self, dataDic, query='', mvpp='', postProcessing=None, dataIdx=None):

        """
        @param dataDic: a dictionary that maps terms to tensors/np-arrays
        @param query: a string which is a set of constraints denoting a query
        @param mvpp: an MVPP program used in inference
        """

        mvppRules = ''
        facts = ''

        # Step 1: get the output of each neural network
        for m in self.networkOutputs:
            self.networkMapping[m].eval()
            
            for o in self.networkOutputs[m]:
                for t in self.networkOutputs[m][o]:

                    if dataIdx is not None:
                        dataTensor = dataDic[t][dataIdx]
                    else:
                        dataTensor = dataDic[t]

                    print(dataTensor.shape)
                    
                    self.networkOutputs[m][o][t] = self.networkMapping[m](dataTensor).view(-1).tolist()

        for ruleIdx in range(self.mvpp['networkPrRuleNum']):
            
            probs = [self.networkOutputs[m][inf_type][t][i*self.n[m]+j] for (m, i, inf_type, t, j) in self.mvpp['networkProb'][ruleIdx]]
            #probs = [self.networkOutputs[m][inf_type][t][0][i*self.n[m]+j] for (m, i, inf_type, t, j) in self.mvpp['networkProb'][ruleIdx]]

            if len(probs) == 1:
                mvppRules += '@{:.15f} {}; @{:.15f} {}.\n'.format(probs[0], self.mvpp['atom'][ruleIdx][0], 1 - probs[0], self.mvpp['atom'][ruleIdx][1])
            else:
                tmp = ''
                for atomIdx, prob in enumerate(probs):
                    tmp += '@{:.15f} {}; '.format(prob, self.mvpp['atom'][ruleIdx][atomIdx])
                mvppRules += tmp[:-2] + '.\n'
        
        # Step 3: find an optimal SM under query
        dmvpp = MVPP(facts + mvppRules + mvpp)
        return dmvpp.find_one_most_probable_SM_under_query_noWC(query=query)


 
    def learn(self, dataset_loader, epoch, method='exact', lr=0.01, opt=False, batchSize=1, use_em=False, train_slot=False,  slot_net=None, p_num=1, marginalisation_masks=None):
        
        """
        @param dataList: a list of dictionaries, where each dictionary maps terms to either a tensor/np-array or a tuple (tensor/np-array, {'m': labelTensor})
        @param queryList: a list of strings, where each string is a set of constraints denoting query

        @param dataset_loader: a pytorch dataloader object returning a dictionary e.g {im1: [bs,28,28], im2:[bs,28,28]} and the queries
        @param epoch: an integer denoting the number of epochs
        @param method: a string in {'exact', 'sampling'} denoting whether the gradients are computed exactly or by sampling
        @param lr: a real number between 0 and 1 denoting the learning rate for the probabilities in probabilistic rules
        @param opt: stands for optimal -> if true we can select optimal stable models
        @param batchSize: a positive integer denoting the batch size, i.e., how many data instances do we use to update the network parameters for once
        @param p_num: a positive integer denoting the number of processor cores to be used during the training
        @param marginalisation_masks: a list entailing one marginalisation mask for each batch of dataList
        """
        
        assert p_num >= 1 and isinstance(p_num, int), 'Error: the number of processors used should greater equals one and a natural number'

        # get the mvpp program by self.mvpp
        if method == 'network_prediction':
            dmvpp = MVPP(self.mvpp['program_pr'])
        elif method == 'penalty':
            dmvpp = MVPP(self.mvpp['program_pr'])
        else:
            dmvpp = MVPP(self.mvpp['program'])

        
        # we train all neural network models
        for m in self.networkMapping:
            self.networkMapping[m].train() #torch training mode
            self.networkMapping[m].module.train() #torch training mode

        
        forward_time = 0.0 
        asp_time = 0.0
        backward_time = 0.0
                       
        # we train for 'epoch' times of epochs. Learning for multiple epochs can also be done in an outer loop by specifying epoch = 1
        for epochIdx in range(epoch):
        
            total_loss = []
    
            
            #iterate over all batches
            for data_batch, query_batch in tqdm(dataset_loader):
                
                start_time = time.time()
                           
                # If we have marginalisation masks, than we have to pick one for the batch
                if marginalisation_masks is not None:
                    marg_mask = marginalisation_masks[i]
                else:
                    marg_mask = None
                
                #STEP 0: APPLY SLOT ATTENTION TO TRANSFORM IMAGE TO SLOTS
                #we have a map which is : im: im_data
                #we want a map which is : s1: slot1_data, s2: slot2_data, s3: slot3_data
                if slot_net is not None:
                    if use_em or train_slot == False:
                        slot_net.eval()
                        with torch.no_grad():
                            dataTensor_after_slot = slot_net(data_batch['im'].to(self.device)) #forward the image

                    #only train the slot module if train_slot is true and we dont use EM
                    else: 
                        slot_net.train()
                        dataTensor_after_slot = slot_net(data_batch['im'].to(self.device)) #forward the image

                    #add the slot outputs to the data batch
                    for slot_num in range(slot_net.n_slots):
                        key = 's'+str(slot_num+1)
                        data_batch[key] = dataTensor_after_slot[:,slot_num,:]
                            
                                       
                    
                #data is a dictionary. we need to edit its key if the key contains a defined const c
                #where c is defined in rule #const c=v.
                data_batch_keys = list(data_batch.keys())
                for key in data_batch_keys:
                    #print(key, self.constReplacement(key))
                    data_batch[self.constReplacement(key)] = data_batch.pop(key)
                            
                
                # Step 1: get the output of each neural network and initialize the gradients
                networkOutput = {}
                networkLLOutput = {}
                
                #print("network outputs", self.networkOutputs)
                for m in self.networkOutputs:
                    if m not in networkOutput:
                        networkOutput[m] = {}

                    #print("M")

                    networkLLOutput[m] = {}
                    for o in self.networkOutputs[m]: #iterate over all output types and forwarded the input t trough the network
                        if o not in networkOutput[m]:
                            networkOutput[m][o] = {}
                                            
                        
                        ablation_output = None
                        #one forward pass to get the outputs
                        if self.networkTypes[m] == 'npp':
                            for t in self.networkOutputs[m][o]:


                                dataTensor = data_batch[t]
                                    
                                
                                #we have a list of data elements but want a Tensor of the form [batchsize,...]
                                if isinstance(dataTensor, list):
                                    dataTensor = torch.stack(dataTensor).squeeze(dim=1)



                                if m == 'color_abl':
                                    if ablation_output is None:
                                        ablation_output = self.networkMapping[m].forward(dataTensor.to(self.device))
                                    networkOutput[m][o][t] = ablation_output[:,0:9]
                                elif m == 'shape_abl':
                                    if ablation_output is None:
                                        ablation_output = self.networkMapping[m].forward(dataTensor.to(self.device))
                                    networkOutput[m][o][t] = ablation_output[:,9:13]
                                elif m == 'size_abl':
                                    if ablation_output is None:
                                        ablation_output = self.networkMapping[m].forward(dataTensor.to(self.device))
                                    networkOutput[m][o][t] = ablation_output[:,13:16]
                                elif m == 'shade_abl':
                                    if ablation_output is None:
                                        ablation_output = self.networkMapping[m].forward(dataTensor.to(self.device))
                                    networkOutput[m][o][t] = ablation_output[:,16:19]
                                    
                                else: 
                                    networkOutput[m][o][t] = self.networkMapping[m].forward(
                                                                                    dataTensor.to(self.device),
                                                                                    marg_idx=marg_mask,
                                                                                    type=o)

                                    #if the network predicts only one probability we add a placeholder for the false class
                                    if m in self.mvpp['networkProbSinglePred']:
                                        networkOutput[m][o][t] = torch.stack((networkOutput[m][o][t], torch.zeros_like(networkOutput[m][o][t])), dim=-1)



                        #store the outputs of the neural networks as a class variable
                        self.networkOutputs[m][o] = networkOutput[m][o] #this is of shape [first batch entry, second batch entry,...]
                                  
                        
                step1 = time.time()
                forward_time += step1 - start_time
                            
                # Step 2: compute the gradients

                if len(query_batch)< p_num:
                    p_num=query_batch.shape[0]

                #partition dictionary for different processors                
                splits = np.arange(0, int(p_num))
                partition = int(len(query_batch) / p_num)
                partition_mod = len(query_batch) % p_num 
                partition = [partition]*p_num
                partition[-1]+= partition_mod
                
                query_batch_split = np.split(query_batch,np.cumsum(partition))[:-1]


                #create an empty dictionary with nested structure 
                #{procces_id: {m: {t: {o:[bs_split, num_classes_of_property]}}}
                split_networkoutputs = {}
                for s in splits: #process id
                    if s not in split_networkoutputs:
                        split_networkoutputs[s] = {}
                        for m in networkOutput: #m
                            if m not in split_networkoutputs[s]:
                                split_networkoutputs[s][m] = {}
                            for o in networkOutput[m]: #t
                                if o not in split_networkoutputs[s][m]:
                                    split_networkoutputs[s][m][o] =  {}
                                    for t in networkOutput[m][o]:
                                        if t not in split_networkoutputs[s][m][o]:
                                            split_networkoutputs[s][m][o][t] =  {}


                #split the batch of npp outputs among the amount of processes
                for m in networkOutput:
                    for o in networkOutput[m]:
                        for t in networkOutput[m][o]:
                            split = torch.split(networkOutput[m][o][t].detach().cpu(),partition , dim=0)
                            for sidx, s in enumerate(split):
                                split_networkoutputs[sidx][m][o][t] = s.detach().cpu()



                #Compute the splits (gradients, P(Q), stable models) in parallel with JobLib
                split_outputs = Parallel(n_jobs=p_num, backend='loky')(
                    delayed(compute_gradients_splitwise)
                    (
                        list(split_networkoutputs.values())[i], query_batch_split[i],
                            self.mvpp, self.n, self.normalProbs,
                            dmvpp, method, opt
                    )
                         for i in range(p_num)) 



                #concatenate the gradients, stable models and query p(Q) from all splits back into a single batch
                gradient_batch_list_splits = []
                model_batch_list_splits = []
                prob_q_batch_list_splits = []

                for i in range(p_num):
                    gradient_batch_list_splits.append(split_outputs[i][0])
                    model_batch_list_splits.append(split_outputs[i][1])
                    prob_q_batch_list_splits.append(split_outputs[i][2])
                
                gradient_batch_list = np.concatenate(gradient_batch_list_splits)

                try:
                    model_batch_list = np.concatenate(model_batch_list_splits)
                except ValueError as e:
                    print("fix later")
                    #print(e)
                    #for i in range(0, len(model_batch_list_splits)):
                    #    print("NUM:",i)
                    #    print(model_batch_list_splits[i])
                
                prob_q_batch_list = np.concatenate(prob_q_batch_list_splits)


                #store the gradients, the stable models and p(Q) of the last batch processed
                self.networkGradients = gradient_batch_list
                self.stableModels = model_batch_list
                self.prob_q = prob_q_batch_list

                # Step 3: update parameters in neural networks
                step2 = time.time()
                asp_time += step2 - step1


                #the gradient list is a nested list with unequal dimensions. We pad it and transform it to a 3d-tensor
                gradient_batch_list = pad_3d_tensor(gradient_batch_list, 'numpy', len(query_batch),self.mvpp['networkPrRuleNum'],self.max_n ).to(device=self.device)                


                #transform hashmap of npp outputs to nested list and then transform it to a 3d tensor
                networkOutput_stacked = [] 

                for m in networkOutput:
                    for o in networkOutput[m]:
                        for t in networkOutput[m][o]:
                            networkOutput_stacked.append(networkOutput[m][o][t])

                networkOutput_stacked = pad_3d_tensor(networkOutput_stacked, 'torch', len(query_batch),self.mvpp['networkPrRuleNum'],self.max_n)


                #multiply every probability with its gradient 
                result = torch.einsum("bjc, jbc -> bjc", gradient_batch_list, networkOutput_stacked)
                

                #filter out npps that are not used 
                tmp = result.shape
                not_used_npps = result[result.abs().sum(dim=2) != 0].shape[0] / (tmp[0] * tmp[1])
                
                result = result[result.abs().sum(dim=2) != 0].sum()
                
                #get the number of discrete properties, e.g. sum all npps times the atoms entailing the npps 
                sum_discrete_properties = sum([len(listElem) for listElem in self.mvpp['atom']]) * len(query_batch)
                sum_discrete_properties = torch.Tensor([sum_discrete_properties]).to(device=self.device)

                #scale to actualy used npps
                sum_discrete_properties = sum_discrete_properties * not_used_npps


                #get the mean over the sum of discrete properties
                result_ll = result / sum_discrete_properties


                #backward pass
                #for EM we maximize the log likelihood
                if use_em:
                    result_ll.backward()
                    
                    #apply EM
                    for midx, m in enumerate(networkLLOutput):
                        #iterate over all neural literals t
                        self.networkMapping[m].module.em_process_batch()

                #for gradient descent we minimize the negative log likelihood    
                else:
                    result_nll = -result_ll
                    
                    #reset optimizers
                    for midx, m in enumerate(self.optimizers):
                        self.optimizers[m].zero_grad()

                    #append the loss value
                    total_loss.append(result_nll.cpu().detach().numpy())
                    
                    #backward pass
                    result_nll.backward(retain_graph=True)
                    
                    #apply gradients with each optimizer
                    for midx, m in enumerate(self.optimizers):
                        self.optimizers[m].step()
                    
            
            
                last_step = time.time()
                backward_time += last_step - step2
                        
            
                # Step 4: we update probabilities in normal prob. rules
                # TODO what happens here?
                if self.normalProbs:
                    gradientsNormal = gradients[self.mvpp['networkPrRuleNum']:].tolist()
                    for ruleIdx, ruleGradients in enumerate(gradientsNormal):
                        ruleIdxMVPP = self.mvpp['networkPrRuleNum']+ruleIdx
                        for atomIdx, b in enumerate(dmvpp.learnable[ruleIdxMVPP]):
                            if b == True:
                                dmvpp.parameters[ruleIdxMVPP][atomIdx] += lr * gradientsNormal[ruleIdx][atomIdx]
                    dmvpp.normalize_probs()
                    self.normalProbs = dmvpp.parameters[self.mvpp['networkPrRuleNum']:]
  

            #em update
            if use_em:
                for midx, m in enumerate(networkLLOutput):
                    self.networkMapping[m].module.em_update()    

            print("avg loss:", np.mean(total_loss))
            print("forward time: ", forward_time)
            print("asp time:", asp_time)
            print("backward time: ", backward_time)
            return np.mean(total_loss)


            

    def testNetwork(self, network, testLoader, ret_confusion=False):
        """
        Return a real number in [0,100] denoting accuracy
        @network is the name of the neural network or probabilisitc circuit to check the accuracy. 
        @testLoader is the input and output pairs.
        """
        self.networkMapping[network].eval()
        # check if total prediction is correct
        correct = 0
        total = 0
        # check if each single prediction is correct
        singleCorrect = 0
        singleTotal = 0
        
        #list to collect targets and predictions for confusion matrix
        y_target = []
        y_pred = []
        with torch.no_grad():
            for data, target in testLoader:
                                
                output = self.networkMapping[network](data.to(self.device))
                if self.n[network] > 2 :
                    pred = output.argmax(dim=-1, keepdim=True) # get the index of the max log-probability
                    target = target.to(self.device).view_as(pred)
                    
                    correctionMatrix = (target.int() == pred.int()).view(target.shape[0], -1)
                    y_target = np.concatenate( (y_target, target.int().flatten().cpu() ))
                    y_pred = np.concatenate( (y_pred , pred.int().flatten().cpu()) )
                    
                    
                    correct += correctionMatrix.all(1).sum().item()
                    total += target.shape[0]
                    singleCorrect += correctionMatrix.sum().item()
                    singleTotal += target.numel()
                else: 
                    pred = np.array([int(i[0]<0.5) for i in output.tolist()])
                    target = target.numpy()
                    
                    #y_target.append(target)
                    #y_pred.append(pred.int())
                    
                    correct += (pred.reshape(target.shape) == target).sum()
                    total += len(pred)
        accuracy = correct / total

        if self.n[network] > 2:
            singleAccuracy = singleCorrect / singleTotal
        else:
            singleAccuracy = 0
        
        #print(correct,"/", total, "=", correct/total)
        #print(singleCorrect,"/", singleTotal, "=", singleCorrect/singleTotal)

        
        if ret_confusion:
            confusionMatrix = confusion_matrix(np.array(y_target), np.array(y_pred))
            return accuracy, singleAccuracy, confusionMatrix

        return accuracy, singleAccuracy
    
    # We interprete the most probable stable model(s) as the prediction of the inference mode
    # and check the accuracy of the inference mode by checking whether the query is satisfied by the prediction
    def testInferenceResults(self, dataset_loader):
        """ Return a real number in [0,1] denoting the accuracy
        @param dataset_loader: a dataloader object loading a dataset to test on
        """

        correct = 0
        len_dataset = 0
        #iterate over batch
        for data_batch, query_batch in dataset_loader:
            len_dataset += len(query_batch)

            #iterate over each entry in batch
            for dataIdx in range(0, len(query_batch)):
                models = self.infer(data_batch, query=':- mistake.', mvpp=self.mvpp['program_asp'],  dataIdx= dataIdx)

                query,_ =  replace_plus_minus_occurences(query_batch[dataIdx])

                for model in models:
                    if self.satisfy(model, query):
                        correct += 1
                        break

        accuracy = 100. * correct / len_dataset
        return accuracy


    def testConstraint(self, dataset_loader, mvppList):
        """
        @param dataList: a list of dictionaries, where each dictionary maps terms to tensors/np-arrays
        @param queryList: a list of strings, where each string is a set of constraints denoting a query
        @param mvppList: a list of MVPP programs (each is a string)
        """

        # we evaluate all nerual networks
        for func in self.networkMapping:
            self.networkMapping[func].eval()

        # we count the correct prediction for each mvpp program
        count = [0]*len(mvppList)


        len_data = 0
        for data_batch, query_batch in dataset_loader:
            len_data += len(query_batch)

            # data is a dictionary. we need to edit its key if the key contains a defined const c
            # where c is defined in rule #const c=v.
            data_batch_keys = list(data_batch.keys())
            for key in data_batch_keys:
                data_batch[self.constReplacement(key)] = data_batch.pop(key)

            # Step 1: get the output of each neural network

            for m in self.networkOutputs:
                for o in self.networkOutputs[m]: #iterate over all output types and forwarded the input t trough the network
                    for t in self.networkOutputs[m][o]:
                        self.networkOutputs[m][o][t] = self.networkMapping[m].forward(
                                                                                    data_batch[t].to(self.device),
                                                                                    marg_idx=None,
                                                                                    type=o)

            # Step 2: turn the network outputs into a set of ASP facts            
            aspFactsList = []
            for bidx in range(len(query_batch)):
                aspFacts = ''
                for ruleIdx in range(self.mvpp['networkPrRuleNum']):
                    
                    #get the network outputs for the current element in the batch and put it into the correct rule
                    probs = [self.networkOutputs[m][inf_type][t][bidx][i*self.n[m]+j] for (m, i, inf_type, t, j) in self.mvpp['networkProb'][ruleIdx]]

                    if len(probs) == 1:
                        atomIdx = int(probs[0] < 0.5) # t is of index 0 and f is of index 1
                    else:
                        atomIdx = probs.index(max(probs))
                    aspFacts += self.mvpp['atom'][ruleIdx][atomIdx] + '.\n'
                aspFactsList.append(aspFacts)
        

            # Step 3: check whether each MVPP program is satisfied
            for bidx in range(len(query_batch)):
                for programIdx, program in enumerate(mvppList):

                    query,_ =  replace_plus_minus_occurences(query_batch[bidx])
                    program,_ = replace_plus_minus_occurences(program)

                    # if the program has weak constraints
                    if re.search(r':~.+\.[ \t]*\[.+\]', program) or re.search(r':~.+\.[ \t]*\[.+\]', query):
                        choiceRules = ''
                        for ruleIdx in range(self.mvpp['networkPrRuleNum']):
                            choiceRules += '1{' + '; '.join(self.mvpp['atom'][ruleIdx]) + '}1.\n'


                        mvpp = MVPP(program+choiceRules)
                        models = mvpp.find_all_opt_SM_under_query_WC(query=query)
                        models = [set(model) for model in models] # each model is a set of atoms
                        targetAtoms = aspFacts[bidx].split('.\n')
                        targetAtoms = set([atom.strip().replace(' ','') for atom in targetAtoms if atom.strip()])
                        if any(targetAtoms.issubset(model) for model in models):
                            count[programIdx] += 1
                    else:
                        mvpp = MVPP(aspFacts[bidx] + program)
                        if mvpp.find_one_SM_under_query(query=query):
                            count[programIdx] += 1
        for programIdx, program in enumerate(mvppList):
            print('The accuracy for constraint {} is {}({}/{})'.format(programIdx+1, float(count[programIdx])/len_data,float(count[programIdx]),len_data ))

            
            
    
    def forward_slot_attention_pipeline(self, slot_net, dataset_loader):
        """
        Makes one forward pass trough the slot attention pipeline to obtain the probabilities/log likelihoods for all classes for each object. 
        The pipeline includes  the SlotAttention module followed by probabilisitc circuits for probabilites for the discrete properties.
        @param slot_net: The SlotAttention module
        @param dataset_loader: Dataloader containing a shapeworld/clevr dataset to be forwarded
        """
        with torch.no_grad():

            probabilities = {}  # map to store all output probabilities(posterior)
            ll = {} #map to store all log likelihoods
            slot_map = {} #map to store all slot module outputs
            
            for data_batch, _,_ in dataset_loader:
                
                #forward img to get slots
                dataTensor_after_slot = slot_net(data_batch['im'].to(self.device))#SHAPE [BS,SLOTS, SLOTSIZE]
                
                #dataTensor_after_slot has shape [bs, num_slots, slot_vector_length]
                _, num_slots ,_ = dataTensor_after_slot.shape 

                
                for sdx in range(0, num_slots):
                    slot_map["s"+str(sdx)] = dataTensor_after_slot[:,sdx,:]
                

                #iterate over all slots and forward them through all nets (shape + color + ... )
                for key in slot_map:
                    if key not in probabilities:
                        probabilities[key] = {}
                        ll[key] = {}
                    
                    for network in self.networkMapping:
                        posterior= self.networkMapping[network].forward(slot_map[key])#[BS, num_discrete_props]
                        if network not in probabilities[key]:
                            probabilities[key][network] = posterior
                        else: 
                            probabilities[key][network] = torch.cat((probabilities[key][network], posterior))

                        #ll[key][network] = out

            return probabilities, ll
