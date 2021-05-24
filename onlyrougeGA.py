import random
from features import compute_fitness
from preprocess import preprocess_raw_sent
from copy import copy
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import nltk
import os.path
import statistics as sta
from preprocess import sim_with_doc
from preprocess import sim_2_sent
from preprocess import count_noun
from rouge import Rouge
import re
import time
import os
import glob
from shutil import copyfile
import pandas as pd
import math
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer


class Summerizer(object):
    def __init__(self, title, sentences, raw_sentences, population_size, max_generation, crossover_rate, mutation_rate, num_picked_sents, rougeforsentences, abstract, order_params):
        self.title = title
        self.raw_sentences = raw_sentences
        self.sentences = sentences
        self.num_objects = len(sentences)
        self.population_size = population_size
        self.max_generation = max_generation
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.num_picked_sents = num_picked_sents
        self.rougeforsentences = rougeforsentences
        self.order_params = order_params
        self.abstract = abstract


    def generate_population(self, amount):
        # print("Generating population...")
        population = []
        typeA = []
        typeB = []

        
        # for i in range(int(amount/2)):
        for i in range(amount):
            #creat type A
            chromosome1 = np.zeros(self.num_objects)
            chromosome1[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
            chromosome1 =  chromosome1.tolist()
            fitness1 = compute_fitness(self.title, self.sentences, chromosome1, self.rougeforsentences, self.abstract, self.order_params)
            
            chromosome2 = np.zeros(self.num_objects)
            chromosome2[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
            chromosome2 =  chromosome2.tolist()
            fitness2 = compute_fitness(self.title, self.sentences, chromosome2, self.rougeforsentences, self.abstract, self.order_params)
            fitness = max(fitness1, fitness2)
            
            typeA.append((chromosome1, chromosome2, fitness))
            
            
            #creat type B
            chromosome3 = np.zeros(self.num_objects)
            chromosome3[np.random.choice(list(range(self.num_objects)), self.num_picked_sents, replace=False)] = 1
            chromosome3 =  chromosome3.tolist()
            fitness3 = compute_fitness(self.title, self.sentences, chromosome3, self.rougeforsentences, self.abstract, self.order_params)
            chromosome4 = []
            typeB.append((chromosome3, chromosome4, fitness3))


        population.append(typeA)
        population.append(typeB)
        return population 

    def roulette_select(self, total_fitness, population):
        ps = len(population)
        if ps == 0:
            return None
        fitness_slice = np.random.rand() * total_fitness
        fitness_so_far = 0.0
        for phenotype in population:
            fitness_so_far += phenotype[2]
            if fitness_so_far >= fitness_slice:
                return phenotype
        return None


    def rank_select(self, population):
        ps = len(population)
        if ps == 0:
            return None
        population = sorted(population, key=lambda x: x[2], reverse=True)
        fitness_value = []
        for individual in population:
            fitness_value.append(individual[2])
        if len(fitness_value) == 0:
            return None
        fittest_individual = max(fitness_value)
        medium_individual = sta.median(fitness_value)
        selective_pressure = fittest_individual - medium_individual
        j_value = 1
        a_value = np.random.rand()   
        for agent in population:
            if ps == 0:
                return None
            elif ps == 1:
                return agent
            else:
                range_value = selective_pressure - (2*(selective_pressure - 1)*(j_value - 1))/( ps - 1) 
                prb = range_value/ps
                if prb > a_value:
                    return agent
            j_value +=1

    def reduce_no_memes(self, agent, max_sent):
        sum_sent_in_summary = sum(agent)
        if sum_sent_in_summary > max_sent:
            while(sum_sent_in_summary > max_sent):
                remove_point = 1 + random.randint(0, self.num_objects - 2)
                if agent[remove_point] == 1:
                    agent[remove_point] = 0
                    sent = self.sentences[remove_point]
                    sum_sent_in_summary -=1    
        return agent

    def crossover(self, individual_1, individual_2, max_sent):

        # check tỷ lệ crossover
        if self.num_objects < 2 or random.random() >= self.crossover_rate:
            return individual_1[:], individual_2[:]

        individual_2 = random.choice(individual_2[:-1])

        if len(individual_2) == 0:
            fitness1 = compute_fitness(self.title, self.sentences, individual_1[0], self.rougeforsentences, self.abstract, self.order_params)
            child1 = (individual_1[0], individual_2, fitness1)
            fitness2 = compute_fitness(self.title, self.sentences, individual_1[1], self.rougeforsentences, self.abstract, self.order_params)
            child2 = (individual_1[1], individual_2, fitness2)
            return child1, child2



        individual_1 = random.choice(individual_1[:-1])
        
        #tìm điểm chéo 1
        crossover_point = 1 + random.randint(0, self.num_objects - 2)
        agent_1a = individual_1[:crossover_point] + individual_2[crossover_point:]
        agent_1a = self.reduce_no_memes(agent_1a, max_sent)
        fitness_1a = compute_fitness(self.title, self.sentences, agent_1a, self.rougeforsentences, self.abstract, self.order_params)
        
        agent_1b = individual_2[:crossover_point] + individual_1[crossover_point:]
        agent_1b = self.reduce_no_memes(agent_1b, max_sent)
        fitness_1b = compute_fitness(self.title, self.sentences, agent_1b, self.rougeforsentences, self.abstract, self.order_params)
        
        if fitness_1a > fitness_1b:
            child_1 = (agent_1a, agent_1b, fitness_1a)
        else:
            child_1 = (agent_1a, agent_1b, fitness_1b)

        #tìm điểm chéo 2
        crossover_point_2 = 1 + random.randint(0, self.num_objects - 2)
        
        agent_2a = individual_1[:crossover_point_2] + individual_2[crossover_point_2:]
        agent_2a = self.reduce_no_memes(agent_2a, max_sent)
        fitness_2a = compute_fitness(self.title, self.sentences, agent_2a, self.rougeforsentences, self.abstract, self.order_params)
        
        agent_2b = individual_2[:crossover_point_2] + individual_1[crossover_point_2:]
        agent_2b = self.reduce_no_memes(agent_2b, max_sent)
        fitness_2b = compute_fitness(self.title, self.sentences, agent_2b, self.rougeforsentences, self.abstract, self.order_params)
        
        if fitness_2a > fitness_2b:
            child_2 = (agent_2a, agent_2b, fitness_2a)
        else:
            child_2 = (agent_2a, agent_2b, fitness_2b)        
        
        return child_1, child_2
    

    def mutate(self, individual, max_sent):
        sum_sent_in_summary = sum(individual[0])
        sum_sent_in_summary2 =sum(individual[1])
        if len(individual[1]) == 0:
            self.mutation_rate = 2/self.num_objects
            chromosome = individual[0][:]
            for i in range(len(chromosome)):
                if random.random() < self.mutation_rate and sum_sent_in_summary < max_sent :
                    if chromosome[i] == 0 :
                        chromosome[i] = 1
                        sum_sent_in_summary +=1
                    elif random.random() < 0.05:
                        chromosome[i] = 0
                        sum_sent_in_summary -=1     
            fitness = compute_fitness(self.title, self.sentences, chromosome , self.rougeforsentences, self.abstract, self.order_params)
            return (chromosome, individual[1], fitness)

        if random.random() < 0.05 :
            chromosome  = random.choice(individual[:-1])
            null_chromosome = []
            fitness = compute_fitness(self.title, self.sentences, chromosome , self.rougeforsentences, self.abstract, self.order_params)
            return(chromosome, null_chromosome, fitness) 


        chromosome1 = individual[0][:]
        chromosome2 = individual[1][:]
        self.mutation_rate = 1/self.num_objects

        for i in range(len(chromosome1)):
            if random.random() < self.mutation_rate and sum_sent_in_summary < max_sent :
                if chromosome1[i] == 0 :
                    chromosome1[i] = 1
                    sum_sent_in_summary +=1
                elif random.random() < 0.05:
                    chromosome1[i] = 0
                    sum_sent_in_summary -=1 
        
        
        for i in range(len(chromosome2)):
            if random.random() < self.mutation_rate and sum_sent_in_summary2 < max_sent :
                if chromosome2[i] == 0 :
                    chromosome2[i] = 1
                    sum_sent_in_summary2 +=1
                elif random.random() < 0.05:
                    chromosome1[i] = 0
                    sum_sent_in_summary -=1 
        
        
        fitness1 = compute_fitness(self.title, self.sentences, chromosome1, self.rougeforsentences, self.abstract, self.order_params)
        fitness2 = compute_fitness(self.title, self.sentences, chromosome2, self.rougeforsentences, self.abstract, self.order_params)
        fitness = max(fitness1, fitness2)
        return (chromosome1, chromosome2, fitness)

    def compare(self, lst1, lst2):
        for i in range(self.num_objects):
            if lst1[i] != lst2[i]:
                return False
        return True

    def survivor_selection (self, individual , population, check, max_sent ):
        if len(population) > 4 :
            competing = random.sample(population, 4)
            lowest_individual = max(competing , key = lambda x: x[2])
            if individual[2] > lowest_individual[2]:
                check = 1
                return individual, check
            else:
                check = 1
                return lowest_individual, check
        return individual, check


    def selection(self, population):
        max_sent = int(0.2*self.num_objects)
        condition_for_survivor = 0 # khoong co survivor
        tmp_popu = population.copy()
        new_population = []
        new_typeA = []
        new_typeB = []
    
        


        if condition_for_survivor == 1:
            population[0] = sorted(population[0], key=lambda x: x[2], reverse=True)
            population[1] = sorted(population[1], key=lambda x: x[2], reverse=True)
            chosen_agents_A = int(0.1*len(population[0]))
            chosen_agents_B = int(0.1*len(population[1]))
            
            elitismA = population[0][ : chosen_agents_A ]
            new_typeA = elitismA

            elitismB = population[1][ : chosen_agents_B]
            new_typeB = elitismB

            population[0] = population[0][ chosen_agents_A :]
            population[1] = population[1][ chosen_agents_B :]
        
        total_fitness = 0
        for indivi in population[1]:
            total_fitness = total_fitness + indivi[2]  
        
        population_size = self.population_size
        cpop = 0.0

        #chọn cá thể  loại A bằng rank_selection, cá thể loại B bằng RW
        while cpop <= population_size:
            population[0] = sorted(population[0], key=lambda x: x[2], reverse=True)

            parent_1 = None
            check_time_1 = time.time()
            while parent_1 == None:
                parent_1 = self.rank_select(population[0])
                if parent_1 == None and (time.time() - check_time_1) > 60:
                    return tmp_popu

            parent_2 = None
            check_time_2 = time.time()
            while parent_2 == None :
                parent_2 = self.roulette_select(total_fitness, population[1])
                if parent_2 == None and (time.time() - check_time_2) > 60:
                    return tmp_popu

                if parent_2 != None:
                    if self.compare(parent_2[0], parent_1[0]) or self.compare(parent_2[0], parent_1[1]):
                        parent_2 = self.roulette_select(total_fitness, population[1])


            parent_1, parent_2 = copy(parent_1), copy(parent_2)
            child_1, child_2 = self.crossover(parent_1, parent_2, max_sent)

            check1 = 0
            check2 = 0

            # child_1
            individual_X = self.mutate(child_1, max_sent)
            # child_2
            individual_Y = self.mutate(child_2, max_sent)

            if condition_for_survivor == 1:
                #Nếu X loại B:
                if len(individual_X[1]) == 0:
                    individual_X , check1 = self.survivor_selection(individual_X, population[1], check1, max_sent)
                    if check1 == 1:
                        new_typeB.append(individual_X)
                else:
                    individual_X , check1 = self.survivor_selection(individual_X, population[0], check1, max_sent)
                    if check1 == 1:
                        new_typeA.append(individual_X)

                #Nếu Y loại B:
                if len(individual_Y[1]) == 0:
                    individual_Y , check1 = self.survivor_selection(individual_Y, population[1], check2, max_sent)
                    if check2 == 1:
                        new_typeB.append(individual_Y)
                else:
                    individual_Y , check1 = self.survivor_selection(individual_Y, population[0], check2, max_sent)
                    if check2 == 1:
                        new_typeA.append(individual_Y)

            else:
                check1 = 1
                check2 = 1
                if len(individual_X[1]) == 0:
                    tmp_popu[1].append(individual_X)
                else:
                    tmp_popu[0].append(individual_X)
                if len(individual_Y[1]) == 0:
                    tmp_popu[1].append(individual_Y)
                else:
                    tmp_popu[0].append(individual_Y)

            if check1 + check2 == 0:
                cpop += 0.1
            else:
                cpop += check1 + check2

        if condition_for_survivor == 0:
            new_typeA = sorted(tmp_popu[0], key=lambda x: x[2], reverse=True)[:self.population_size]
            new_typeB = sorted(tmp_popu[1], key=lambda x: x[2], reverse=True)[:self.population_size]

        new_population.append(new_typeA)
        new_population.append(new_typeB)        
        fitness_value = []
        for individual in new_typeA:
            fitness_value.append(individual[2])
        for individual in new_typeB:
            fitness_value.append(individual[2])   
        try:
            avg_fitness = sta.mean(fitness_value)
        except: 
            return tmp_popu
        agents_in_Ev = [] 

        for agent in new_typeA:
            if (agent[2] > 0.95*avg_fitness) and (agent[2] < 1.05*avg_fitness):
                agents_in_Ev.append(agent)
        for agent in new_typeB:
            if (agent[2] > 0.95*avg_fitness) and (agent[2] < 1.05*avg_fitness):
                agents_in_Ev.append(agent)


        if len(agents_in_Ev) >= self.population_size*0.9:

            new_population = self.generate_population(int(self.population_size*0.7))
            chosen = self.population_size - int(self.population_size*0.7)

            type_A = new_population[0]
            type_B = new_population[1]

            new_typeA = sorted(new_typeA, key=lambda x: x[2], reverse=True)
            new_typeB = sorted(new_typeB, key=lambda x: x[2], reverse=True)
            new_typeA = new_typeA[ : chosen]
            new_typeB = new_typeB[ : chosen]


            for x in new_typeA:
                type_A.append(x)
            for y in new_typeB:
                type_B.append(y)
            new_population = []
            new_population.append(type_A)
            new_population.append(type_B)

        return new_population 
    

    def find_best_individual(self, population, final_res):
        try:
            population[0] = sorted(population[0], key=lambda x: x[2], reverse=True)
            population[1] = sorted(population[1], key=lambda x: x[2], reverse=True)
            best_type_A = population[0][0]
            best_type_B = population[1][0]
            if best_type_A[2] > best_type_B[2]:
                fitness1 = compute_fitness(self.title, self.sentences, best_type_A[0], self.rougeforsentences, self.abstract, self.order_params)
                fitness2 = compute_fitness(self.title, self.sentences, best_type_A[1], self.rougeforsentences, self.abstract, self.order_params)
                if fitness1 >= fitness2:
                    best = (best_type_A[0], fitness1)
                else:
                    best = (best_type_A[1], fitness2)
            else:
                best = (best_type_B[0], best_type_B[2])
                
            if best[1] >= final_res[2]:
                return best
            else:
                fitness1 = compute_fitness(self.title, self.sentences, final_res[0], self.rougeforsentences, self.abstract, self.order_params)
                fitness2 = compute_fitness(self.title, self.sentences, final_res[1], self.rougeforsentences, self.abstract, self.order_params)
                if fitness1 >= fitness2:
                    best = (final_res[0], fitness1)
                else:
                    best = (final_res[1], fitness2)
                return best
       
        except:
            return None



    def check_best(self, arr):
        if len(arr) > 20:
            reversed_arr = arr[::-1][:20]
            if reversed_arr[0] > reversed_arr[-1]:
                return True
            else: 
                return False
        else:
            return True



    def solve(self):
        population = self.generate_population(self.population_size)
        best_individual = sorted(population[0], key=lambda x: x[2], reverse=True)[0]
        best_fitness_value = best_individual[0]
        tmp_arr = []
        tmp_arr.append(best_fitness_value[2])
        count = 0
        final_res = best_individual
        while count < self.max_generation and (self.check_best(tmp_arr) == True):
            population = self.selection(population)
            try:
                best_individual = sorted(population[0], key=lambda x: x[2], reverse=True)[0]
            except:
                return None
            best_fitness_value = best_individual[2]  
            tmp_arr.append(best_fitness_value)  
            if final_res[2] < best_individual[2]:
                final_res = best_individual
            count +=1
        return self.find_best_individual(population, final_res)
    
    
    def show(self, individual,  file):
        index = individual[0]
        f = open(file,'w', encoding='utf-8')
        for i in range(len(index)):
            if index[i] == 1:
                f.write(self.raw_sentences[i] + ' ')
        f.close()

def load_a_doc(filename):
    file = open(filename, encoding='utf-8')
    article_text = file.read()
    file.close()
    return article_text   


def load_docs(directory):
	docs = list()  
	for name in os.listdir(directory):
		filename = directory + '/' + name
		doc = load_a_doc(filename)
		docs.append((doc, name))
	return docs

def clean_text(text):
    cleaned = "".join(u for u in text if u not in ("?", ".", ";", ":", "!", ",", "'", "(", ")")).strip()
    check_text = "".join((item for item in cleaned if not item.isdigit())).strip()
    if len(check_text.split(" ")) < 4:
        return 'None'
    return text

def evaluate_rouge(raw_sentences, abstract):
    rouge_scores = []
    for sent in raw_sentences:
        try:
            rouge = Rouge()
            scores = rouge.get_scores(sent, abstract, avg=True)
            rouge1f = scores["rouge-1"]["f"]
        except Exception:
            rouge1f = 0 
        rouge_scores.append(rouge1f)
    return rouge_scores
  
def solution_for_exception(rougeforsentences, raw_sentences, file_name):
    rank_rougeforsentences = sorted(rougeforsentences, key=lambda x: x[1], reverse=True)
    length_of_summary = int(0.2*len(raw_sentences))
    rank_rouge = rank_rougeforsentences[ : length_of_summary]
    rank_rouge = sorted(rank_rouge, key=lambda x: x[2], reverse=False)    
    f = open(file_name,'w', encoding='utf-8')
    for sent in rank_rouge:
        f.write(sent[0] + ' ')
    f.close()

def start_run(processID, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, sub_stories, save_path, order_params):
   
    for example in sub_stories:
        start_time = time.time()
        raw_sents = re.split("\n\n", example[0])[1].split(' . ')
        title = re.split("\n\n", example[0])[0] 
        abstract = re.split("\n\n", example[0])[2]

        #remove too short sentences
        df = pd.DataFrame(raw_sents, columns =['raw'])
        df['preprocess_raw'] = df['raw'].apply(lambda x: clean_text(x))
        newdf = df.loc[(df['preprocess_raw'] != 'None')]
        raw_sentences = newdf['preprocess_raw'].values.tolist()
        if len(raw_sentences) == 0:
            continue

        preprocessed_sentences = []
        for raw_sent in raw_sentences:
            preprocessed_sent = preprocess_raw_sent(raw_sent)
            preprocessed_sentences.append(preprocessed_sent)

        preprocessed_abs_sentences_list = []
        raw_abs_sent_list = abstract.split(' . ')
        for abs_sent in raw_abs_sent_list:
            preprocessed_abs_sent = preprocess_raw_sent(abs_sent)
            preprocessed_abs_sentences_list.append(preprocessed_abs_sent)    
        preprocessed_abs_sentences = (" ").join(preprocessed_abs_sentences_list)  

        if len(preprocessed_sentences) < 7 or len(preprocessed_abs_sentences_list) < 3:
            continue

        rougeforsentences = evaluate_rouge(raw_sentences,abstract)
          
        print("Done preprocessing!")
        
        print('time for processing', time.time() - start_time)
        if len(preprocessed_sent) < 4:
            NUM_PICKED_SENTS = len(preprocessed_sentences)
        else:
            NUM_PICKED_SENTS = 4
        # DONE!
        Solver = Summerizer(title, preprocessed_sentences, raw_sentences, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, NUM_PICKED_SENTS, rougeforsentences, abstract, order_params)
        best_individual = Solver.solve()
        file_name = os.path.join(save_path, example[1] )    

        print(file_name)
        if best_individual is None:
            solution_for_exception(rougeforsentences, raw_sentences, file_name)     
        else:
            print(best_individual)
            Solver.show(best_individual, file_name)
        
    
def multiprocess(num_process, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, stories, save_path):
    print("The number of processes %d" %(num_process))
    processes = []
    n = math.floor(len(stories)/5)
    set_of_docs = [stories[i:i + n] for i in range(0, len(stories), n)] 
    for index, sub_stories in enumerate(set_of_docs):
        p = multiprocessing.Process(target=start_run, args=(
            index, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE,sub_stories, save_path[index], 1))
        processes.append(p)
        p.start()      
    for p in processes:
        p.join()



def main():
    # Setting Variables
    POPU_SIZE = 40
    MAX_GEN = 100
    CROSS_RATE = 0.8
    MUTATE_RATE = 0.4

    directory = 'full_text_data'
    save_path=['hyp1', 'hyp2', 'hyp3', 'hyp4', 'hyp5']

    if not os.path.exists('hyp1'):
        os.makedirs('hyp1')
    if not os.path.exists('hyp2'):
        os.makedirs('hyp2')
    if not os.path.exists('hyp3'):
        os.makedirs('hyp3')
    if not os.path.exists('hyp4'):
        os.makedirs('hyp4')
    if not os.path.exists('hyp5'):
        os.makedirs('hyp5')


    print("Setting: ")
    print("POPULATION SIZE: {}".format(POPU_SIZE))
    print("MAX NUMBER OF GENERATIONS: {}".format(MAX_GEN))
    print("CROSSING RATE: {}".format(CROSS_RATE))
    print("MUTATION SIZE: {}".format(MUTATE_RATE))

    # list of documents
    stories = load_docs(directory)
    start_time = time.time()
    
    # multiprocess(5, POPU_SIZE, MAX_GEN, CROSS_RATE,
    #              MUTATE_RATE, stories, save_path)
    start_run(1, POPU_SIZE, MAX_GEN, CROSS_RATE, MUTATE_RATE, stories, save_path[0], 1)

    print("--- %s mins ---" % ((time.time() - start_time)/(60.0*len(stories))))

if __name__ == '__main__':
    main()  
        
        
     
    


    
    
    
    
        
            
            
         
