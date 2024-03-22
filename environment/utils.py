import numpy as np
from deap import tools
from deap import base
from deap import creator
import pandas as pd
import plotly.express as px

from environment.mangoEnv import MangoEnv, MAX_AMOUNT
import environment.elitism as elitism


default_params = {
    'render_mode': None,
    'ext_mngo_price_mean': 0.1,
    'init_mngo_pool_balance': 1e6 / MAX_AMOUNT,
    'init_usdc_pool_balance': 1e5 / MAX_AMOUNT,
    'init_treasury_size_usdc': 150e6 / MAX_AMOUNT,
    'usdc_user_balance_init': 2e8 / MAX_AMOUNT,
    'mngo_collateral_factor': 1.5,
    'arb_efficiency_factor': 0.95,
    'seed': 12345
}

class BruteForcer:
    def __init__(self, max_lvl, params=None):
        if params is None:
            self.env = MangoEnv(**default_params)
        else:
            self.env = MangoEnv(**params)
        self.max_lvl = max_lvl
        self.children_num = self.env.action_space.n
        self.reset_stats()

    def reset_stats(self):
        self.stats = {
            'reward_count': 5 * [0],
            'seq_count_visited': 0,
            'seq_w_penalty_count': 0,
            'last_best_seq': [],
        }

    def _test_sequence(self, seq, seed=None):
        self.env.reset(seed=seed)
        done = False
        reward = 0
        overall_reward = 0
        w_penalty = False
        for s in seq:
            _, reward, terminated, truncated, _ = self.env.step(s)
            overall_reward += reward
            w_penalty = w_penalty or (reward < 0)
            done = terminated or truncated
            if done:
                break

        return done, reward, overall_reward, w_penalty
    
    def _test_root(self, root, halt_on_first_best, seed=None):
        best_found = False
        done, _, overall_reward, w_penalty = self._test_sequence(root, seed)

        if done or len(root) == self.max_lvl:
            if overall_reward == 0.05:
                self.stats['reward_count'][1] += 1
            elif overall_reward == 0.2:
                self.stats['reward_count'][2] += 1
            elif overall_reward == 2.0:
                self.stats['reward_count'][3] += 1
            elif overall_reward == 10.0:
                self.stats['reward_count'][4] += 1
                self.stats['last_best_seq'] = root[:]
                best_found = True
            else:
                self.stats['reward_count'][0] += 1
            self.stats['seq_count_visited'] += 1
            if w_penalty:
                self.stats['seq_w_penalty_count'] += 1           
            return best_found

        for ii in range(self.children_num):
            child_seq = root[:] + [ii]
            child_best_found = self._test_root(child_seq, halt_on_first_best, seed)
            best_found = best_found or child_best_found
            if best_found and halt_on_first_best:
                return best_found
            
        return best_found

    def test_tree(self, halt_on_first_best=False, seed=None):
        best_found = self._test_root([], halt_on_first_best=halt_on_first_best, seed=seed)
        return best_found


class GeneticOptimizer:
    def __init__(
            self,
            max_lvl=5,
            params=None
    ):
        if params is None:
            self.env = MangoEnv(**default_params)
        else:
            self.env = MangoEnv(**params)
        self.actions_num = self.env.action_space.n
        self.chromosome_len = max_lvl

        # define genetic algorithm
        self.toolbox = base.Toolbox()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        # create random individual 
        self.toolbox.register("actionSampler", self.env.action_space.sample)
        self.toolbox.register("individualCreator", tools.initRepeat, creator.Individual, 
                              self.toolbox.actionSampler, max_lvl)
        # create the population (list of individuals):
        self.toolbox.register("populationCreator", tools.initRepeat, list, self.toolbox.individualCreator)
        # Genetic operators:
        self.toolbox.register("select", tools.selTournament, tournsize=2)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=self.actions_num - 1,
                              indpb=1.0/self.chromosome_len)

    def test_sequence(self, seq, seed=None):
        self.env.reset(seed=seed)
        overall_reward = 0
        for s in seq:
            _, reward, terminated, truncated, _ = self.env.step(s)
            overall_reward += reward
            done = terminated or truncated
            if done:
                break

        return overall_reward, # return tuple! It is strictly necessary for eaSimpleWithElitism()    

    def run(
            self,
            population_size=300,
            max_generations=100,
            hall_of_fame_size=30,
            p_crossover=0.9,
            p_mutation=0.1,
            seed=None
    ):
        # register evaluation function with seed
        self.toolbox.register("evaluate", self.test_sequence, seed=seed)
        # create initial population (generation 0):
        population = self.toolbox.populationCreator(n=population_size)
        # prepare the statistics object:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)                                
        # define the hall-of-fame object:
        hof = tools.HallOfFame(hall_of_fame_size)
        # perform the Genetic Algorithm flow with hof feature added:
        population, logbook = elitism.eaSimpleWithElitism(population, self.toolbox, 
                                                          cxpb=p_crossover, mutpb=p_mutation,
                                                          ngen=max_generations, stats=stats, 
                                                          halloffame=hof, verbose=False)

        return hof, logbook
    
    def show(self, hof, logbook, plot_height=600):
        print("- Best solutions are:")
        for ii, el in enumerate(hof.items):
            print(ii, ": ", el.fitness.values[0], " -> ", el)

        # plot statistics:
        maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

        columns = ['Generation', 'Max Fitness', 'Average Fitness']
        gens = np.arange(1, len(maxFitnessValues)+1)
        data = np.vstack((gens, maxFitnessValues, meanFitnessValues))
        df = pd.DataFrame(data.T, columns=columns)

        fig = px.line(df, x='Generation', y=['Max Fitness', 'Average Fitness'], 
                      title="Min and Average fitness over Generations", height=plot_height)
        fig.show()
                