import pandas as pd
import numpy as np
from mip import *
import time
import logging

class meal_allocation():

    def __init__(self):
        self.logger = logging.getLogger('miplog')
        self.recipes_df = pd.read_csv('cleaned_db.csv')
        self.period_number = 6
        self.deviation_percentage = 0.05
        self.r = 5
        self.n_assigned = 30

        #prepare main dataframe and list of periods
        print(time.time())
        print('creating main_df')
        self.main_df = self.prepare_df(self.recipes_df,self.period_number)
        self.main_df = self.robust_variables(self.main_df,1,self.deviation_percentage)
        #lists
        self.unique_recipe_id_list = self.main_df['ix'].unique()
        self.period_list = self.main_df['period'].unique()
        #Create mipmodel, set solver
        print(time.time())
        print('creating mipmodel')
        self.mipmodel = Model(sense = MAXIMIZE, solver_name = 'GRB')

        #add binary decision variables, recipe x in period y or not
        print(time.time())
        print('creating decision variables')
        self.xij = [self.mipmodel.add_var(name=self.main_df.iloc[row, 8], var_type=BINARY) for row in self.main_df.index]

        # add objective function profit * decvar
        print(time.time())
        print('creating objective function')
        #max ∑ i∈I j∈J pij xij
        self.mipmodel.objective = maximize(xsum(list(self.main_df['profit'])[i] * self.xij[i] for i in self.main_df.index)/(self.n_assigned*len(self.period_list)))

        #CONSTRAINTS##
        self.constraint_1(2)
        self.constraint_2(self.n_assigned)
        self.constraint_3(890)
        self.constraint_4(55)
        self.constraint_5(57)
        self.constraint_6(4.2)
        #self.constraint_6_robust(3.95,self.r)
        # self.constraint_7()
        # self.constraint_8()
        self.constraint_9(6)

        self.mipmodel.write('model.mps')
        self.mipmodel.write('model.lp')

        #OPTIMIZE#
        self.mipmodel.max_mip_gap = 0.01
        status = self.mipmodel.optimize(max_seconds=7200)

        #RESULT#
        if status == OptimizationStatus.OPTIMAL:
            print('optimal solution cost {} found'.format(self.mipmodel.objective_value))
        elif status == OptimizationStatus.FEASIBLE:
            print('sol.cost {} found, best possible: {}'.format(self.mipmodel.objective_value, self.mipmodel.objective_bound))
        elif status == OptimizationStatus.NO_SOLUTION_FOUND:
            print('no feasible solution found, lower bound is: {}'.format(self.mipmodel.objective_bound))
        if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
            print('solution:')
            list_dec_var_start = []
            for v in self.mipmodel.vars:
               if abs(v.x) > 1e-6: # only printing non-zeros
                  # print('{} : {}'.format(v.name, v.x))
                  var_tuple = (v, v.x)
                  list_dec_var_start.append(var_tuple)
            self.warm_start(list_dec_var_start)

    def warm_start(self,list_dec_var_start):
        print(time.time())
        print('creating mipmodel')
        self.mipmodel = Model(sense = MAXIMIZE, solver_name = 'GRB')
        #add binary decision variables, recipe x in period y or not
        print(time.time())
        print('creating decision variables')
        self.xij = [self.mipmodel.add_var(name=self.main_df.iloc[row, 8], var_type=BINARY) for row in self.main_df.index]
        self.zij = [self.mipmodel.add_var(name='z' + str(self.main_df.iloc[row, 8]), var_type=CONTINUOUS) for row in self.main_df.index]
        self.wj = [self.mipmodel.add_var(name='w' + str(item), var_type=CONTINUOUS) for item in self.period_list]
        self.mipmodel.objective = maximize(xsum(list(self.main_df['profit'])[i] * self.xij[i] for i in self.main_df.index)/(self.n_assigned*len(self.period_list)))
        #CONSTRAINTS##
        self.constraint_1(2)
        self.constraint_2(self.n_assigned)
        self.constraint_3(890)
        self.constraint_4(55)
        self.constraint_5(57)
        #self.constraint_6(4)
        self.constraint_6_robust(3.90,self.r)
        # self.constraint_7()
        # self.constraint_8()
        self.constraint_9(6)
        self.mipmodel.write('model2.mps')
        self.mipmodel.write('model2.lp')
        #OPTIMIZE#
        self.mipmodel.max_mip_gap = 0.01
        self.mipmodel.start = list_dec_var_start
        status = self.mipmodel.optimize(max_seconds=7200)
        if status == OptimizationStatus.OPTIMAL:
            print('optimal solution cost {} found'.format(self.mipmodel.objective_value))
        elif status == OptimizationStatus.FEASIBLE:
            print('sol.cost {} found, best possible: {}'.format(self.mipmodel.objective_value, self.mipmodel.objective_bound))
        elif status == OptimizationStatus.NO_SOLUTION_FOUND:
            print('no feasible solution found, lower bound is: {}'.format(self.mipmodel.objective_bound))
        if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
            print('solution:')
            list_dec_var_solution = []
            for v in self.mipmodel.vars:
               if abs(v.x) > 1e-6: # only printing non-zeros
                  # print('{} : {}'.format(v.name, v.x))
                  list_dec_var_solution.append(v.name)
            result_df = self.main_df[self.main_df['dec_var'].isin(list_dec_var_solution)]
            result_df.to_csv('result')
            print(result_df)
    def prepare_df (self,recipes_df,period_number):
        self.period_list = []
        for period in range (1,period_number + 1):
            self.period_list.append(period)
        period_df_list = []
        recipes_df['dec_var'] = ''
        recipes_df['period'] = ''

        for period in self.period_list:
            period_df = globals()['period'+str(period)]  = recipes_df.copy()
            period_df['period'] = period_df['period'].replace([''],period)
            period_df['dec_var'] = period_df['ix'].astype(str) + ',' + period_df['period'].astype(str)
            period_df_list.append(period_df)

        main_df = pd.concat(period_df_list)
        main_df = main_df.reset_index(drop = True)
        return(main_df)
    def robust_variables(self,main_df,r,deviation_percentage):
        main_df['zij'] = 'z' + main_df['dec_var']
        main_df['dijmax'] = (deviation_percentage)*main_df['rating']
        list_wj = main_df['period'].unique()
        return(main_df)
    def constraint_1(self,repetition_interval):
        for period in self.period_list[:-repetition_interval]:
            list_periods_selected = []
            for selected_period in range(period, period + repetition_interval + 1):
                list_periods_selected.append(selected_period)
            for recipe in self.unique_recipe_id_list:
                selected_recipes = self.main_df.query("ix == @recipe and period in @list_periods_selected")
                self.mipmodel += sum(self.xij[i] for i in selected_recipes.index) <= 1
    def constraint_2(self,n_assigned):
        for period in self.period_list:
            selected_recipes = self.main_df[self.main_df['period'] == period]
            self.mipmodel += sum(self.xij[i] for i in selected_recipes.index) == n_assigned
    def constraint_3(self,target_calorie):
        for period in self.period_list:
            selected_recipes = self.main_df[self.main_df['period'] == period]
            self.mipmodel += xsum(list(self.main_df['calories'])[i] * self.xij[i] for i in selected_recipes.index) / self.n_assigned >= target_calorie
    def constraint_4(self,target_protein):
        for period in self.period_list:
            selected_recipes = self.main_df[self.main_df['period'] == period]
            self.mipmodel += xsum(list(self.main_df['protein'])[i] * self.xij[i] for i in selected_recipes.index) / self.n_assigned >= target_protein
    def constraint_5(self,target_fat):
        for period in self.period_list:
            selected_recipes = self.main_df[self.main_df['period'] == period]
            self.mipmodel += xsum(list(self.main_df['fat'])[i] * self.xij[i] for i in selected_recipes.index) / self.n_assigned >= target_fat
    def constraint_6(self,target_rating):
        for period in self.period_list:
            selected_recipes = self.main_df[self.main_df['period'] == period]
            self.mipmodel += xsum(list(self.main_df['rating'])[i] * self.xij[i] for i in selected_recipes.index) >= self.n_assigned * target_rating
    def constraint_6_robust(self,target_rating,r):
        for index_p, period in enumerate(self.period_list):
            selected_recipes = self.main_df[self.main_df['period'] == period]
            self.mipmodel += xsum(list(self.main_df['rating'])[i] * self.xij[i] for i in selected_recipes.index) - r * self.wj[index_p] - xsum(self.zij[i] for i in selected_recipes.index) >= self.n_assigned * 4
            for index_r, recipe in selected_recipes.iterrows():
                self.mipmodel += self.wj[index_p] + self.zij[index_r] >= recipe['dijmax'] * self.xij[index_r]
        for w in self.wj:
            self.mipmodel += w >= 0
        for z in self.zij:
            self.mipmodel += z >= 0
    def constraint_7(self):
        winter_period = [12, 1, 2]
        selected_recipes = self.main_df.query("period in @winter_period")
        selected_recipes = selected_recipes[selected_recipes['tags'].str.contains('summer')]
        self.mipmodel += sum(self.xij[i] for i in selected_recipes.index) == 0
    def constraint_8(self):
        summer_period = [6,7,8]
        selected_recipes = self.main_df.query("period in @summer_period")
        selected_recipes = selected_recipes[selected_recipes['tags'].str.contains('winter')]
        self.mipmodel += sum(self.xij[i] for i in selected_recipes.index) == 0
    def constraint_9(self,target_tags):
        tags_df = self.main_df['tags']
        list_tags = []
        for row in tags_df.index:
            list_tags = list_tags + tags_df[row].split(',')
        unique_list_tags = []
        for item in list_tags:
            if item not in unique_list_tags and item != '':
                unique_list_tags.append(item)
        for period in self.period_list:
            selected_period = self.main_df[self.main_df['period'] == period]
            for tag in unique_list_tags:
                selected_recipes = selected_period[selected_period['tags'].str.contains(tag)]
                self.mipmodel += sum(self.xij[i] for i in selected_recipes.index) <= target_tags

meal_allocation()






