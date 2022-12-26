import pandas as pd
import numpy as np
from mip import *
import time
import logging

logger = logging.getLogger('miplog')

recipes_df = pd.read_csv('cleaned_db.csv')
period_number = 12

def prepare_df (recipes_df,period_number):
    period_list = []
    for period in range (1,period_number + 1):
        period_list.append(period)
    period_df_list = []
    recipes_df['dec_var'] = ''
    recipes_df['period'] = ''

    for period in period_list:
        period_df = globals()['period'+str(period)]  = recipes_df.copy()
        period_df['period'] = period_df['period'].replace([''],period)
        period_df['dec_var'] = period_df['ix'].astype(str) + ',' + period_df['period'].astype(str)
        period_df_list.append(period_df)

    main_df = pd.concat(period_df_list)
    main_df = main_df.reset_index(drop = True)
    return(main_df)

def robust_variables(main_df,r,interval):
    main_df['zij'] = 'z' + main_df['dec_var']
    main_df['dijmax'] = (interval)*main_df['rating']

    list_wj = main_df['period'].unique()
    return(main_df)



#prepare main dataframe and list of periods
print(time.time())
print('creating main_df')
main_df = prepare_df(recipes_df,period_number)
main_df = robust_variables(main_df,1,0.05)
#lists
unique_recipe_id_list = main_df['ix'].unique()
period_list = main_df['period'].unique()


#Create mipmodel, set solver
print(time.time())
print('creating mipmodel')
mipmodel = Model(sense = MAXIMIZE, solver_name = 'GRB')

#add binary decision variables, recipe x in period y or not
print(time.time())
print('creating decision variables')
xij = [mipmodel.add_var(name=main_df.iloc[row, 8], var_type=BINARY) for row in main_df.index]
zij = [mipmodel.add_var(name='z' + str(main_df.iloc[row, 8]), var_type=CONTINUOUS) for row in main_df.index]
wj = [mipmodel.add_var(name='w' + str(item), var_type=CONTINUOUS) for item in period_list]
r = 5
n_assigned = 60


# add objective function profit * decvar
print(time.time())
print('creating objective function')
#max ∑ i∈I j∈J pij xij
mipmodel.objective = maximize(xsum(list(main_df['profit'])[i] * xij[i] for i in main_df.index)/(n_assigned*len(period_list)))

#CONSTRAINTS##

#∑j=j xij ≤ 1, ∀i ∈ I, ∀j ∈ J\{11, 12}
#CONS 1 : repeat all recipes on given interval:
repetition_interval = 2
for period in period_list[:-repetition_interval]:
    list_periods_selected = []
    for selected_period in range(period, period + repetition_interval + 1):
        list_periods_selected.append(selected_period)
    for recipe in unique_recipe_id_list:
        selected_recipes = main_df.query("ix == @recipe and period in @list_periods_selected")
        mipmodel += sum(xij[i] for i in selected_recipes.index) <= 1


#∑i∈Ixij = n, ∀j ∈ J
#Assign n recipe per period
for period in period_list:
    selected_recipes = main_df[main_df['period'] == period]
    mipmodel += sum(xij[i] for i in selected_recipes.index) == n_assigned

#∑i∈I xij × Caloriei n >= 1000, ∀j ∈ J
#Average calorie per period greater than value
for period in period_list:
    selected_recipes = main_df[main_df['period'] == period]
    mipmodel += xsum(list(main_df['calories'])[i]*xij[i] for i in selected_recipes.index)/n_assigned >= 890

#Average protein per period greater than value
for period in period_list:
    selected_recipes = main_df[main_df['period'] == period]
    mipmodel += xsum(list(main_df['protein'])[i]*xij[i] for i in selected_recipes.index)/n_assigned >= 55

#Average fat per period greater than value
for period in period_list:
    selected_recipes = main_df[main_df['period'] == period]
    mipmodel += xsum(list(main_df['fat'])[i]*xij[i] for i in selected_recipes.index)/n_assigned >= 57

#robust rating  4*n
for index_p, period in enumerate(period_list):
    selected_recipes = main_df[main_df['period'] == period]
    mipmodel += xsum(list(main_df['rating'])[i]*xij[i] for i in selected_recipes.index) - r*wj[index_p] - xsum(zij[i] for i in selected_recipes.index) >= n_assigned*4
    for index_r, recipe in selected_recipes.iterrows():
        mipmodel += wj[index_p] + zij[index_r] >= recipe['dijmax'] * xij[index_r]
for w in wj:
    mipmodel += w >= 0
for z in zij:
    mipmodel += z >= 0


#No summer recipes allowed in winter
winter_period = [12,1,2]
selected_recipes = main_df.query("period in @winter_period")
selected_recipes = selected_recipes[selected_recipes['tags'].str.contains('summer')]
mipmodel += sum(xij[i] for i in selected_recipes.index) == 0

#No winter recipes allowed in summer
summer_period = [6,7,8]
selected_recipes = main_df.query("period in @summer_period")
selected_recipes = selected_recipes[selected_recipes['tags'].str.contains('winter')]
mipmodel += sum(xij[i] for i in selected_recipes.index) == 0

#Limiting the appearance of  same tag per period
tags_df = main_df['tags']
list_tags = []
for row in tags_df.index:
    list_tags = list_tags + tags_df[row].split(',')
unique_list_tags = []
for item in list_tags:
    if item not in unique_list_tags and item != '':
        unique_list_tags.append(item)
for period in period_list:
    selected_period = main_df[main_df['period'] == period]
    for tag in unique_list_tags:
        selected_recipes = selected_period[selected_period['tags'].str.contains(tag)]
        mipmodel += sum(xij[i] for i in selected_recipes.index) <= 6

mipmodel.write('model.mps')
mipmodel.write('model.lp')

#OPTIMIZE#
mipmodel.max_mip_gap = 0.01
status = mipmodel.optimize(max_seconds=7200)

#RESULT#
if status == OptimizationStatus.OPTIMAL:
    print('optimal solution cost {} found'.format(mipmodel.objective_value))
elif status == OptimizationStatus.FEASIBLE:
    print('sol.cost {} found, best possible: {}'.format(mipmodel.objective_value, mipmodel.objective_bound))
elif status == OptimizationStatus.NO_SOLUTION_FOUND:
    print('no feasible solution found, lower bound is: {}'.format(mipmodel.objective_bound))
if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
    print('solution:')
    list_dec_var_solution = []
    for v in mipmodel.vars:
       if abs(v.x) > 1e-6: # only printing non-zeros
          # print('{} : {}'.format(v.name, v.x))
          list_dec_var_solution.append(v.name)
    result_df = main_df[main_df['dec_var'].isin(list_dec_var_solution)]
    result_df.to_csv('result')
    print(result_df)






