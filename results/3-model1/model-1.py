import pandas as pd
import numpy as np
from mip import *
import time

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

#prepare main dataframe and list of periods
print(time.time())
print('creating main_df')
main_df = prepare_df(recipes_df,period_number)
period_list = main_df['period'].unique()

#Create mipmodel, set solver
print(time.time())
print('creating mipmodel')
mipmodel = Model(sense = MAXIMIZE, solver_name = 'GRB')

#add binary decision variables, recipe x in period y or not
print(time.time())
print('creating decision variables')
dec_var = [mipmodel.add_var(name = main_df.iloc[row,8], var_type = BINARY) for row in main_df.index]

# add objective function profit * decvar
print(time.time())
print('creating objective function')
mipmodel.objective = maximize(xsum(list(main_df['profit'])[i] * dec_var[i] for i in main_df.index)/(30*len(period_list)))

#lists
unique_recipe_id_list = main_df['ix'].unique()
period_list = main_df['period'].unique()

#CONSTRAINTS##

# #Every recipe can appear at most given value
# for recipe in unique_recipe_id_list:
#     selected_recipes = main_df[main_df['ix'] == recipe]
#     mipmodel += sum(dec_var[i] for i in selected_recipes.index) <= 4

#CONS 1 : repeat all recipes on given interval:
repetition_interval = 2
for period in period_list[:-repetition_interval]:
    list_periods_selected = []
    for selected_period in range(period, period + repetition_interval + 1):
        list_periods_selected.append(selected_period)
    for recipe in unique_recipe_id_list:
        selected_recipes = main_df.query("ix == @recipe and period in @list_periods_selected")
        mipmodel += sum(dec_var[i] for i in selected_recipes.index) <= 1

#Assign 30 recipe per period
for period in period_list:
    selected_recipes = main_df[main_df['period'] == period]
    mipmodel += sum(dec_var[i] for i in selected_recipes.index) == 30

#Average calorie per period greater than value
for period in period_list:
    selected_recipes = main_df[main_df['period'] == period]
    mipmodel += xsum(list(main_df['calories'])[i]*dec_var[i] for i in selected_recipes.index)/30 >= 890

#Average protein per period greater than value
for period in period_list:
    selected_recipes = main_df[main_df['period'] == period]
    mipmodel += xsum(list(main_df['protein'])[i]*dec_var[i] for i in selected_recipes.index)/30 >= 55

#Average fat per period greater than value
for period in period_list:
    selected_recipes = main_df[main_df['period'] == period]
    mipmodel += xsum(list(main_df['fat'])[i]*dec_var[i] for i in selected_recipes.index)/30 >= 57

#Average rating per period greater than value
for period in period_list:
    selected_recipes = main_df[main_df['period'] == period]
    mipmodel += xsum(list(main_df['rating'])[i]*dec_var[i] for i in selected_recipes.index) >= 30*4.1

#No summer recipes allowed in winter
winter_period = [12,1,2]
selected_recipes = main_df.query("period in @winter_period")
selected_recipes = selected_recipes[selected_recipes['tags'].str.contains('summer')]
mipmodel += sum(dec_var[i] for i in selected_recipes.index) == 0

#No winter recipes allowed in summer
summer_period = [6,7,8]
selected_recipes = main_df.query("period in @summer_period")
selected_recipes = selected_recipes[selected_recipes['tags'].str.contains('winter')]
mipmodel += sum(dec_var[i] for i in selected_recipes.index) == 0

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
        mipmodel += sum(dec_var[i] for i in selected_recipes.index) <= 6

mipmodel.write('model.mps')
mipmodel.write('model.lp')

#OPTIMIZE#
mipmodel.max_mip_gap = 0.01
status = mipmodel.optimize(max_seconds=3600)

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






