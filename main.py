import pandas as pd
import numpy as np
from mip import *

recipes_df = pd.read_csv('cleaned_db.csv')
period_list = [1,2,3]
period_df = recipes_df
period_df['dec_var'] = ''
period_df['period'] = ''
period1 = period_df.copy()
period2 = period_df.copy()
period3 = period_df.copy()

for row in period1.index:
    period1.iloc[row,8] = ('r'+str(period1.iloc[row,0])) + ',' + 'p1'
    period1.iloc[row,9] = 1
for row in period2.index:
    period2.iloc[row,8] = ('r'+str(period2.iloc[row,0])) + ',' + 'p2'
    period2.iloc[row, 9] = 2
for row in period3.index:
    period3.iloc[row,8] = ('r'+str(period3.iloc[row,0])) + ',' + 'p3'
    period3.iloc[row, 9] = 3

main_df = pd.concat([period1,period2,period3])
main_df = main_df.reset_index(drop = True)

mipmodel = Model(sense = MAXIMIZE, solver_name = 'CBC')

#add binary decision variables, recipe x in period y or not
dec_var = [mipmodel.add_var(name = main_df.iloc[row,8], var_type = BINARY) for row in main_df.index]

# add objective function profit * decvar
mipmodel.objective = maximize(xsum(list(main_df['profit'])[i] * dec_var[i] for i in main_df.index)/90)

##CONSTRAINTS##

#Every recipe can appear at most once
unique_recipe_id_list = main_df['ix'].unique()
unique_recipe_id_list = unique_recipe_id_list.tolist()
for recipe in unique_recipe_id_list:
    selected_recipes = main_df[main_df['ix'] == recipe]
    mipmodel += sum(dec_var[i] for i in selected_recipes.index) <= 1

#Assign 30 recipe per period
period_list = main_df['period'].unique()
for recipe in period_list:
    selected_recipes = main_df[main_df['period'] == recipe]
    mipmodel += sum(dec_var[i] for i in selected_recipes.index) == 30

#Average calorie per period greater than value
for recipe in period_list:
    selected_recipes = main_df[main_df['period'] == recipe]
    mipmodel += xsum(list(main_df['calories'])[i]*dec_var[i] for i in selected_recipes.index)/30 >= 1200

#Average protein per period greater than value
for recipe in period_list:
    selected_recipes = main_df[main_df['period'] == recipe]
    mipmodel += xsum(list(main_df['protein'])[i]*dec_var[i] for i in selected_recipes.index)/30 >= 80

#Average fat per period greater than value
for recipe in period_list:
    selected_recipes = main_df[main_df['period'] == recipe]
    mipmodel += xsum(list(main_df['fat'])[i]*dec_var[i] for i in selected_recipes.index)/30 >= 70

#Average rating per period greater than value
for recipe in period_list:
    selected_recipes = main_df[main_df['period'] == recipe]
    mipmodel += xsum(list(main_df['rating'])[i]*dec_var[i] for i in selected_recipes.index)/30 >= 4

#Limiting the appearance of  same tag per week
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
        mipmodel += sum(dec_var[i] for i in selected_recipes.index) <= 5

#OPTIMIZE#
mipmodel.max_gap = 0.01
status = mipmodel.optimize(max_seconds=100)

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






