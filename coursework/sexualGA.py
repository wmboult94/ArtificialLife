# Genetic algorithm with sexual selection, Python implementation
# Warren Boult, Sussex University, 2017
# Based upon work by Sanchez-Velazco and Bullinaria, https://www.cs.bham.ac.uk/~jxb/PUBS/NCI.pdf
import math
import time
import numpy as np
import random
import pprint
import matplotlib.pyplot as plt
triangle = np.random.triangular # Use for female / male age weighting
randa = np.random.rand
randval = np.random.random_sample

def selection(individuals, population, pop_fit):
	index1 =  int(np.floor(randval()*individuals))
	index2 =  int(np.floor(randval()*individuals))
	p1 = population[index1]
	p2 = population[index2]

	if pop_fit[index1] > pop_fit[index2]:
		winner = p1
		loser = p2
		winner_ind = index1
		loser_ind = index2
	else:
		winner = p2
		loser = p1
		winner_ind = index2
		loser_ind = index1

	return winner_ind, loser_ind

def femaleSelection(individuals, male_individual, population, female_attributes, male_fitness, W):
	index1 =  int(np.floor(randval()*individuals))
	index2 =  int(np.floor(randval()*individuals))
	p1 = population[index1]
	p2 = population[index2]
	p1_fit = femaleFitnessPubGA(female_attributes['children'][index1],male_fitness,female_attributes['age'][index1],female_attributes['fitness'][index1],W,male_individual)
	p2_fit = femaleFitnessPubGA(female_attributes['children'][index2],male_fitness,female_attributes['age'][index2],female_attributes['fitness'][index2],W,male_individual)

	if p1_fit > p2_fit:
		winner = p1
		loser = p2
		winner_ind = index1
		loser_ind = index2
	else:
		winner = p2
		loser = p1
		winner_ind = index2
		loser_ind = index1

	return winner_ind, loser_ind

# Calculate competitive fitness (ie quality of solution) of an individual
# TODO: Consider including an age element to male fitness
def maleFitnessPubGA(male_individual,W):
	return np.matmul( np.matmul( male_individual, W ), male_individual.T )

# Calculate fitness of female individual
def femaleFitnessPubGA(female_children,male_fitness,female_age,female_fit,W,male_individual):
	w1 = 0.8
	w2 = 0.4
	w3 = 0.2
	comp_fit = female_fit

	male_child_index = female_children[0]
	if male_child_index != -1:
		male_child_fit = male_fitness[male_child_index]
		delta_fit = male_child_fit - maleFitnessPubGA(male_individual,W)
	else:
		delta_fit = 0

	g_age = ageFunction(4,2,female_age)

	return (w1*comp_fit + w2*delta_fit + w3*g_age) / w1 + w2 + w3

def interactionMatrix(genes):
		W = np.floor(randa(genes,genes)+0.5);
		# Bit of a hacky method, set all 0s to -1s
		W[W==0] = -1;
		# Make into diagonal matrix
		i_lower = np.tril_indices(genes, -1)
		W[i_lower] = W.T[i_lower]
		# Set diagonal to 0
		np.fill_diagonal(W,0)
		return W

# TODO: Implement the knapsack problem
# Ref: http://www.sc.ehu.es/ccwbayes/docencia/kzmm/files/AG-knapsack.pdf
# Use inner products to calculate 'benefit' and 'volume' constraint
# If volume exceeds limit, use item_inds = np.where(individual==1) to generate
# indices of items in a given knapsack, then choose a random index with
# ind = int(np.floor(randval()*len(item_inds))); item_ind = item_inds[ind]
# Flip bit of item in individual ==> individual[item_ind] = 0
# Recalculate 'benefit' and 'volume', repeat above if still above capacity constraint
def maleFitnessKnapsack(individual, items, capacity):
	fitness = np.inner(individual,items['benefit'])
	volume = np.inner(individual,items['volume'])
	# Ensure a valid individual is produced
	while volume > capacity:
		item_inds = np.where(individual==1)[0]
		random_ind = int(np.floor(randval()*len(item_inds)))
		item_ind = item_inds[random_ind]
		individual[item_ind] = 0
		fitness = np.inner(individual,items['benefit'])
		volume = np.inner(individual,items['volume'])
	return individual, fitness

def femaleFitnessKnapsack(female_children,male_fitnesses,female_age,female_fit,items,capacity,male_fitness):
	w1 = 0.8
	w2 = 0.3
	w3 = 0.3
	comp_fit = female_fit

	male_child_index = female_children[0]
	if male_child_index != -1:
		male_child_fit = male_fitnesses[male_child_index]
		delta_fit = male_child_fit - male_fitness
	else:
		delta_fit = 0

	g_age = ageFunction(4,2,female_age)

	return (w1*comp_fit + w2*delta_fit + w3*g_age) / w1 + w2 + w3

def femaleSelectionKnapsack(individuals, male_fitness, population, female_attributes, male_fitnesses, items, capacity):
	index1 =  int(np.floor(randval()*individuals))
	index2 =  int(np.floor(randval()*individuals))
	p1 = population[index1]
	p2 = population[index2]
	p1_fit = femaleFitnessKnapsack(female_attributes['children'][index1],male_fitnesses,female_attributes['age'][index1],female_attributes['fitness'][index1],items,capacity,male_fitness)
	p2_fit = femaleFitnessKnapsack(female_attributes['children'][index2],male_fitnesses,female_attributes['age'][index2],female_attributes['fitness'][index2],items,capacity,male_fitness)

	if p1_fit > p2_fit:
		winner = p1
		loser = p2
		winner_ind = index1
		loser_ind = index2
	else:
		winner = p2
		loser = p1
		winner_ind = index2
		loser_ind = index1

	return winner_ind, loser_ind

# Triangular function to give age score
def ageFunction(width,max_fert,age):
	if age < max_fert + width:
		return 1 - (math.fabs(age - max_fert)/width)
	else:
		return 0

# Checks if either of the individuals to mate is a parent of the other
def isParent(male_index, female_index, male_parents, female_parents):
	if male_parents[male_index][1] == female_index or female_parents[female_index][1] == male_index:
		return True
	else:
		return False

def minSexualGAKnapsack(individuals,genes,generations,male_mutation,female_mutation,crossover,repetitions,incest=1):

	benefits = np.random.randint(1,30,size=genes)
	volumes = np.random.randint(1,30,size=genes)
	keys = ['benefit','volume']
	items = dict(zip(keys,[benefits,volumes]))
	capacity = 80

	avg_fit = []
	avg_num_genos = []
	many_run_best_fit = np.zeros(generations)
	many_run_pop_avg_fit = np.zeros(generations)
	print '\n\nMinimal Sexual GA, knapsack problem in progress..'
	for n in range(repetitions):
		print 'Run ', n+1

		# Make random set of genotypes, alleles are 0 and 1 in simple bit string approach
		male_population = np.floor(randa(individuals/2,genes)+0.5)
		female_population = np.floor(randa(individuals/2,genes)+0.5)
		# print male_population[0]
		# population = male_population + female_population

		keys = ['fitness','parents','children','age']
		male_keys = ['fitness','parents']
		# initialise fitness
		male_pop_fit = np.zeros(individuals/2)
		female_pop_fit = np.zeros(individuals/2)
		# Build array of parent indices for each gender
		male_parents = map(tuple,np.full((individuals/2,2),-1,dtype=int))
		female_parents = map(tuple,np.full((individuals/2,2),-1,dtype=int))
		# Build array of children indices for each gender -- dont really need for males, but could be useful to have along the line !
		# male_children = map(tuple,np.full((individuals/2,2),-1,dtype=int))
		female_children = map(tuple,np.full((individuals/2,2),-1,dtype=int))
		# Initialise ages
		# male_ages = np.zeros(individuals/2)
		female_ages = np.zeros(individuals/2)
		# Build dictionaries of attributes for each gender
		male_attributes = dict(zip(male_keys,[male_pop_fit,male_parents]))
		female_attributes = dict(zip(keys,[female_pop_fit,female_parents,female_children,female_ages]))
		for j in range(individuals/2):
			# pop_fit[j] = fitness(population[j])
			# pop_fit[j] = population[j] * W * population[j].T
			male_population[j],male_attributes['fitness'][j] = maleFitnessKnapsack(male_population[j],items,capacity)
			female_population[j],female_attributes['fitness'][j] = maleFitnessKnapsack(female_population[j],items,capacity)

		# print male_attributes['fitness']
		best_fit = np.zeros(generations)
		pop_avg_fit = np.zeros(generations)
		for k in range(generations):
			for l in range(individuals):

				# Gendered tournament selection
				# Take two males, compare fitness, retain winner, retain index of losing indivdidual
				# Compute female fitness based on winning male individual
				# Select two females, compare fitness, retain winner, retain index of loser
				male_winner_ind, male_loser_ind = selection(individuals/2, male_population, male_attributes['fitness'])
				female_winner_ind, female_loser_ind = femaleSelectionKnapsack(individuals/2, male_attributes['fitness'][male_winner_ind], female_population, female_attributes, male_attributes['fitness'], items, capacity)
				if incest:
					while isParent(male_winner_ind,female_winner_ind,male_parents,female_parents):
						male_winner_ind, male_loser_ind = selection(individuals/2, male_population, male_attributes['fitness'])
						female_winner_ind, female_loser_ind = femaleSelectionKnapsack(individuals/2, male_attributes['fitness'][male_winner_ind], female_population, female_attributes, male_attributes['fitness'], items, capacity)

				# Male crossover, select random crossover point, copy mother material past this point
				# Copy into the losing individuals of each gender
				cross_male = int(np.floor(randval()*genes))
				male_population[male_loser_ind][cross_male:] = male_population[male_winner_ind][cross_male:]
				male_population[male_loser_ind][:cross_male] = female_population[female_winner_ind][:cross_male]

				# Female crossover, select random crossover point, copy reverse of mother material up to this point
				# copy rest normally
				cross_female = int(np.floor(randval()*genes))
				female_population[female_loser_ind][cross_female:] = female_population[female_winner_ind][cross_female:]
				female_population[female_loser_ind][:cross_female] = female_population[female_winner_ind][:cross_female][::-1]

				# Set parents
				male_attributes['parents'][male_loser_ind] = (male_winner_ind,female_winner_ind)
				female_attributes['parents'][female_loser_ind] = (male_winner_ind,female_winner_ind)

				# Set children
				# male_attributes['children'][male_winner_ind] = (male_loser_ind,female_loser_ind)
				female_attributes['children'][female_winner_ind] = (male_loser_ind,female_loser_ind)

				# Perform mutation
				for locus in range(genes):
					if randval() < male_mutation:
						male_population[male_loser_ind,locus] = 1 - male_population[male_loser_ind,locus]
					if randval() < female_mutation:
						female_population[female_loser_ind,locus] = 1 - female_population[female_loser_ind,locus]

				# Calculate new genotypes' fitness, add best fit of generation to bestFit
				# pop_fit[loser_ind] = fitness(populaion[loser_ind])
				# pop_fit[loser_ind] = population[loser_ind] * W * population[loser_ind].T
				male_population[male_loser_ind],male_attributes['fitness'][male_loser_ind] = maleFitnessKnapsack(male_population[male_loser_ind],items,capacity)
				female_population[female_loser_ind],female_attributes['fitness'][female_loser_ind] = maleFitnessKnapsack(female_population[female_loser_ind],items,capacity)
				# Set ages of children to 0
				# male_attributes['age'][male_loser_ind] = 0
				female_attributes['age'][female_loser_ind] = 0

			# Increment all ages each generation
			# male_attributes['age'] = [val+1 for val in male_attributes['age']]
			female_attributes['age'] = [val+1 for val in female_attributes['age']]

			# Set best_fit and best_geno
			male_max_ind = np.argmax(male_attributes['fitness'])
			female_max_ind = np.argmax(female_attributes['fitness'])
			if male_attributes['fitness'][male_max_ind] > female_attributes['fitness'][female_max_ind]:
				best_fit[k] = male_attributes['fitness'][male_max_ind]
				best_geno = male_population[male_max_ind]
			else:
				best_fit[k] = female_attributes['fitness'][female_max_ind]
				best_geno = female_population[female_max_ind]

			# Set average fitness of population
			pop_avg_fit[k] = ( sum(male_attributes['fitness']) + sum(female_attributes['fitness']) ) / individuals

		final_fit = best_fit[-1]
		avg_fit.append(final_fit)
		# meanfit_n[i] = final_fit

		# Add best_fit values and pop_avg_fit values
		many_run_best_fit += best_fit
		many_run_pop_avg_fit += pop_avg_fit

		# print '\n\nbest_fit'
		# print best_fit
		# ax.plot(best_fit)
		# print 'Final fitness score: ', final_fit

		maxf = max(male_attributes['fitness'])
		fmaxf = max(female_attributes['fitness'])
		popGT100 = np.array(male_population[np.where(maxf - male_attributes['fitness'] < 50)])
		fpopGT100 = np.array(female_population[np.where(fmaxf - female_attributes['fitness'] < 50)])
		# print popGT100
		mbestgenos = np.unique(popGT100, axis=0)
		fbestgenos = np.unique(fpopGT100, axis=0)
		num_good = len(mbestgenos) + len(fbestgenos)
		avg_num_genos.append(num_good)
		# print 'Num good genotypes found: ', num_good
	print 'Best fitness sexual GA on knapsack problem: ', max(avg_fit)
	std_fit = np.std(avg_fit)
	avg_fit = sum(avg_fit) / repetitions
	std_genos = np.std(avg_num_genos)
	avg_num_genos = sum(avg_num_genos) / repetitions
	print 'Average fitness score sexual GA: ', avg_fit
	print '  Standard deviation: +/-', round(std_fit,5)
	print 'Average num good genos found sexual GA on knapsack problem: ', avg_num_genos
	print '  Standard deviation: +/-', round(std_genos,5)

	many_run_best_fit = many_run_best_fit / repetitions
	many_run_pop_avg_fit = many_run_pop_avg_fit / repetitions
	return many_run_best_fit, many_run_pop_avg_fit

def regSexualGAKnapsack(individuals,genes,generations,male_mutation,female_mutation,crossover,repetitions):

	benefits = np.random.randint(1,30,size=genes)
	volumes = np.random.randint(1,30,size=genes)
	keys = ['benefit','volume']
	items = dict(zip(keys,[benefits,volumes]))
	capacity = 80

	avg_fit = []
	avg_num_genos = []
	many_run_best_fit = np.zeros(generations)
	many_run_pop_avg_fit = np.zeros(generations)
	print '\n\nRegular Sexual GA, knapsack problem in progress..'
	for n in range(repetitions):
		print 'Run ', n+1

		# Make random set of genotypes, alleles are 0 and 1 in simple bit string approach
		male_population = np.floor(randa(individuals/2,genes)+0.5)
		female_population = np.floor(randa(individuals/2,genes)+0.5)
		# print male_population[0]
		# population = male_population + female_population

		keys = ['fitness','children','age']
		male_keys = ['fitness']
		# initialise fitness
		male_pop_fit = np.zeros(individuals/2)
		female_pop_fit = np.zeros(individuals/2)
		# Build array of children indices for females
		female_children = map(tuple,np.full((individuals/2,2),-1,dtype=int))
		# Initialise ages
		female_ages = np.zeros(individuals/2)
		# Build dictionaries of attributes for each gender
		male_attributes = dict(zip(male_keys,[male_pop_fit]))
		female_attributes = dict(zip(keys,[female_pop_fit,female_children,female_ages]))
		for j in range(individuals/2):
			# pop_fit[j] = fitness(population[j])
			# pop_fit[j] = population[j] * W * population[j].T
			male_population[j],male_attributes['fitness'][j] = maleFitnessKnapsack(male_population[j],items,capacity)
			female_population[j],female_attributes['fitness'][j] = maleFitnessKnapsack(female_population[j],items,capacity)

		# print male_attributes['fitness']
		best_fit = np.zeros(generations)
		pop_avg_fit = np.zeros(generations)
		for k in range(generations):
			for l in range(individuals):

				# Roulette wheel selection
				# Select male first, then female
				# print male_attributes['fitness']
				max_fit = sum(male_attributes['fitness'])
				pick = random.uniform(0, max_fit)
				current = 0
				for index,fit in enumerate(male_attributes['fitness']):
					current += fit
					if current > pick:
						male_winner_ind = index
				male_loser_ind = np.argmin(male_attributes['fitness'])

				female_coop_fit = []
				for i,individual in enumerate(female_population):
					female_coop_fit.append(femaleFitnessKnapsack(female_attributes['children'][i],male_attributes['fitness'],female_attributes['age'][i],female_attributes['fitness'][i],items,capacity,male_attributes['fitness'][male_winner_ind]))
				max_fit = sum(female_coop_fit)
				pick = random.uniform(0, max_fit)
				current = 0
				for index,fit in enumerate(female_coop_fit):
					current += fit
					if current > pick:
						female_winner_ind = index
				female_loser_ind = np.argmin(female_coop_fit)

				# Male crossover, select random crossover point, copy mother material past this point
				# Copy into the losing individuals of each gender
				cross_male = int(np.floor(randval()*genes))
				male_population[male_loser_ind][cross_male:] = male_population[male_winner_ind][cross_male:]
				male_population[male_loser_ind][:cross_male] = female_population[female_winner_ind][:cross_male]

				# Female crossover, select random crossover point, copy reverse of mother material up to this point
				# copy rest normally
				cross_female = int(np.floor(randval()*genes))
				female_population[female_loser_ind][cross_female:] = female_population[female_winner_ind][cross_female:]
				female_population[female_loser_ind][:cross_female] = female_population[female_winner_ind][:cross_female][::-1]

				# # Set parents
				# male_attributes['parents'][male_loser_ind] = (male_winner_ind,female_winner_ind)
				# female_attributes['parents'][female_loser_ind] = (male_winner_ind,female_winner_ind)

				# Set children
				# male_attributes['children'][male_winner_ind] = (male_loser_ind,female_loser_ind)
				female_attributes['children'][female_winner_ind] = (male_loser_ind,female_loser_ind)

				# Perform mutation
				for locus in range(genes):
					if randval() < male_mutation:
						male_population[male_loser_ind,locus] = 1 - male_population[male_loser_ind,locus]
					if randval() < female_mutation:
						female_population[female_loser_ind,locus] = 1 - female_population[female_loser_ind,locus]

				# Calculate new genotypes' fitness, add best fit of generation to bestFit
				# pop_fit[loser_ind] = fitness(populaion[loser_ind])
				# pop_fit[loser_ind] = population[loser_ind] * W * population[loser_ind].T
				male_population[male_loser_ind],male_attributes['fitness'][male_loser_ind] = maleFitnessKnapsack(male_population[male_loser_ind],items,capacity)
				female_population[female_loser_ind],female_attributes['fitness'][female_loser_ind] = maleFitnessKnapsack(female_population[female_loser_ind],items,capacity)
				# Set ages of children to 0
				# male_attributes['age'][male_loser_ind] = 0
				female_attributes['age'][female_loser_ind] = 0

			# Increment all ages each generation
			# male_attributes['age'] = [val+1 for val in male_attributes['age']]
			female_attributes['age'] = [val+1 for val in female_attributes['age']]

			# Set best_fit and best_geno
			male_max_ind = np.argmax(male_attributes['fitness'])
			female_max_ind = np.argmax(female_attributes['fitness'])
			if male_attributes['fitness'][male_max_ind] > female_attributes['fitness'][female_max_ind]:
				best_fit[k] = male_attributes['fitness'][male_max_ind]
				best_geno = male_population[male_max_ind]
			else:
				best_fit[k] = female_attributes['fitness'][female_max_ind]
				best_geno = female_population[female_max_ind]

			# Set average fitness of population
			pop_avg_fit[k] = ( sum(male_attributes['fitness']) + sum(female_attributes['fitness']) ) / individuals

		final_fit = best_fit[-1]
		avg_fit.append(final_fit)
		# meanfit_n[i] = final_fit

		# Add best_fit values and pop_avg_fit values
		many_run_best_fit += best_fit
		many_run_pop_avg_fit += pop_avg_fit

		# print '\n\nbest_fit'
		# print best_fit
		# ax.plot(best_fit)
		# print 'Final fitness score: ', final_fit

		maxf = max(male_attributes['fitness'])
		fmaxf = max(female_attributes['fitness'])
		popGT100 = np.array(male_population[np.where(maxf - male_attributes['fitness'] < 50)])
		fpopGT100 = np.array(female_population[np.where(fmaxf - female_attributes['fitness'] < 50)])
		# print popGT100
		mbestgenos = np.unique(popGT100, axis=0)
		fbestgenos = np.unique(fpopGT100, axis=0)
		num_good = len(mbestgenos) + len(fbestgenos)
		avg_num_genos.append(num_good)
		# print 'Num good genotypes found: ', num_good
	print 'Best fitness sexual GA on knapsack problem: ', max(avg_fit)
	std_fit = np.std(avg_fit)
	avg_fit = sum(avg_fit) / repetitions
	std_genos = np.std(avg_num_genos)
	avg_num_genos = sum(avg_num_genos) / repetitions
	print 'Average fitness score sexual GA: ', avg_fit
	print '  Standard deviation: +/-', round(std_fit,5)
	print 'Average num good genos found sexual GA on knapsack problem: ', avg_num_genos
	print '  Standard deviation: +/-', round(std_genos,5)

	many_run_best_fit = many_run_best_fit / repetitions
	many_run_pop_avg_fit = many_run_pop_avg_fit / repetitions
	return many_run_best_fit, many_run_pop_avg_fit

def microbialGAKnapsack(individuals,genes,generations,mutation,crossover,repetitions):

	benefits = np.random.randint(1,30,size=genes)
	volumes = np.random.randint(1,30,size=genes)
	keys = ['benefit','volume']
	items = dict(zip(keys,[benefits,volumes]))
	capacity = 80

	avg_fit = []
	avg_num_genos = []
	many_run_best_fit = np.zeros(generations)
	many_run_pop_avg_fit = np.zeros(generations)
	print '\n\nMicrobial GA, knapsack problem in progress..'
	for n in range(repetitions):
		print 'Run ', n+1

		# Make random set of genotypes, alleles are 0 and 1 in simple bit string approach
		population = np.floor(randa(individuals,genes)+0.5)
		# Evaluate initial fitness
		pop_fit = np.zeros(individuals)
		for j in range(individuals):
			# pop_fit[j] = fitness(population[j])
			# pop_fit[j] = population[j] * W * population[j].T
			population[j],pop_fit[j] = maleFitnessKnapsack(population[j],items,capacity)
		# print pop_fit
		best_fit = np.zeros(generations)
		pop_avg_fit = np.zeros(generations)
		for k in range(generations):
			for l in range(individuals):

				# Gendered tournament selection
				# Take two males, compare fitness, retain winner, retain index of losing indivdidual
				# Compute female fitness based on winning male individual
				# Select two females, compare fitness, retain winner, retain index of loser
				# Perform crossover and mutation, produce two children, one of each gender
				# Replace the losers of each gender with the children
				winner_ind, loser_ind = selection(individuals,population,pop_fit)

				# Perform crossover and mutation
				for locus in range(genes):
					if randval() < crossover:
						# print 'crossover at locus ', locus
						population[loser_ind,locus] = population[winner_ind,locus]
					if randval() < mutation:
						population[loser_ind,locus] = 1 - population[loser_ind,locus] # turn 1 to 0, and vice versa

				# Calculate new genotype's fitness, add best fit of generation to bestFit
				# pop_fit[loser_ind] = fitness(populaion[loser_ind])
				# pop_fit[loser_ind] = population[loser_ind] * W * population[loser_ind].T
				population[loser_ind],pop_fit[loser_ind] = maleFitnessKnapsack(population[loser_ind],items,capacity)

			max_ind = np.argmax(pop_fit)
			# print pop_fit
			best_fit[k] = pop_fit[max_ind]
			best_geno = population[max_ind]

			# Set average fitness of population
			pop_avg_fit[k] = sum(pop_fit) / individuals

		final_fit = best_fit[-1]
		avg_fit.append(final_fit)

		# Add best_fit values and pop_avg_fit values
		many_run_best_fit += best_fit
		many_run_pop_avg_fit += pop_avg_fit

		# print '\n\nbest_fit'
		# print best_fit
		# ax.plot(best_fit,label='Single run')
		# print 'Final fitness score: ', final_fit

		maxf = max(pop_fit)
		popGT100 = population[np.where(maxf - pop_fit < 50)]
		# print popGT100
		bestgenos = np.unique(popGT100, axis=0)
		num_good = len(bestgenos)
		avg_num_genos.append(num_good)
		# print 'Num good genotypes found: ', num_good
	print 'Best fitness microbial GA on knapsack problem: ', max(avg_fit)
	std_fit = np.std(avg_fit)
	avg_fit = sum(avg_fit) / repetitions
	std_genos = np.std(avg_num_genos)
	avg_num_genos = sum(avg_num_genos) / repetitions
	print 'Average fitness microbial GA on knapsack problem: ', avg_fit
	print '  Standard deviation: +/-', round(std_fit,5)
	print 'Average num good genos found microbial GA on knapsack problem: ', avg_num_genos
	print '  Standard deviation: +/-', round(std_genos,5)

	many_run_best_fit = many_run_best_fit / repetitions
	many_run_pop_avg_fit = many_run_pop_avg_fit / repetitions
	# print many_run_pop_avg_fit
	return many_run_best_fit, many_run_pop_avg_fit

def minSexualGAPub(individuals,genes,generations,male_mutation,female_mutation,crossover,repetitions):

	avg_fit = []
	avg_num_genos = []
	many_run_best_fit = np.zeros(generations)
	many_run_pop_avg_fit = np.zeros(generations)
	print '\n\nMinimal Sexual GA in progress..'
	for n in range(repetitions):
		print 'Run ', n+1

		# Make random set of genotypes, alleles are 0 and 1 in simple bit string approach
		male_population = np.floor(randa(individuals/2,genes)+0.5)
		male_population[male_population==0] = -1;
		female_population = np.floor(randa(individuals/2,genes)+0.5)
		female_population[female_population==0] = -1;
		# print male_population[0]
		# population = male_population + female_population

		# Build randomly initialised interaction matrix of 1s and 0s
		W = interactionMatrix(genes)

		keys = ['fitness','parents','children','age']
		# initialise fitness
		male_pop_fit = np.zeros(individuals/2)
		female_pop_fit = np.zeros(individuals/2)
		# Build array of parent indices for each gender
		male_parents = map(tuple,np.full((individuals/2,2),-1,dtype=int))
		female_parents = map(tuple,np.full((individuals/2,2),-1,dtype=int))
		# Build array of children indices for each gender -- dont really need for males, but could be useful to have along the line !
		male_children = map(tuple,np.full((individuals/2,2),-1,dtype=int))
		female_children = map(tuple,np.full((individuals/2,2),-1,dtype=int))
		# Initialise ages
		male_ages = np.zeros(individuals/2)
		female_ages = np.zeros(individuals/2)
		# Build dictionaries of attributes for each gender
		male_attributes = dict(zip(keys,[male_pop_fit,male_parents,male_children,male_ages]))
		female_attributes = dict(zip(keys,[female_pop_fit,female_parents,female_children,female_ages]))
		for j in range(individuals/2):
			# pop_fit[j] = fitness(population[j])
			# pop_fit[j] = population[j] * W * population[j].T
			male_attributes['fitness'][j] = maleFitnessPubGA(male_population[j],W)
			female_attributes['fitness'][j] = maleFitnessPubGA(female_population[j],W)

		# print male_attributes['fitness']
		best_fit = np.zeros(generations)
		pop_avg_fit = np.zeros(generations)
		for k in range(generations):
			for l in range(individuals/2):

				# Gendered tournament selection
				# Try to avoid incest between parents and children
				# Take two males, compare fitness, retain winner, retain index of losing indivdidual
				# Compute female fitness based on winning male individual
				# Select two females, compare fitness, retain winner, retain index of loser
				# Perform crossover and mutation, produce two children, one of each gender
				# Replace the losers of each gender with the children

				male_winner_ind, male_loser_ind = selection(individuals/2, male_population, male_attributes['fitness'])
				female_winner_ind, female_loser_ind = femaleSelection(individuals/2, male_population[male_winner_ind], female_population, female_attributes, male_attributes['fitness'], W)
				while isParent(male_winner_ind,female_winner_ind,male_parents,female_parents):
					male_winner_ind, male_loser_ind = selection(individuals/2, male_population, male_attributes['fitness'])
					female_winner_ind, female_loser_ind = femaleSelection(individuals/2, male_population[male_winner_ind], female_population, female_attributes, male_attributes['fitness'], W)

				# Male crossover, select random crossover point, copy mother material past this point
				# Copy into the losing individuals of each gender
				cross_male = int(np.floor(randval()*genes))
				male_population[male_loser_ind][:cross_male] = male_population[male_winner_ind][:cross_male]
				male_population[male_loser_ind][cross_male:] = female_population[female_winner_ind][cross_male:]

				# Female crossover, select random crossover point, copy reverse of mother material up to this point
				# copy rest normally
				cross_female = int(np.floor(randval()*genes))
				female_population[female_loser_ind][:cross_female] = female_population[female_winner_ind][:cross_female][::-1]
				female_population[female_loser_ind][cross_female:] = female_population[female_winner_ind][cross_female:]

				# Set parents
				male_attributes['parents'][male_loser_ind] = (male_winner_ind,female_winner_ind)
				female_attributes['parents'][female_loser_ind] = (male_winner_ind,female_winner_ind)

				# Set children
				male_attributes['children'][male_winner_ind] = (male_loser_ind,female_loser_ind)
				female_attributes['children'][female_winner_ind] = (male_loser_ind,female_loser_ind)

				# Perform mutation
				for locus in range(genes):
					if randval() < male_mutation:
						male_population[male_loser_ind,locus] = - male_population[male_loser_ind,locus]
					if randval() < female_mutation:
						female_population[female_loser_ind,locus] = - female_population[female_loser_ind,locus]

				# Calculate new genotypes' fitness, add best fit of generation to bestFit
				# pop_fit[loser_ind] = fitness(populaion[loser_ind])
				# pop_fit[loser_ind] = population[loser_ind] * W * population[loser_ind].T
				male_attributes['fitness'][male_loser_ind] = maleFitnessPubGA(male_population[male_loser_ind],W)
				female_attributes['fitness'][female_loser_ind] = maleFitnessPubGA(female_population[female_loser_ind],W)
				# Set ages of children to 0
				male_attributes['age'][male_loser_ind] = 0
				female_attributes['age'][female_loser_ind] = 0

			# Increment all ages each generation
			male_attributes['age'] = [val+1 for val in male_attributes['age']]
			female_attributes['age'] = [val+1 for val in female_attributes['age']]

			# Set best_fit and best_geno
			male_max_ind = np.argmax(male_attributes['fitness'])
			female_max_ind = np.argmax(female_attributes['fitness'])
			if male_attributes['fitness'][male_max_ind] > female_attributes['fitness'][female_max_ind]:
				best_fit[k] = male_attributes['fitness'][male_max_ind]
				best_geno = male_population[male_max_ind]
			else:
				best_fit[k] = female_attributes['fitness'][female_max_ind]
				best_geno = female_population[female_max_ind]

			# Set average fitness of population
			pop_avg_fit[k] = ( sum(male_attributes['fitness']) + sum(female_attributes['fitness']) ) / individuals

		final_fit = best_fit[-1]
		avg_fit.append(final_fit)

		# Add best_fit values and pop_avg_fit values
		many_run_best_fit += best_fit
		many_run_pop_avg_fit += pop_avg_fit

		# ax.plot(best_fit)
		# print 'Final fitness score: ', final_fit

		maxf = max(male_attributes['fitness'])
		fmaxf = max(female_attributes['fitness'])
		popGT100 = np.array(male_population[np.where(maxf - male_attributes['fitness'] < 50)])
		fpopGT100 = np.array(female_population[np.where(fmaxf - female_attributes['fitness'] < 50)])
		mbestgenos = np.unique(popGT100, axis=0)
		fbestgenos = np.unique(fpopGT100, axis=0)
		num_good = len(mbestgenos) + len(fbestgenos)
		avg_num_genos.append(num_good)
		# print popGT100
		# print np.unique(popGT100, axis=0)
		# print np.unique(fpopGT100, axis=0)

	print 'Best fitness minimal sexual GA on pub problem: ', max(avg_fit)
	avg_fit = sum(avg_fit) / repetitions
	avg_num_genos = sum(avg_num_genos) / repetitions
	print 'Average fitness minimal sexual GA on pub problem: ', avg_fit
	print 'Average num good genos found minimal sexual GA on pub problem: ', avg_num_genos

	many_run_best_fit = many_run_best_fit / repetitions
	many_run_pop_avg_fit = many_run_pop_avg_fit / repetitions
	return many_run_best_fit, many_run_pop_avg_fit

def regSexualGAPub(individuals,genes,generations,male_mutation,female_mutation,crossover,repetitions):

	fig, ax = plt.subplots()

	avg_fit = []
	avg_num_genos = []
	many_run_best_fit = np.zeros(generations)
	many_run_pop_avg_fit = np.zeros(generations)
	print '\n\nRoulette Sexual GA in progress..'
	for n in range(repetitions):
		print 'Run ', n+1

		# Make random set of genotypes, alleles are 0 and 1 in simple bit string approach
		male_population = np.floor(randa(individuals/2,genes)+0.5)
		male_population[male_population==0] = -1;
		female_population = np.floor(randa(individuals/2,genes)+0.5)
		female_population[female_population==0] = -1;

		# Build randomly initialised interaction matrix of 1s and 0s
		W = interactionMatrix(genes)

		# keys = ['fitness','parents','children','age']
		keys = ['fitness','children','age']
		male_keys = ['fitness']
		# initialise fitness
		male_pop_fit = np.zeros(individuals/2)
		female_pop_fit = np.zeros(individuals/2)
		# Build array of children indices for females
		female_children = map(tuple,np.full((individuals/2,2),-1,dtype=int))
		# Initialise ages
		male_ages = np.zeros(individuals/2)
		female_ages = np.zeros(individuals/2)
		# Build dictionaries of attributes for each gender
		male_attributes = dict(zip(male_keys,[male_pop_fit]))
		female_attributes = dict(zip(keys,[female_pop_fit,female_children,female_ages]))
		for j in range(individuals/2):
			male_attributes['fitness'][j] = maleFitnessPubGA(male_population[j],W)
			female_attributes['fitness'][j] = maleFitnessPubGA(female_population[j],W)

		# print male_attributes['fitness']
		best_fit = np.zeros(generations)
		pop_avg_fit = np.zeros(generations)
		for k in range(generations):
			for l in range(individuals/2):

				# Roulette wheel selection
				# Select male first, then female
				# print male_attributes['fitness']
				max_fit = sum(male_attributes['fitness'])
				pick = random.uniform(0, max_fit)
				current = 0
				for index,fit in enumerate(male_attributes['fitness']):
					current += fit
					if current > pick:
						male_winner_ind = index
				male_loser_ind = np.argmin(male_attributes['fitness'])

				female_coop_fit = []
				for i,individual in enumerate(female_population):
					female_coop_fit.append(femaleFitnessPubGA(female_attributes['children'][i],male_attributes['fitness'],female_attributes['age'][i],female_attributes['fitness'][i],W,male_population[male_winner_ind]))
				max_fit = sum(female_coop_fit)
				pick = random.uniform(0, max_fit)
				current = 0
				for index,fit in enumerate(female_coop_fit):
					current += fit
					if current > pick:
						female_winner_ind = index
				female_loser_ind = np.argmin(female_coop_fit)

				# Male crossover, select random crossover point, copy mother material past this point
				# Copy into the losing individuals of each gender
				cross_male = int(np.floor(randval()*genes))
				male_population[male_loser_ind][:cross_male] = male_population[male_winner_ind][:cross_male]
				male_population[male_loser_ind][cross_male:] = female_population[female_winner_ind][cross_male:]

				# Female crossover, select random crossover point, copy reverse of mother material up to this point
				# copy rest normally
				cross_female = int(np.floor(randval()*genes))
				female_population[female_loser_ind][:cross_female] = female_population[female_winner_ind][:cross_female][::-1]
				female_population[female_loser_ind][cross_female:] = female_population[female_winner_ind][cross_female:]

				# # Set parents
				# male_attributes['parents'][male_loser_ind] = (male_winner_ind,female_winner_ind)
				# female_attributes['parents'][female_loser_ind] = (male_winner_ind,female_winner_ind)

				# Set children
				# male_attributes['children'][male_winner_ind] = (male_loser_ind,female_loser_ind)
				female_attributes['children'][female_winner_ind] = (male_loser_ind,female_loser_ind)

				# Perform mutation
				for locus in range(genes):
					if randval() < male_mutation:
						male_population[male_loser_ind,locus] = - male_population[male_loser_ind,locus]
					if randval() < female_mutation:
						female_population[female_loser_ind,locus] = - female_population[female_loser_ind,locus]

				# Calculate new genotypes' fitness, add best fit of generation to bestFit
				# pop_fit[loser_ind] = fitness(populaion[loser_ind])
				# pop_fit[loser_ind] = population[loser_ind] * W * population[loser_ind].T
				male_attributes['fitness'][male_loser_ind] = maleFitnessPubGA(male_population[male_loser_ind],W)
				female_attributes['fitness'][female_loser_ind] = maleFitnessPubGA(female_population[female_loser_ind],W)
				# Set ages of children to 0
				# male_attributes['age'][male_loser_ind] = 0
				female_attributes['age'][female_loser_ind] = 0

			# Increment all ages each generation
			# male_attributes['age'] = [val+1 for val in male_attributes['age']]
			female_attributes['age'] = [val+1 for val in female_attributes['age']]

			# Set best_fit and best_geno
			male_max_ind = np.argmax(male_attributes['fitness'])
			female_max_ind = np.argmax(female_attributes['fitness'])
			if male_attributes['fitness'][male_max_ind] > female_attributes['fitness'][female_max_ind]:
				best_fit[k] = male_attributes['fitness'][male_max_ind]
				best_geno = male_population[male_max_ind]
			else:
				best_fit[k] = female_attributes['fitness'][female_max_ind]
				best_geno = female_population[female_max_ind]

			# Set average fitness of population
			pop_avg_fit[k] = ( sum(male_attributes['fitness']) + sum(female_attributes['fitness']) ) / individuals

		final_fit = best_fit[-1]
		avg_fit.append(final_fit)

		# Add best_fit values and pop_avg_fit values
		many_run_best_fit += best_fit
		many_run_pop_avg_fit += pop_avg_fit

		# ax.plot(best_fit)
		# print 'Final fitness score: ', final_fit

		maxf = max(male_attributes['fitness'])
		fmaxf = max(female_attributes['fitness'])
		popGT100 = np.array(male_population[np.where(maxf - male_attributes['fitness'] < 50)])
		fpopGT100 = np.array(female_population[np.where(fmaxf - female_attributes['fitness'] < 50)])
		mbestgenos = np.unique(popGT100, axis=0)
		fbestgenos = np.unique(fpopGT100, axis=0)
		num_good = len(mbestgenos) + len(fbestgenos)
		avg_num_genos.append(num_good)
		# print popGT100
		# print np.unique(popGT100, axis=0)
		# print np.unique(fpopGT100, axis=0)

	print 'Best fitness regular sexual GA on pub problem: ', max(avg_fit)
	avg_fit = sum(avg_fit) / repetitions
	avg_num_genos = sum(avg_num_genos) / repetitions
	print 'Average fitness regular sexual GA on pub problem: ', avg_fit
	print 'Average num good genos found regular sexual GA on pub problem: ', avg_num_genos

	many_run_best_fit = many_run_best_fit / repetitions
	many_run_pop_avg_fit = many_run_pop_avg_fit / repetitions
	return many_run_best_fit, many_run_pop_avg_fit

def microbialGAPub(individuals,genes,generations,mutation,crossover,repetitions):

	fig, ax = plt.subplots()

	avg_fit = []
	avg_num_genos = []
	many_run_best_fit = np.zeros(generations)
	many_run_pop_avg_fit = np.zeros(generations)
	print '\n\nMicrobial GA in progress..'
	for n in range(repetitions):
		print 'Run ', n+1

		# Make random set of genotypes, alleles are 0 and 1 in simple bit string approach
		population = np.floor(randa(individuals,genes)+0.5)
		population[population==0] = -1;

		# Build randomly initialised interaction matrix of 1s and 0s
		W = interactionMatrix(genes)

		# Evaluate initial fitness
		pop_fit = np.zeros(individuals)
		for j in range(individuals):
			pop_fit[j] = maleFitnessPubGA(population[j],W)

		# print pop_fit
		best_fit = np.zeros(generations)
		pop_avg_fit = np.zeros(generations)
		for k in range(generations):
			for l in range(individuals):

				# Gendered tournament selection
				# Take two males, compare fitness, retain winner, retain index of losing indivdidual
				# Compute female fitness based on winning male individual
				# Select two females, compare fitness, retain winner, retain index of loser
				# Perform crossover and mutation, produce two children, one of each gender
				# Replace the losers of each gender with the children
				winner_ind, loser_ind = selection(individuals,population,pop_fit)

				# Perform crossover and mutation
				for locus in range(genes):
					if randval() < crossover:
						# print 'crossover at locus ', locus
						population[loser_ind,locus] = population[winner_ind,locus]
					if randval() < mutation:
						population[loser_ind,locus] = - population[loser_ind,locus] # turn 1 to -1, and vice versa

				# Calculate new genotype's fitness, add best fit of generation to bestFit
				# pop_fit[loser_ind] = fitness(populaion[loser_ind])
				# pop_fit[loser_ind] = population[loser_ind] * W * population[loser_ind].T
				pop_fit[loser_ind] = maleFitnessPubGA(population[loser_ind],W)

			max_ind = np.argmax(pop_fit)
			# print pop_fit
			best_fit[k] = pop_fit[max_ind]
			best_geno = population[max_ind]

			# Set average fitness of population
			pop_avg_fit[k] = sum(pop_fit) / individuals

		final_fit = best_fit[-1]
		avg_fit.append(final_fit)

		# Add best_fit values and pop_avg_fit values
		many_run_best_fit += best_fit
		many_run_pop_avg_fit += pop_avg_fit

		# print '\n\nbest_fit'
		# print best_fit
		# ax.plot(best_fit,label='Single run')
		# print 'Final fitness score: ', final_fit

		maxf = max(pop_fit)
		popGT100 = population[np.where(maxf - pop_fit < 50)]
		# print popGT100
		bestgenos = np.unique(popGT100, axis=0)
		num_good = len(bestgenos)
		avg_num_genos.append(num_good)
		# print 'Num good genotypes found: ', num_good
	print 'Best fitness microbial GA on pub problem: ', max(avg_fit)
	avg_fit = sum(avg_fit) / repetitions
	avg_num_genos = sum(avg_num_genos) / repetitions
	print 'Average fitness microbial GA on pub problem: ', avg_fit
	print 'Average num good genos found microbial GA on pub problem: ', avg_num_genos

	many_run_best_fit = many_run_best_fit / repetitions
	many_run_pop_avg_fit = many_run_pop_avg_fit / repetitions
	# print many_run_pop_avg_fit
	return many_run_best_fit, many_run_pop_avg_fit

# Set up and run algorithms #

# Define parameters for pub problem
num_individuals = 100
num_genes = 20
num_generations = 100
mutation_rate= 1.5/num_genes # regular GA mutation rate should match the sum of the female and male rates
female_mutation_rate = 0.5/num_genes # male mutation rate higher than female in nature
male_mutation_rate = 1/num_genes
no_cross_prob = 1
cross_prob = 0.5
num_repetitions = 20

# min_best_fit, min_avg_fit = minSexualGAPub(num_individuals,num_genes,num_generations,male_mutation_rate,female_mutation_rate,cross_prob,num_repetitions)
# reg_best_fit, reg_avg_fit = regSexualGAPub(num_individuals,num_genes,num_generations,male_mutation_rate,female_mutation_rate,cross_prob,num_repetitions)
# mic_best_fit, mic_avg_fit = microbialGAPub(num_individuals,num_genes,num_generations,mutation_rate,cross_prob,num_repetitions)
# fig, ax = plt.subplots()
# ax.plot(min_best_fit,label='Minimal sexual GA')
# ax.plot(reg_best_fit,label='Regular sexual GA')
# ax.plot(mic_best_fit,label='Microbial GA')
# ax.set_xlabel('Generations')
# ax.set_ylabel('Fitness')
# ax.set_title('Average best individual fitness pub problem, 20 people')
# ax.legend(loc='lower right')
# fig.savefig('./images/avgbestfitness100',bbox_inches='tight')
# fig, ax = plt.subplots()
# ax.plot(min_avg_fit,label='Minimal sexual GA')
# ax.plot(reg_avg_fit,label='Regular sexual GA')
# ax.plot(mic_avg_fit,label='Microbial GA')
# ax.set_xlabel('Generations')
# ax.set_ylabel('Fitness')
# ax.set_title('Average population fitness pub problem, 20 people')
# ax.legend(loc='lower right')
# fig.savefig('./images/avgpopfitness100',bbox_inches='tight')
# # plt.show()

# # Timings for pub problem
# avg_time = []
# for i in range(10):
# 	start= time.clock()
# 	_, _ = minSexualGAPub(num_individuals,num_genes,num_generations,male_mutation_rate,female_mutation_rate,cross_prob,num_repetitions)
# 	end= time.clock()
# 	avg_time.append(end-start)
# print 'Time taken by minimal sexual GA: ', sum(avg_time) / 10
# print 'Standard deviation: ', np.std(avg_time)
# avg_time = []
# for i in range(10):
# 	start= time.clock()
# 	_, _ = regSexualGAPub(num_individuals,num_genes,num_generations,male_mutation_rate,female_mutation_rate,cross_prob,num_repetitions)
# 	end= time.clock()
# 	avg_time.append(end-start)
# print 'Time taken by regular sexual GA: ', sum(avg_time) / 10
# print 'Standard deviation: ', np.std(avg_time)
# avg_time = []
# for i in range(10):
# 	start= time.clock()
# 	_, _ = microbialGAPub(num_individuals,num_genes,num_generations,mutation_rate,cross_prob,num_repetitions)
# 	end= time.clock()
# 	avg_time.append(end-start)
# print 'Time taken by microbial GA: ', sum(avg_time) / 10
# print 'Standard deviation: ', np.std(avg_time)

# Define parameters for knapsack problem
# TODO: get avg, best and timings for range of num_items
num_items = 50

print 'Running genetic algorithms on knapsack problem...'

# min_best_fit, min_avg_fit = minSexualGAKnapsack(num_individuals,num_items,num_generations,male_mutation_rate,female_mutation_rate,cross_prob,num_repetitions,1)
# reg_best_fit, reg_avg_fit = regSexualGAKnapsack(num_individuals,num_items,num_generations,male_mutation_rate,female_mutation_rate,cross_prob,num_repetitions)
# mic_best_fit, mic_avg_fit = microbialGAKnapsack(num_individuals,num_items,num_generations,mutation_rate,cross_prob,num_repetitions)
#
# fig, ax = plt.subplots()
# ax.plot(min_best_fit,label='Minimal sexual GA')
# ax.plot(reg_best_fit,label='Regular sexual GA')
# ax.plot(mic_best_fit,label='Microbial GA')
# ax.set_xlabel('Generations')
# ax.set_ylabel('Fitness')
# ax.set_title('Average best individual fitness knapsack problem, 50 items')
# ax.legend(loc='lower right')
# fig.savefig('./images/avgbestfitnessKP100alt',bbox_inches='tight')
# fig, ax = plt.subplots()
# ax.plot(min_avg_fit,label='Minimal sexual GA')
# ax.plot(reg_avg_fit,label='Regular sexual GA')
# ax.plot(mic_avg_fit,label='Microbial GA')
# ax.set_xlabel('Generations')
# ax.set_ylabel('Fitness')
# ax.set_title('Average population fitness knapsack problem, 50 items')
# ax.legend(loc='lower right')
# fig.savefig('./images/avgpopfitnessKP100alt',bbox_inches='tight')
# plt.show()

# Timings for knapsack problem
avg_time = []
for i in range(10):
	start= time.clock()
	_, _ = minSexualGAKnapsack(num_individuals,num_items,num_generations,male_mutation_rate,female_mutation_rate,cross_prob,num_repetitions,1)
	end= time.clock()
	avg_time.append(end-start)
print 'Time taken by minimal sexual GA: ', sum(avg_time) / 10
print 'Standard deviation: ', np.std(avg_time)
avg_time = []
for i in range(10):
	start= time.clock()
	_, _ = regSexualGAKnapsack(num_individuals,num_items,num_generations,male_mutation_rate,female_mutation_rate,cross_prob,num_repetitions)
	end= time.clock()
	avg_time.append(end-start)
print 'Time taken by regular sexual GA: ', sum(avg_time) / 10
print 'Standard deviation: ', np.std(avg_time)
avg_time = []
for i in range(10):
	start= time.clock()
	_, _ = microbialGAKnapsack(num_individuals,num_items,num_generations,mutation_rate,cross_prob,num_repetitions)
	end= time.clock()
	avg_time.append(end-start)
print 'Time taken by microbial GA: ', sum(avg_time) / 10
print 'Standard deviation: ', np.std(avg_time)
