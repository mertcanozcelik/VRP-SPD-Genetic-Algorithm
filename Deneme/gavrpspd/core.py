# -*- coding: utf-8 -*-

import os
import random
from json import load
from csv import DictWriter
from deap import base, creator, tools


# Çizilen Route'un Gösterimi Yapılır
def printRoute(route, merge=False):
    routeStr = '0'
    subRouteCount = 0
    twoResources = 0
    for subRoute in route:
        subRouteCount += 1
        subRouteStr = '0'
        if twoResources == 1:
           pass
                        
        else:
            for customerID in subRoute:
                subRouteStr = subRouteStr + ' - ' + str(customerID)
                routeStr = routeStr + ' - ' + str(customerID)
            subRouteStr = subRouteStr + ' - 0'
            if not merge:
                print ('  Vehicle %d\'s route: %s' % (subRouteCount, subRouteStr))
                routeStr = routeStr + ' - 0'
        if merge:
            print (routeStr)

    return
# İndisler routelara çevrilir.
def ind2route(individual, instance, speed=1.0):
    route = []
    
    vehicleCapacity = instance['vehicle_capacity']
    

    # Sub Routelar hesaplanır.
    subRoute = []
    vehicleLoad = 0
    pick_upload= 0
    lastCustomerID = 0
    for customerID in individual:
        # Update vehicle load
        demand = instance['customer_%d' % customerID]['demand']
        pick_up = instance['customer_%d' % customerID]['pick_up']
        
        updatedVehicleLoad = vehicleLoad+demand #Toplam Demand Miktarı hesaplanır.
        
        pick_upload += pick_up # Toplam Pick-Up Miktarı hesaplanır.
        

        if  (updatedVehicleLoad<=vehicleCapacity) and (pick_upload<=vehicleCapacity):
            # Add to current sub-route
            subRoute.append(customerID)
            vehicleLoad = updatedVehicleLoad
            
        else:
            # Save current sub-route
            route.append(subRoute)
            # Initialize a new sub-route and add to it
            subRoute = [customerID]
            vehicleLoad = demand
            pick_upload = pick_up
        # En sonki müşteri ID'si güncellenir.
        lastCustomerID = customerID
        
    if subRoute != []:
        # Eğer Route değeri boş değilse en son route değeri kaydedilir.
        route.append(subRoute)
    
    return route




# Fitness Değeri Hesaplanır.
def evalVRPTW(individual, instance, unitCost=1.0, initCost= 0, speed=1):
    totalCost = 0
    route = ind2route(individual, instance, speed)
    totalCost = 0
    for subRoute in route:
        
        subRouteDistance = 0
        
        lastCustomerID = 0
        for customerID in subRoute:
            # Distance matrixten mesafe hesaplanır.
            distance = instance['distance_matrix'][lastCustomerID][customerID] * speed
            # Son Sub-Route mesafesi güncellenir.
            subRouteDistance = subRouteDistance + distance
            # Son müşteri ID'si güncellenir.
            lastCustomerID = customerID
        # Calculate transport cost
        subRouteDistance = subRouteDistance + (instance['distance_matrix'][lastCustomerID][0] * speed)
        subRouteTranCost = unitCost * subRouteDistance
        # Obtain sub-route cost
        subRouteCost = subRouteTranCost
        # Update total cost
        totalCost = totalCost + subRouteCost
    fitness = 1.0 / totalCost
    return fitness,

#PMX prosedürü hesaplanır.
def cxPartialyMatched(ind1, ind2):
    #individual büyüklüğü kadar 1xindis büyüklüğünde 0 matrisi oluşturulur.
    size = min(len(ind1), len(ind2))
    p1, p2 = [0]*size, [0]*size

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]-1] = i
        p2[ind2[i]-1] = i
    # Crossover Noktaları seçilir
    cxpoint1 = random.randint(0, size)
    cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else: #Crossover noktaları yer değiştirilir
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    
    # Crossover noktaları crossover uygulanır.
    for i in range(cxpoint1, cxpoint2):
        # Seçilen değerler takip edilir.
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Eşleşmiş değerler yer değiştirilir.
        ind1[i], ind1[p1[temp2-1]] = temp2, temp1
        ind2[i], ind2[p2[temp1-1]] = temp1, temp2
        # En son pozisyon kaydedilir
        p1[temp1-1], p1[temp2-1] = p1[temp2-1], p1[temp1-1]
        p2[temp1-1], p2[temp2-1] = p2[temp2-1], p2[temp1-1]
    
    return ind1, ind2

def mutInverseIndexes(individual): # Mutasyon yapılır.
    start, stop = sorted(random.sample(range(len(individual)), 2))
    individual = individual[:start] + individual[stop:start-1:-1] + individual[stop+1:]
    return individual,

#Genetik Algoritma çalıştırılır.
def gaVRPTW(instName, unitCost, indSize, popSize, cxPb, mutPb, NGen, exportCSV=False, customizeData=False):
    # Dosya yolundan data import edilir.
    if customizeData:
        jsonDataDir = os.path.join(BASE_DIR,'data', 'json_customize')
    with open(jsonFile) as f:
        instance = load(f)

    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register('indexes', random.sample, range(1, indSize + 1), indSize)
    # Structure initializers
    toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.indexes)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # Operator registering
    toolbox.register('evaluate', evalVRPTW, instance=instance, unitCost=unitCost )
    toolbox.register('select', tools.selRoulette)
    toolbox.register('mate', cxPartialyMatched)
    toolbox.register('mutate', mutInverseIndexes)
    # Popülasyon Başlatılır.
    pop = toolbox.population(n=popSize)
    # Results holders for exporting results to CSV file
    csvData = []
    print ('Start of evolution')
    # Evaluate the entire population
    fitnesses = list(toolbox.map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # Debug, suppress print()
    # print '  Evaluated %d individuals' % len(pop)
    # Begin the evolution
    for g in range(NGen):

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(toolbox.map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxPb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        for mutant in offspring:
            if random.random() < mutPb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalidInd = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalidInd)
        for ind, fit in zip(invalidInd, fitnesses):
            ind.fitness.values = fit
        # Debug, suppress print()
        # print '  Evaluated %d individuals' % len(invalidInd)
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        # Debug, suppress print()
        # print '  Min %s' % min(fits)
        # print '  Max %s' % max(fits)
        # print '  Avg %s' % mean
        # print '  Std %s' % std
        # Write data to holders for exporting results to CSV file

    print ('-- End of (successful) evolution --')
    bestInd = tools.selBest(pop, 1)[0]
    print ('Best individual: %s' % bestInd)
    print ('Fitness: %s' % bestInd.fitness.values[0])
    print(Route(ind2route(bestInd, instance)))
    print ('Total cost: %s' % (1 / bestInd.fitness.values[0]))


