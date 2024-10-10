import random
from entities import Entity

def select_parents(population):
    return random.sample(population, 2)

def crossover(parents):
    parent1, parent2 = parents
    child1 = Entity(
        height=(parent1.height + parent2.height) / 2,
        speed=(parent1.speed + parent2.speed) / 2,
        cold_tolerance=(parent1.cold_tolerance + parent2.cold_tolerance) / 2,
        heat_tolerance=(parent1.heat_tolerance + parent2.heat_tolerance) / 2,
        altitude_tolerance=(parent1.altitude_tolerance + parent2.altitude_tolerance) / 2,
        energy=(parent1.energy + parent2.energy) / 2,
        lifespan=(parent1.lifespan + parent2.lifespan) / 2,
        age=0,
        reproduction_type=random.choice([parent1.reproduction_type, parent2.reproduction_type]),
        region=random.choice([parent1.region, parent2.region]),
        food_preference=random.choice([parent1.food_preference, parent2.food_preference]),
        color=random.choice([parent1.color, parent2.color]),
        mutations=(parent1.mutations + parent2.mutations) // 2,
        fitness_score=(parent1.fitness_score + parent2.fitness_score) / 2,
        health_status=random.choice([parent1.health_status, parent2.health_status]),
        reproductive_success=(parent1.reproductive_success + parent2.reproductive_success) / 2,
        movement_patterns=random.choice([parent1.movement_patterns, parent2.movement_patterns]),
        entity_type=random.choice([parent1.entity_type, parent2.entity_type]) 
    )
    return [child1]

def mutate(entity):
    mutation_prob = 0.1
    if random.random() < mutation_prob:
        entity.height += random.uniform(-0.1, 0.1)
    return entity

def evolve_population(population):
    selected = select_parents(population)
    offspring = crossover(selected)
    offspring = [mutate(child) for child in offspring]
    return population + offspring

def genetic_algorithm(population, generations):
    for _ in range(generations):
        population = evolve_population(population)
    return population