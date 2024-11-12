import random
import pandas as pd
from entities import Entity
from genetic_algo import genetic_algorithm
from log_data import log_population_data

def initialize_population(size):
    return [Entity(
        height=random.uniform(1.0, 2.0),
        speed=random.uniform(1.0, 10.0),
        cold_tolerance=random.uniform(0.0, 1.0),
        heat_tolerance=random.uniform(0.0, 1.0),
        altitude_tolerance=random.uniform(0.0, 1.0),
        energy=random.uniform(10.0, 100.0),
        lifespan=random.uniform(10.0, 50.0),
        age=0,
        reproduction_type=random.choice(['asexual', 'sexual']),
        region=random.choice(['cold', 'hot', 'temperate']),
        food_preference=random.choice(['plants', 'animals']),
        color=random.choice(['red', 'blue', 'green']),
        mutations=random.randint(0, 5),
        fitness_score=random.uniform(0.0, 1.0),
        health_status=random.choice(['healthy', 'sick']),
        reproductive_success=random.uniform(0.0, 1.0),
        movement_patterns=random.choice(['linear', 'random']),
        entity_type=random.choice(['A', 'B', 'C'])  # Added entity type
    ) for _ in range(size)]

def run_simulation():
    population = initialize_population(10)
    for generation in range(1, 51):  # 100 generations
        print(f"Generation {generation}")
        population = genetic_algorithm(population, 1)  # Run one generation at a time
        log_population_data(population, generation)  # Log data for each generation

if __name__ == '__main__':
    run_simulation()