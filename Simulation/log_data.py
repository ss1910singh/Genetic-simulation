import pandas as pd
import os

def log_population_data(population, generation):
    data = []
    for entity in population:
        data.append({
            'generation': generation,
            'entity_type': entity.entity_type,
            'height': entity.height,
            'speed': entity.speed,
            'cold_tolerance': entity.cold_tolerance,
            'heat_tolerance': entity.heat_tolerance,
            'altitude_tolerance': entity.altitude_tolerance,
            'energy': entity.energy,
            'lifespan': entity.lifespan,
            'age': entity.age,
            'reproduction_type': entity.reproduction_type,
            'region': entity.region,
            'food_preference': entity.food_preference,
            'color': entity.color,
            'mutations': entity.mutations,
            'fitness_score': entity.fitness_score,
            'health_status': entity.health_status,
            'reproductive_success': entity.reproductive_success,
            'movement_patterns': entity.movement_patterns
        })

    df = pd.DataFrame(data)
    file_exists = os.path.isfile('population_log.csv')
    df.to_csv('population_log.csv', index=False, mode='a', header=not file_exists)