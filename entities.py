import random

class Entity:
    def __init__(self, height, speed, cold_tolerance, heat_tolerance, altitude_tolerance, energy, lifespan,
                 age, reproduction_type, region, food_preference, color, mutations, fitness_score, health_status,
                 reproductive_success, movement_patterns, entity_type):
        self.height = height
        self.speed = speed
        self.cold_tolerance = cold_tolerance
        self.heat_tolerance = heat_tolerance
        self.altitude_tolerance = altitude_tolerance
        self.energy = energy
        self.lifespan = lifespan
        self.age = age
        self.reproduction_type = reproduction_type
        self.region = region
        self.food_preference = food_preference
        self.color = color
        self.mutations = mutations
        self.fitness_score = fitness_score
        self.health_status = health_status
        self.reproductive_success = reproductive_success
        self.movement_patterns = movement_patterns
        self.entity_type = entity_type
