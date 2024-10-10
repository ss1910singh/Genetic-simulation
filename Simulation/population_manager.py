import random

class PopulationManager:
    def __init__(self):
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(100):  # Start with 100 entities
            entity = {
                'type': random.choice(['herbivore', 'carnivore', 'omnivore']),
                'x': random.randint(0, 800),  # Random position on the screen (X-axis)
                'y': random.randint(0, 600),  # Random position on the screen (Y-axis)
                'speed': random.uniform(1, 5),  # Random speed for movement
                'fitness': random.uniform(0, 100),  # Fitness score
                'alive': True
            }
            population.append(entity)
        return population

    def evolve_population(self):
        for entity in self.population:
            self.move_entity(entity)
            self.check_reproduction_or_death(entity)

    def move_entity(self, entity):
        # Simple random movement (can be expanded based on behaviors)
        entity['x'] += random.randint(-1, 1) * entity['speed']
        entity['y'] += random.randint(-1, 1) * entity['speed']

        # Ensure the entity stays within the screen bounds
        entity['x'] = max(0, min(800, entity['x']))
        entity['y'] = max(0, min(600, entity['y']))

    def check_reproduction_or_death(self, entity):
        # Simulate random reproduction and death
        if random.random() < 0.01:  # 1% chance of reproduction per frame
            self.population.append({
                'type': entity['type'],
                'x': entity['x'],
                'y': entity['y'],
                'speed': entity['speed'],
                'fitness': random.uniform(0, 100),
                'alive': True
            })
        if random.random() < 0.005:  # 0.5% chance of death per frame
            entity['alive'] = False

        # Remove dead entities
        self.population = [e for e in self.population if e['alive']]

    def get_population(self):
        return self.population
