class Environment:
    def __init__(self, width, height, regions):
        self.width = width
        self.height = height
        self.grid = [[None for _ in range(width)] for _ in range(height)]
        self.regions = regions

    def place_entity(self, entity, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = entity

    def get_region(self, x, y):
        return self.regions.get(self.grid[y][x].region, None)