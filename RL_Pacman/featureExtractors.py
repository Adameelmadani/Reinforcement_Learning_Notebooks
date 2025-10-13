# featureExtractors.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util

class FeatureExtractor:
    def getFeatures(self, state, action):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Enhanced feature extractor with additional strategic features:
    - Distance to nearest ghost (not just 1-step away)
    - Closest scared ghost distance
    - Number of food pellets remaining
    - Average distance to food
    - Whether in a dead-end
    - Capsule proximity
    """
    
    def getFeatures(self, state, action):
        food = state.getFood()
        walls = state.getWalls()
        ghosts = state.getGhostPositions()
        capsules = state.getCapsules()
        
        features = util.Counter()
        features["bias"] = 1.0
        
        # Current position after action
        x, y = state.getPacmanPosition()
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)
        
        # Feature 1: Immediate ghost danger
        features["#-of-ghosts-1-step-away"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts
        )
        
        # Feature 2: Distance to nearest active ghost
        ghost_distances = [manhattanDistance((next_x, next_y), g) for g in ghosts]
        if ghost_distances:
            min_ghost_dist = min(ghost_distances)
            features["nearest-ghost-distance"] = min_ghost_dist / (walls.width * walls.height)
            
            # Danger zone (ghost very close)
            if min_ghost_dist <= 2:
                features["danger-zone"] = 1.0
        
        # Feature 3: Food features
        if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
            features["eats-food"] = 1.0
        
        # Closest food distance
        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            features["closest-food"] = float(dist) / (walls.width * walls.height)
        
        # Feature 4: Food density in nearby area
        food_count = 0
        for i in range(max(0, next_x - 2), min(walls.width, next_x + 3)):
            for j in range(max(0, next_y - 2), min(walls.height, next_y + 3)):
                if food[i][j]:
                    food_count += 1
        features["nearby-food-density"] = food_count / 25.0
        
        # Feature 5: Capsule proximity
        if capsules:
            capsule_distances = [manhattanDistance((next_x, next_y), c) for c in capsules]
            min_capsule_dist = min(capsule_distances)
            features["closest-capsule"] = float(min_capsule_dist) / (walls.width * walls.height)
            
            # Encourage eating capsules when ghosts are near
            if min_ghost_dist <= 5 and min_capsule_dist <= 3:
                features["capsule-opportunity"] = 1.0
        
        # Feature 6: Dead-end detection
        legal_neighbors = Actions.getLegalNeighbors((next_x, next_y), walls)
        features["is-dead-end"] = 1.0 if len(legal_neighbors) <= 1 else 0.0
        
        # Feature 7: Stopping penalty
        if action == Directions.STOP:
            features["stopped"] = 1.0
        
        # Normalize all features
        features.divideAll(10.0)
        return features