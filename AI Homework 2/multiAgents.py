# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distanceToNearestFood = 100000
        for food in newFood.asList():
            distanceToNearestFood = min(distanceToNearestFood, manhattanDistance(food, newPos))
        if distanceToNearestFood == 100000:
            distanceToNearestFood = 0
        return successorGameState.getScore() - distanceToNearestFood / 4

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def computeValue(self, gameState, depth, agent):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState), None)
            if agent == 0:
                return maximum(self, gameState, depth)
            else:
                return minimum(self, gameState, depth, agent)
        def maximum(self, gameState, depth):
            maxValue = -100000
            maxAction = None
            for action in gameState.getLegalActions(0):
                value = computeValue(self, gameState.generateSuccessor(0, action), depth, 1)
                if maxValue < value[0]:
                    maxValue = value[0]
                    maxAction = action
            return (maxValue, maxAction)

        def minimum(self, gameState, depth, agent):
            successorAgent = agent
            successorDepth = depth
            if agent == gameState.getNumAgents() - 1:
                successorAgent = 0
                successorDepth -= 1
            else:
                successorAgent += 1
            minValue = 100000
            minAction = None
            for action in gameState.getLegalActions(agent):
                value = computeValue(self, gameState.generateSuccessor(agent, action), successorDepth, successorAgent)
                if minValue > value[0]:
                    minValue = value[0]
                    minAction = action
            return (minValue, minAction)

        return computeValue(self, gameState, self.depth, 0)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def computeValue(self, gameState, depth, agent, alpha, beta):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState), None)
            if agent == 0:
                return maximum(self, gameState, depth, alpha, beta)
            else:
                return minimum(self, gameState, depth, agent, alpha, beta)
        def maximum(self, gameState, depth, alpha, beta):
            maxValue = -100000
            maxAction = None
            for action in gameState.getLegalActions(0):
                value = computeValue(self, gameState.generateSuccessor(0, action), depth, 1, alpha, beta)
                if maxValue < value[0]:
                    maxValue = value[0]
                    maxAction = action
                if maxValue > beta:
                    return (maxValue, maxAction)
                alpha = max(alpha, maxValue)
            return (maxValue, maxAction)

        def minimum(self, gameState, depth, agent, alpha, beta):
            successorAgent = agent
            successorDepth = depth
            if agent == gameState.getNumAgents() - 1:
                successorAgent = 0
                successorDepth -= 1
            else:
                successorAgent += 1
            minValue = 100000
            minAction = None
            for action in gameState.getLegalActions(agent):
                value = computeValue(self, gameState.generateSuccessor(agent, action), successorDepth, successorAgent, alpha, beta)
                if minValue > value[0]:
                    minValue = value[0]
                    minAction = action
                if minValue < alpha:
                    return (minValue, minAction)
                beta = min(minValue, beta)
            return (minValue, minAction)

        return computeValue(self, gameState, self.depth, 0, -100000, 100000)[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def computeValue(self, gameState, depth, agent):
            if depth == 0 or gameState.isLose() or gameState.isWin():
                return (self.evaluationFunction(gameState), None)
            if agent == 0:
                return maximum(self, gameState, depth)
            else:
                return minimum(self, gameState, depth, agent)
        def maximum(self, gameState, depth):
            maxValue = -100000
            maxAction = None
            for action in gameState.getLegalActions(0):
                value = computeValue(self, gameState.generateSuccessor(0, action), depth, 1)
                if maxValue < value[0]:
                    maxValue = value[0]
                    maxAction = action
            return (maxValue, maxAction)

        def minimum(self, gameState, depth, agent):
            successorAgent = agent
            successorDepth = depth
            if agent == gameState.getNumAgents() - 1:
                successorAgent = 0
                successorDepth -= 1
            else:
                successorAgent += 1
            minValue = 0
            minAction = None
            for action in gameState.getLegalActions(agent):
                value = computeValue(self, gameState.generateSuccessor(agent, action), successorDepth, successorAgent)
                minValue += value[0] / len(gameState.getLegalActions(agent))
                minAction = action
            return (minValue, minAction)

        return computeValue(self, gameState, self.depth, 0)[1]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    tmpFood = currentGameState.getFood()
    newFood = tmpFood.asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    distanceToNearestFood = 100000
    for food in newFood:
        distanceToNearestFood = min(distanceToNearestFood, manhattanDistance(food, newPos))

    distanceToNearestGhost = 100000
    for ghost in newGhostStates:
        distanceToNearestGhost = min(distanceToNearestGhost, manhattanDistance(ghost.getPosition(), newPos))

    sign = 1
    if not newScaredTimes: sign = -1

    return currentGameState.getScore() + sign * 2 / (distanceToNearestGhost + 1) + 2 / (distanceToNearestFood + 1)
# Abbreviation
better = betterEvaluationFunction
