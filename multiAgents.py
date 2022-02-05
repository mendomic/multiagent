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


from hashlib import blake2b
from util import manhattanDistance
from game import Directions
import random, util
import copy

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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

    def evaluationFunction(self, currentGameState, action):
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

        # Less total food = higher score
        # If newPos is in newFood, raise score
        # Lower distance to scared ghost = higher score
        
        if successorGameState.isWin():
            return float('inf')
        
        if successorGameState.isLose():
            return float('-inf')

        score = 0
        x,y = newPos
        
        minFoodDist = float('inf')
        for food in newFood.asList():
            distance = manhattanDistance(newPos, food)
            if distance < minFoodDist:
                minFoodDist = distance
        
        score += 10 / minFoodDist
            
        score += successorGameState.getScore()
            
        for ghost in newGhostStates:
            distance = manhattanDistance(newPos, ghost.getPosition())
            if ghost.scaredTimer > distance:
                score += ghost.scaredTimer - distance
            else:
                score -= 10 / distance

        return score

def scoreEvaluationFunction(currentGameState):
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

    def getAction(self, gameState):
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
        
        def value(state):
            # Reached terminal state
            if len(self.agentList) == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # 0 = Pacman, maximizer
            if self.agentList[0] == 0:
                return maxValue(state)
            
            # >0 = ghost, minimizer
            else:
                return minValue(state)
        
        def maxValue(state):
            # Get successors of the state
            currentSuccessors = []
            for action in state.getLegalActions(0):
                currentSuccessors.append(state.generateSuccessor(0, action))
            
            self.agentList.pop(0) # move on to next agent
            # save agentList for future use
            currAgentList = copy.deepcopy(self.agentList)

            # Identify the successor with the highest score
            v = float('-inf')
            for successor in currentSuccessors:
                # re-instate where agentList was at, helps when there is more
                # than 1 successor
                self.agentList = copy.deepcopy(currAgentList)
                v = max(v, value(successor))
            return v

        def minValue(state):
            # Get successors of the state
            currentSuccessors = []
            for action in state.getLegalActions(self.agentList[0]):
                currentSuccessors.append(state.generateSuccessor(self.agentList[0], action))
            
            self.agentList.pop(0) # move on to next agent
            # save agentList for future use
            currAgentList = copy.deepcopy(self.agentList)

            # Identify the successor with the lowest score
            v = float('inf')
            for successor in currentSuccessors:
                # re-instate where agentList was at, helps when there is more
                # than 1 successor
                self.agentList = copy.deepcopy(currAgentList)
                v = min(v, value(successor))
            return v

        # Get successors of the state
        legalMoves = gameState.getLegalActions()
        
        # get the scores 
        scores = []
        for action in legalMoves:
            self.agentList = []
            for depth in range(0, self.depth):
                for agent in range(0, gameState.getNumAgents()):
                    self.agentList.append(agent)
            self.agentList.pop(0)
            score = value(gameState.generateSuccessor(0, action))
            scores.append(score)

        # Choose one of the best actions
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]  

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(state,a,b):
            # Reached terminal state
            if len(self.agentList) == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            # 0 = Pacman, maximizer
            if self.agentList[0] == 0:
                return maxValue(state,a,b)
            
            # >0 = ghost, minimizer
            else:
                return minValue(state,a,b)
        
        def maxValue(state,a,b):
            # save agentList for future use
            currAgentList = copy.deepcopy(self.agentList)
            v = float('-inf')
            
            # Get successors of the state
            for action in state.getLegalActions(0):
                # re-instate where agentList was at, helps when there is more
                # than 1 successor
                self.agentList = copy.deepcopy(currAgentList)
                successor = state.generateSuccessor(0, action)
                self.agentList.pop(0) # move on to next agent
                
                v = max(v, value(successor,a,b))
                if v > b:
                    break
                a = max(a, v)
            return v

        def minValue(state,a,b):
            # save agentList for future use
            currAgentList = copy.deepcopy(self.agentList)
            v = float('inf')
        
            # Get successors of the state
            for action in state.getLegalActions(self.agentList[0]):
                # re-instate where agentList was at, helps when there is more
                # than 1 successor
                self.agentList = copy.deepcopy(currAgentList)
                successor = state.generateSuccessor(self.agentList[0], action)
                self.agentList.pop(0) # move on to next agent
                
                v = min(v, value(successor,a,b))
                if v < a:
                    break
                b = min(b, v)
            return v

        # Get successors of the state
        legalMoves = gameState.getLegalActions()
        
        # get the scores 
        scores = []
        a = float('-inf')
        b = float('inf')
        for action in legalMoves:
            self.agentList = []
            for depth in range(0, self.depth):
                for agent in range(0, gameState.getNumAgents()):
                    self.agentList.append(agent)
            self.agentList.pop(0)
            score = value(gameState.generateSuccessor(0, action),a,b)
            a = max(a,score)
            scores.append(score)

        # Choose one of the best actions
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex] 

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def value(state):
            if len(self.agentList) == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # 0 = Pacman, maximizer
            if self.agentList[0] == 0:
                return maxValue(state)
            
            # >0 = ghost, minimizer
            else:
                return expValue(state)
        
        def maxValue(state):
            # Get successors of the state
            currAgentList = copy.deepcopy(self.agentList)
            v = float('-inf')

            # Get successors of the state
            for action in state.getLegalActions(0):
                # re-instate where agentList was at, helps when there is more
                # than 1 successor
                self.agentList = copy.deepcopy(currAgentList)
                successor = state.generateSuccessor(0, action)
                self.agentList.pop(0) # move on to next agent
                
                v = max(v, value(successor))
            return v
        
        def expValue(state):
            # Get successors of the state
            currentSuccessors = []
            for action in state.getLegalActions(self.agentList[0]):
                currentSuccessors.append(state.generateSuccessor(self.agentList[0], action))
            
            self.agentList.pop(0) # move on to next agent
            # save agentList for future use
            currAgentList = copy.deepcopy(self.agentList)

            # Identify the successor with the lowest score
            v = 0.0
            p = 1.0 / len(currentSuccessors)
            for successor in currentSuccessors:
                # re-instate where agentList was at, helps when there is more
                # than 1 successor
                self.agentList = copy.deepcopy(currAgentList)
                v += p * value(successor)
            return v
        
        # Get successors of the state
        legalMoves = gameState.getLegalActions()
        
        # get the scores 
        scores = []
        for action in legalMoves:
            self.agentList = []
            for depth in range(0, self.depth):
                for agent in range(0, gameState.getNumAgents()):
                    self.agentList.append(agent)
            self.agentList.pop(0)
            score = value(gameState.generateSuccessor(0, action))
            scores.append(score)

        # Choose one of the best actions
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]  

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    # Less total food = higher score
    # If newPos is in newFood, raise score
    # Lower distance to scared ghost = higher score
        
    if currentGameState.isWin():
        return float('inf')
        
    if currentGameState.isLose():
        return float('-inf')

    score = 0
    score += currentGameState.getScore()
    
    x,y = newPos
        
    minFoodDist = float('inf')
    for food in newFood.asList():
        distance = manhattanDistance(newPos, food)
        if distance < minFoodDist:
            minFoodDist = distance
        
    score += 10 / minFoodDist
            
    for ghost in newGhostStates:
        distance = manhattanDistance(newPos, ghost.getPosition())
        if ghost.scaredTimer > distance:
            score += ghost.scaredTimer - distance
        else:
            score -= 1 / distance

    return score

# Abbreviation
better = betterEvaluationFunction
