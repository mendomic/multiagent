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

        agentList = []
        for depth in range(0, self.depth):
            for agent in range(0, gameState.getNumAgents()):
                agentList.append(agent)
        print(agentList)

        def value(state):
            if len(agentList) == 1:
                return state.getScore()
            if agentList[0] == 0:
                return maxValue(state)
            else:
                return minValue(state)
            agentList.pop(0)

        
        def maxValue(state):
            v = float('-inf')

            # Get successors of the state
            currentSuccessors = []
            for action in state.getLegalActions(0):
                currentSuccessors.append(state.generateSuccessor(0, action))

            # Identify the successor with the highest score
            for successor in state.generateSuccessor(0):
                v = max(v, value(successor))
            return v

        def minValue(state):
            v = float('inf')

            # Get successors of the state
            currentSuccessors = []
            for action in state.getLegalActions(0):
                currentSuccessors.append(state.generateSuccessor(0, action))

            # Identify the successor with the highest score
            for successor in state.generateSuccessor(0):
                v = min(v, value(successor))
            return v
        

        currentSuccessors = [gameState]
        nextSuccessors = []
        
        for successor in currentSuccessors:
            for action in successor.getLegalActions(0):
                nextSuccessors.append((successor.generateSuccessor(0, action), action))
                
        print("here", nextSuccessors)
        currentSuccessors = nextSuccessors.copy()
        nextSuccessors = []
        print("there", currentSuccessors)
        
        tempGhostSuccessors = []
        for level in range(2, self.depth * 2 - 1):
            # Pacman
            if (level % 2 == 1):
                for successor in currentSuccessors:
                    currSuccessor,currAction = successor
                    print(currSuccessor)
                    print(currSuccessor.getLegalActions(0))
                    for action in currSuccessor.getLegalActions(0):
                        print("here")
                        nextSuccessors.append((successor.generateSuccesor(0,action), currAction))
                       

            # Ghost Agent(s)
            else:
                for successor in currentSuccessors:
                    currSuccessor,currAction = successor
                    print("howdy",currSuccessor)
                    for action in currSuccessor.getLegalActions(1):
                        print("yeet")
                        tempGhostSuccessors.append((successor.generateSuccesor(1, action), currAction))
                    print("yoooo",tempGhostSuccessors)
                if gameState.getNumAgents() == 3:
                    for successor in tempGhostSuccessors:
                        currSuccessor,currAction = successor
                        for action in currSuccessor.getLegalActions(2):
                            nextSuccessors.append((successor.generateSuccesor(2, action), currAction))
                else:
                    nextSuccessors = tempGhostSuccessors.copy()
                    
                tempGhostSuccessors = []

            currentSuccessors = nextSuccessors.copy()
            nextSuccessors = []
            
            
        max = float('-inf')
        for terminalSuccessor in currentSuccessors:
            currSuccessor,currAction = terminalSuccessor
            print(currSuccessor.evaluationFunction)
            if currSuccessor.evaluationFunction > max:
                max = terminalSuccessor

        successor,action = max
        
        return action
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
