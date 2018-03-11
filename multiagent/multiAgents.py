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

import sys
from util import manhattanDistance
from game import Directions
from game import Actions
import random, util

from game import Agent

def closestFoodOrComputeDistance(curP, destP, gameState):
    if curP == destP: return None, 0
    walls = gameState.getWalls()
    foodList = gameState.getFood().asList()
    count =  0
    points = [curP,]
    queue = util.Queue()
    queue.push(curP)
    while not queue.isEmpty():
        count += 1
        qsize = len(queue.list)
        for i in range(qsize):
            temp = queue.pop()
            for direction in [Directions.WEST, Directions.NORTH, Directions.EAST, Directions.SOUTH]:
                x,y = temp
                dx, dy = Actions.directionToVector(direction)
                nextx, nexty = int(x + dx), int(y + dy)
                if not walls[nextx][nexty]:
                    if (nextx, nexty) not in points:
                        queue.push((nextx, nexty))
                        points.append((nextx, nexty))
                    
                    if destP == None:
                        if (nextx, nexty) in foodList:
                            return (nextx, nexty), count
                    elif (nextx, nexty) == destP:
                        return None, count
    return None, None

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        return successorGameState.getScore()

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
    def maxVal(self,state,agent,depth):   #pacman's turn get the max val
        val = -999999
        actions = state.getLegalActions(agent)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        for action in actions:
            val = max(val,self.minimax(state.generateSuccessor(agent,action),agent+1,depth+1))
        return val

    def minVal(self,state,agent,depth):   #ghosts' turn to get the min val
        val = 999999
        for action in state.getLegalActions(agent):
            val = min(val,self.minimax(state.generateSuccessor(agent,action),agent+1,depth+1))
        return val

    def minimax(self,gameState,agent,depth):
        score = 0
        if agent == self.agentCount: ##it's pacman's turn
            agent = self.index
        if depth == self.depth*self.agentCount or gameState.isWin() or gameState.isLose():## finish?
            score = self.evaluationFunction(gameState)
        elif agent == self.index:   ## judge again, is pacman's turn?
            score = self.maxVal(gameState,agent,depth)
        else:         ##ghosts's turn
            score = self.minVal(gameState,agent,depth)
        return score

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
        """
        depth = 0
        agentIndex = self.index
        Dict = {}
        self.agentCount = gameState.getNumAgents()
        actions = gameState.getLegalActions(agentIndex)
        if Directions.STOP in actions:
            actions.remove(Directions.STOP)
        for action in actions:
            eval_f = self.minimax(gameState.generateSuccessor(agentIndex,action),agentIndex+1,depth+1)
            Dict[eval_f] = action
        choices = Dict[max(Dict)]
        return choices


# class MinimaxAgent(MultiAgentSearchAgent):
#     """
#       Your minimax agent (question 2)
#     """

#     def getAction(self, gameState):
#         """
#           Returns the minimax action from the current gameState using self.depth
#           and self.evaluationFunction.

#           Here are some method calls that might be useful when implementing minimax.

#           gameState.getLegalActions(agentIndex):
#             Returns a list of legal actions for an agent
#             agentIndex=0 means Pacman, ghosts are >= 1

#           gameState.generateSuccessor(agentIndex, action):
#             Returns the successor game state after an agent takes an action

#           gameState.getNumAgents():
#             Returns the total number of agents in the game
#         """
#         "*** YOUR CODE HERE ***"
#         # return self.dfMiniMaxSearch(self.depth, gameState, True)[1]

#         def dfMiniMax(depth, curState, agentId):
#             if depth == 0 or curState.isWin() or curState.isLose():
#                 return self.evaluationFunction(curState)

#             legalActions = curState.getLegalActions(agentId)
#             if agentId == 0:
#                 maxScore = -sys.maxint
#                 for action in legalActions:
#                     nextState = curState.generateSuccessor(agentId, action)
#                     maxScore = max(maxScore, dfMiniMax(depth, nextState, agentId+1))
#                 return maxScore
#             else :
#                 minScore = sys.maxint
#                 for action in legalActions:
#                     nextState = curState.generateSuccessor(agentId, action)
#                     if agentId+1 == curState.getNumAgents():
#                         minScore = min(minScore, dfMiniMax(depth-1, nextState, 0))
#                     else: minScore = min(minScore, dfMiniMax(depth, nextState, agentId+1))
#                 return minScore

#         legalActions = gameState.getLegalActions(0)
#         childList = [gameState.generateSuccessor(0, action) for action in legalActions]
#         scores = [dfMiniMax(self.depth, nextState, 1) for nextState in childList]
#         bestScore = max(scores)
#         bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
#         # print dir(gameState.problem)
#         # closestFood = closestFoodOrComputeDistance(gameState.getPacmanPosition(), None, gameState)
#         # dists = [closestFoodOrComputeDistance(nextState.getPacmanPosition(), closestFood[0], nextState)[1] \
#         # for nextState in childList]
#         # chosenIndex = min(bestIndices, key=lambda x: dists[x])
#         random.seed()
#         chosenIndex = random.choice(bestIndices)
#         return legalActions[chosenIndex]

#         util.raiseNotDefined()

#     # consider only pacman, see all ghosts as min
#     # def dfMiniMaxSearch(self, depth, curState, pacmanTurn):
#     #     if depth == 0 or curState.isWin() or curState.isLose():
#     #         return curState.getScore(), None
#     #     legalActions = curState.getLegalActions()
#     #     successors = [curState.generateSuccessor(0, action) for action in legalActions]
#     #     scores = [self.dfMiniMaxSearch(depth-1, nextState, not pacmanTurn)[0] for nextState in successors]
#     #     bestScore = max(scores) if pacmanTurn else min(scores)
#     #     bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
#     #     closestFood = closestFoodOrComputeDistance(curState.getPacmanPosition(), None, curState)
#     #     dists = [closestFoodOrComputeDistance(nextState.getPacmanPosition(), closestFood[0], nextState)[1] \
#     #     for nextState in successors]
#     #     chosenIndex = min(bestIndices, key=lambda x: dists[x])
#     #     # chosenIndex = random.choice(bestIndices) # Pick randomly among the best
#     #     return bestScore, legalActions[chosenIndex]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # return self.alphabeta(self.depth, gameState, -sys.maxint, sys.maxint, True)[1]
        
        def dfAlphaBeta(depth, curState, agentId, alpha, beta):
            legalActions = curState.getLegalActions(agentId)
            if depth == 0 or curState.isWin() or curState.isLose():
                return curState.getScore()
            if agentId == 0:
                v = -sys.maxint
                for action in legalActions:
                    nextState = curState.generateSuccessor(agentId, action)
                    v = max(v, dfAlphaBeta(depth, nextState, agentId+1, alpha, beta))
                    alpha = max(alpha, v)
                    if alpha > beta: break
                return v
            else :
                v = sys.maxint
                for action in legalActions:
                    nextState = curState.generateSuccessor(agentId, action)
                    if agentId+1 == curState.getNumAgents():
                        v = min(v, dfAlphaBeta(depth-1, nextState, 0, alpha, beta))
                    else: v = min(v, dfAlphaBeta(depth, nextState, agentId+1, alpha, beta))
                    beta = min(beta, v)
                    if alpha > beta: break
                return v

        bestIndices = []
        legalActions = gameState.getLegalActions(0)
        alpha, beta = -sys.maxint, sys.maxint
        for i, action in enumerate(legalActions):
            nextState = gameState.generateSuccessor(0, action)
            newAlpha = dfAlphaBeta(self.depth, nextState, 1, alpha, beta)
            if alpha < newAlpha:
                alpha, bestIndices = newAlpha, [i]
            elif alpha == newAlpha:
                bestIndices.append(i)
        
        random.seed()
        chosenIndex = random.choice(bestIndices)
        return legalActions[chosenIndex]

        util.raiseNotDefined()

    # consider only pacman, see all ghosts as min
    # def alphabeta(self, depth, curState, alpha, beta, pacmanTurn):
    #     if depth == 0 or curState.isWin() or curState.isLose():
    #         return curState.getScore(), None
    #     legalActions = curState.getLegalActions()
    #     successors = [curState.generateSuccessor(0, action) for action in legalActions]
    #     bestIndices = []
    #     if pacmanTurn:
    #         for i, nextState in enumerate(successors):
    #             newAlpha = self.alphabeta(depth-1,nextState,alpha,beta,not pacmanTurn)[0]
    #             if alpha < newAlpha:
    #                 alpha, bestIndices = newAlpha, [i]
    #             elif alpha == newAlpha:
    #                 bestIndices.append(i)
    #             if beta < alpha: break
    #         if depth != self.depth: return alpha, None
    #     else :
    #         for i, nextState in enumerate(successors):
    #             newBeta = self.alphabeta(depth-1,nextState,alpha,beta,not pacmanTurn)[0]
    #             if beta > newBeta:
    #                 beta, bestIndices = newBeta, [i]
    #             elif beta == newBeta:
    #                 bestIndices.append(i)
    #             if beta < alpha: break
    #         if depth != self.depth: return beta, None
    #     closestFood = closestFoodOrComputeDistance(curState.getPacmanPosition(), None, curState)
    #     dists = [closestFoodOrComputeDistance(nextState.getPacmanPosition(), closestFood[0], nextState)[1] \
    #     for nextState in successors]
    #     if len(bestIndices) == 0:
    #         bestIndices = [i for i in range(len(legalActions))]
    #     chosenIndex = min(bestIndices, key=lambda x: dists[x])
    #     if pacmanTurn: return alpha, legalActions[chosenIndex]
    #     else : return beta, legalActions[chosenIndex]


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

