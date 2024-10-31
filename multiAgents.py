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
        score = successorGameState.getScore()

        # minimizar distancia hacia la comida mas cercana
        foodDistances = [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        score -= min(foodDistances) if foodDistances else 0
        
        # restar del puntaje la cantidad de comida restante
        score -= 5 * len(newFood.asList())

        # escapar de los fantasmas o perseguirlos si estan asustado
        for ghost, scaredTime in zip(newGhostStates, newScaredTimes):
            distance = manhattanDistance(newPos, ghost.getPosition())
            if distance == 0:
              break
            
            if scaredTime > 0:
                score += 200 / distance
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
        """
        "*** YOUR CODE HERE ***"
        best_score = -100000
        best_action = None
        for action in gameState.getLegalActions(0): 
            successor = gameState.generateSuccessor(0, action)
            score = self.minimax(successor, 1, 0) # empezar con minimax de pacman con depth 0
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def minimax(self, gameState, agentIndex, depth):
        # revisar si se llego al maximo depth o termino el juego
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.maximizar(gameState, depth) # turno de pacman
        else:
            return self.minimizar(gameState, depth, agentIndex) # turno de fantasmas

    def maximizar(self, gameState, depth):
        actions = gameState.getLegalActions(0) 
        if not actions:
            return self.evaluationFunction(gameState)

        maxEval = -100000
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            maxEval = max(maxEval, self.minimax(successor, 1, depth))  # continua con el fantasma 1
        return maxEval

    def minimizar(self, gameState, depth, agentIndex):
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState)

        nextAgent = (agentIndex + 1) % gameState.getNumAgents() # siguiente fantasma
        nextDepth = depth + 1 if nextAgent == 0 else depth  # solo se incrementa depth si es el turno de pacman (agente 0) otra vez

        minEval = 100000
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            minEval = min(minEval, self.minimax(successor, nextAgent, nextDepth))
        return minEval

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """ 
    def getAction(self, gameState): 
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        best_score = -100000
        best_action = None
        
        alpha = -100000
        beta = 100000
        
        for action in gameState.getLegalActions(0): 
            successor = gameState.generateSuccessor(0, action)
            score = self.minimax(successor, 1, 0, alpha, beta) # empezar con minimax de pacman con depth 0
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)
            
        return best_action

    def minimax(self, gameState, agentIndex, depth, alpha, beta):
        # revisar si se llego al maximo depth o termino el juego
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        if agentIndex == 0:
            return self.maximizar(gameState, depth, alpha, beta) # turno de pacman
        else:
            return self.minimizar(gameState, depth, agentIndex, alpha, beta) # turno de fantasmas

    def maximizar(self, gameState, depth, alpha, beta):
        actions = gameState.getLegalActions(0) 
        if not actions:
            return self.evaluationFunction(gameState)

        maxEval = -100000
        for action in actions:
            successor = gameState.generateSuccessor(0, action)
            maxEval = max(maxEval, self.minimax(successor, 1, depth, alpha, beta))  # continua con el fantasma 1
            alpha = max(alpha, maxEval)
            if alpha > beta:
                break  # poda beta
        return maxEval

    def minimizar(self, gameState, depth, agentIndex, alpha, beta):
        actions = gameState.getLegalActions(agentIndex)
        if not actions:
            return self.evaluationFunction(gameState)

        nextAgent = (agentIndex + 1) % gameState.getNumAgents() # siguiente fantasma
        nextDepth = depth + 1 if nextAgent == 0 else depth  # solo se incrementa depth si es el turno de pacman (agente 0) otra vez

        minEval = 100000
        for action in actions:
            successor = gameState.generateSuccessor(agentIndex, action)
            minEval = min(minEval, self.minimax(successor, nextAgent, nextDepth, alpha, beta))
            beta = min(beta, minEval)
            if beta < alpha:
                break  # poda alpha
        return minEval


class ExpectimaxAgent(MultiAgentSearchAgent): 
    """
      Your expectimax agent (question 4)
    """
    alfabeta = AlphaBetaAgent()

    def getAction(self, gameState): # use el mismo agente de la pregunta 3 ya que no tenemos que hacer la 4
        return AlphaBetaAgent.getAction(self.alfabeta, gameState) # y parece funcionar bien?

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: el agente hace casi lo mismo que el agente reflejo
      solo que ahora tiene en cuenta el estado actual del juego y no todas
      las acciones disponibles. en general el agente trata de:
      -buscar comida
      -correr de fantasma y perseguir los asustado
      -buscar capsulas
      -peor puntaje entre mas comida quede
      -peor puntaje entre mas capsulas queden
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()

    # menor distancia a comida
    foodDistances = [manhattanDistance(pacmanPos, foodPos) for foodPos in food.asList()]
    closestFoodDist = min(foodDistances) if foodDistances else 1
    if closestFoodDist == 0:
        closestFoodDist = 0.1

    # menor distancia a fantasma
    ghostDistances = []
    for ghost in ghostStates:
        dist = manhattanDistance(pacmanPos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            ghostDistances.append(-dist)  # perseguir fantasma asustado
        else:
            ghostDistances.append(dist)  # correr de fantasma
    closestGhostDist = min(ghostDistances) if ghostDistances else 1
    if closestGhostDist == 0:
        closestGhostDist = 0.1
    
    # menor distancia a capsulas
    capsuleDistances = [manhattanDistance(pacmanPos, cap) for cap in capsules]
    closestCapsuleDist = min(capsuleDistances) if capsuleDistances else 1
    if closestCapsuleDist == 0:
        closestCapsuleDist = 0.1

    # evaluacion final
    evaluation = (
        score +
        (1 / closestFoodDist) * 10 -   # buscar comida
        (1 / closestGhostDist) * 20 +  # correr de fantasma
        (1 / closestCapsuleDist) * 15  # buscar capsulas
        - len(food.asList()) * 5       # peor puntaje entre mas comida quede
        - len(capsules) * 10           # peor puntaje entre mas capsulas queden
    )

    return evaluation

# Abbreviation
better = betterEvaluationFunction

