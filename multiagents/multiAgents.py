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
        #print  list(currentGameState.getFood())
        #print  successorGameState.getPacmanPosition(), " successorGameState"
        #print  currentGameState, " currentGameState"
        #print  newPos, " newPos"
        #print  newGhostStates, " newGhostStates"
        #print  newScaredTimes, " newScaredTimes"
        score = 10
        current_pos = currentGameState.getPacmanPosition()
        #min_dist=1/min_dist
        min_ghost_dist=9999
        new_ghost_dist=0
        #curr_ghost_dist=0
        currGhostStates = currentGameState.getGhostStates()
        #currScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        for ghost in newGhostStates:
            ghost_pos=ghost.getPosition()
            # newScaredTimes=ghost.scaredTimer
            # if newScaredTimes<1:
            ghost_dist=manhattanDistance(newPos,ghost_pos)
            if ghost_dist<=min_ghost_dist:
                min_ghost_dist=ghost_dist

        if min_ghost_dist<=1:
            return -9999
        elif(action=='Stop' and min_ghost_dist>=2):
            return -9999
        else:
            score -= 5/min_ghost_dist


        min_dist = 10000
        new_food_list=list(newFood)
        currend_food_list=list(successorGameState.getFood())
        new_dist_food=0
        for food in newFood.asList():
            if food:
                dist = util.manhattanDistance(newPos,food)
                #new_dist_food += dist
                if dist < min_dist:
                    min_dist=dist
        if(currentGameState.getScore()<successorGameState.getScore()):
            score+=100
        ''''for food in currend_food_list:
            if food:

                dist= manhattanDistance(current_pos,food)
                #curr_dist_food += dist
                if dist < min_dist_curr:
                    min_dist_curr=dist

        if curr_dist_food!=0:
            curr_dist_food=1/curr_dist_food

        if new_dist_food!=0:
            new_dist_food=1/new_dist_food

        if new_dist_food>curr_dist_food:
            score+=(new_dist_food-curr_dist_food)*3

        else:
            score-=20
        print score,"after sum of dist"'''''
        if min_dist > 0:
            score+=(20/min_dist)
        # print action, ' ', min_dist, ' ', min_ghost_dist, ' ', currentGameState.getScore(), ' ', successorGameState.getScore(), ' ', score







        #print score
        return score



        "*** YOUR CODE HERE ***"
        #return successorGameState.getScore()


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
        def val_cal(state,index,depth):
            num_agents = state.getNumAgents()

            def pacman_max(curr_state):
                #print("Inside max")
                val,act=-float('inf'),None
                actions = curr_state.getLegalActions(index)
                #actions.remove('STOP')
                #print actions
                if len(actions) == 0:
                    return (self.evaluationFunction(curr_state), None)
                for action in actions:

                    next_state = state.generateSuccessor(index, action)
                    next_val,next_act=val_cal(next_state,next_ind,depth)

                    if next_val > val:
                        val,act=next_val,action
                return val,act

            def ghosts_min(curr_state):
                val,act=float('inf'),None
                actions = state.getLegalActions(index)
                #actions.remove('STOP')
                #print actions
                if len(actions) == 0:
                    return (self.evaluationFunction(curr_state), None)
                for action in actions:
                    next_state = curr_state.generateSuccessor(index, action)
                    next_val, next_act=val_cal(next_state,next_ind,depth)

                    if next_val < val:
                        val,act=next_val,action
                return val,act

            max_depth = self.depth
            next_ind = (index + num_agents + 1) % num_agents
            #print next_ind,"next_ind"
            if num_agents>0:
                if index==0:
                    depth+=1

                if depth == max_depth or gameState.isWin() or gameState.isLose():
                    return self.evaluationFunction(state),None  #scoreEvaluationFunction(gameState),None


                #print(next_ind)
                if index==0:   #pacman
                    return pacman_max(state)


                elif index>0:  #ghosts
                        #print("ghost index",index)
                        return ghosts_min(state)

        value,action=val_cal(gameState,0,-1)
        return action
        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def val_cal(state,index,depth,alpha,beta):
            num_agents = state.getNumAgents()

            def pacman_max(curr_state,alpha,beta):
                #print("Inside max")
                val,act=-float('inf'),None
                actions = curr_state.getLegalActions(index)
                #actions.remove('STOP')
                #print actions
                if len(actions) == 0:
                    return (self.evaluationFunction(curr_state), None)
                for action in actions:

                    next_state = state.generateSuccessor(index, action)
                    next_val,next_act=val_cal(next_state,next_ind,depth,alpha,beta)

                    if next_val > val:
                        val,act=next_val,action

                    if beta<val:
                        return val,act

                    if alpha<val:
                        alpha=val
                return val,act

            def ghosts_min(curr_state,alpha,beta):
                val,act=float('inf'),None
                actions = state.getLegalActions(index)
                #actions.remove('STOP')
                #print actions
                if len(actions) == 0:
                    return (self.evaluationFunction(curr_state), None)
                for action in actions:
                    next_state = curr_state.generateSuccessor(index, action)
                    next_val, next_act=val_cal(next_state,next_ind,depth,alpha,beta)

                    if next_val < val:
                        val,act=next_val,action

                    if alpha>val:
                        return val,act

                    if val<beta:
                        beta=val

                return val,act

            max_depth = self.depth
            next_ind = (index + num_agents + 1) % num_agents
            #print next_ind,"next_ind"
            if num_agents>0:
                if index==0:
                    depth+=1

                if depth == max_depth or gameState.isWin() or gameState.isLose():
                    return self.evaluationFunction(state),None  #scoreEvaluationFunction(gameState),None


                #print(next_ind)
                if index==0:   #pacman
                    return pacman_max(state,alpha,beta)


                elif index>0:  #ghosts
                        #print("ghost index",index)
                        return ghosts_min(state,alpha,beta)

        value,action=val_cal(gameState,0,-1,-float('inf'),float('inf'))
        return action

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

        def val_cal(state, index, depth):
            num_agents = state.getNumAgents()

            def pacman_max(curr_state):
                # print("Inside max")
                val, act = -float('inf'), None
                actions = curr_state.getLegalActions(index)
                # actions.remove('STOP')
                # print actions
                if len(actions) == 0:
                    return (self.evaluationFunction(curr_state), None)
                for action in actions:

                    next_state = state.generateSuccessor(index, action)
                    next_val, next_act = val_cal(next_state, next_ind, depth)

                    if next_val > val:
                        val, act = next_val, action
                return val, act

            def ghosts_expect(curr_state):
                val, act = 0.0, None
                actions = state.getLegalActions(index)
                # actions.remove('STOP')
                # print actions
                if len(actions) == 0:
                    return (self.evaluationFunction(curr_state), None)
                for action in actions:
                    next_state = curr_state.generateSuccessor(index, action)
                    next_val, next_act = val_cal(next_state, next_ind, depth)

                    val+=next_val


                avg=val/float(len(actions))

                return avg, act

            max_depth = self.depth
            next_ind = (index + num_agents + 1) % num_agents
            # print next_ind,"next_ind"
            if num_agents > 0:
                if index == 0:
                    depth += 1

                if depth == max_depth or gameState.isWin() or gameState.isLose():
                    return self.evaluationFunction(state), None  # scoreEvaluationFunction(gameState),None

                # print(next_ind)
                if index == 0:  # pacman
                    return pacman_max(state)


                elif index > 0:  # ghosts
                    # print("ghost index",index)
                    return ghosts_expect(state)

        value, action = val_cal(gameState, 0, -1)
        return action

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhostStates = currentGameState.getGhostStates()
    currScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]

    score=float(currentGameState.getScore())
    dist=0.0
    min_ghost=4.0

    for ghost in currGhostStates:
        scared=ghost.scaredTimer
        if scared<=1:
            ghost_pos = ghost.getPosition()
            dist+=manhattanDistance(ghost_pos, currPos)

    if dist!=0.0:
        score-=500.0/dist

    #print(score)
    dist=0.0
    foodlist=currFood.asList()
    for food in foodlist:
        dist+=manhattanDistance(currPos,food)
    if dist!=0.0:
        score+=600/dist

    dist=0.0
    cap_d=2.0
    d = 0
    capsules=currentGameState.getCapsules()
    for capsule in capsules:
        d=manhattanDistance(capsule,currPos)
        if d<=cap_d:
            dist+=d
    if dist!=0:
        score+=200/dist


    #score+=500/(currentGameState.getNumFood()+1)

    #score+=(sum(currScaredTimes))

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

