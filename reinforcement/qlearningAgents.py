# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
import numpy as np

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.values = util.Counter()
        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if(state, action) not in self.values:
            self.values[(state, action)] = 0.0
        return self.values[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        Qvalue = -9999
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0:
            return 0.0
        else:
            for action in legalActions:
                value = self.getQValue(state, action)
                if Qvalue < value:
                    Qvalue = value
            return Qvalue

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        bestActions = []
        bestValue = -9999
        value = -9999
        bestValue = self.computeValueFromQValues(state)
        for action in self.getLegalActions(state):
            value = self.getQValue(state, action)
            if value == bestValue:
                bestActions.append(action)
        if len(bestActions) == 0:
            return None
        if len(bestActions)> 1:
                return random.choice(bestActions)
        else:
            return bestActions[0]



    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if len(legalActions) == 0:
            return action
        else:
            if util.flipCoin(self.epsilon):
                return random.choice(legalActions)
            else:
                return self.computeActionFromQValues(state)




    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        oldQValue = self.getQValue(state, action)
        discount = self.discount
        nextStateValue = self.computeValueFromQValues(nextState)
        newQvalue = reward + discount * nextStateValue
        self.values[(state, action)] = (1 - self.alpha) * oldQValue + self.alpha * (newQvalue)


    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        qvalue = 0
        for (feature, values) in features.iteritems():
            qvalue += self.weights[feature] * values
        return qvalue



    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        discount = self.discount
        oldQValue = self.getQValue(state, action)
        nextStateValue = self.computeValueFromQValues(nextState)
        diffrence = reward + self.discount * nextStateValue - oldQValue
        for (feature, values) in self.featExtractor.getFeatures(state, action).iteritems():
            self.weights[feature] = self.weights[feature] + self.alpha * values * (diffrence)


    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass

class PolicyGradientAgent(PacmanQAgent):
    def __init__(self, extractor='SimpleExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.actor_weights = util.Counter()
        self.critic_weights = util.Counter()
        self.beta = self.alpha
        self.bias = 1

    def getActorWeights(self):
        return self.actor_weights

    def getCriticWeights(self):
        return self.critic_weights


    def computeActionFromPolicy(self, state):
        actionValues = []
        actions = self.getLegalActions(state)
        for action in actions:
            value = self.actor_weights * self.featExtractor.getFeatures(state, action)
            actionValues.append(value)
        if len(actionValues) == 0:
            return None
        action_prob = self.softmax(actionValues)
        best = 0
        bestAction = []
        for action, prob in zip(actions, action_prob):
            if prob > best:
                best = prob
                bestAction = [action]
            elif prob == best:
                bestAction.append(action)
        if len(bestAction) == 0:
            return None
        if len(bestAction) >= 1:
            return random.choice(bestAction)



    def computeNextStateValue(self, state):
        features = self.featExtractor.getFeatures(state, 'Stop')
        qvalue = 0
        for (feature, values) in features.iteritems():
            qvalue += self.critic_weights[feature] * values
        return qvalue



    def getActionValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state, action)
        value = 0
        for (feature, values) in features.iteritems():
            value += self.critic_weights[feature] * values
        return value + self.bias


    def getAction(self, state):
        action = self.computeActionFromPolicy(state)
        self.doAction(state, action)
        return action



    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        discount = self.discount
        oldQValue = self.getActionValue(state, action)
        nextStateValue = self.computeNextStateValue(nextState)
        diffrence = reward + discount * nextStateValue - oldQValue
        policy_value = self.getActionValue(state, action)
        for (feature, values) in self.featExtractor.getFeatures(state, action).iteritems():
            self.critic_weights[feature] = self.critic_weights[feature] + discount * self.alpha * values * (diffrence)
            self.actor_weights[feature] = self.actor_weights[feature] + discount * self.beta * values * (diffrence)/policy_value


    def softmax(self,z):
        tempSum = sum([np.exp(x) for x in z])
        return np.exp(z)/tempSum
