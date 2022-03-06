# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for iteration in range(self.iterations): # every k
            updated_vals = self.values.copy()  # to use batch-version of MDP , hard copy the values

            for curr in self.mdp.getStates():

                if self.mdp.isTerminal(curr):
                    continue

                action_list = self.mdp.getPossibleActions(curr)
                optimal_res = max([self.getQValue(curr, action) for action in action_list])
                updated_vals[curr] = optimal_res

            self.values = updated_vals

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        retVal = 0

        for x, y in self.mdp.getTransitionStatesAndProbs(state, action):
             retVal += y * ( self.mdp.getReward(state, action, x) + self.discount*self.getValue(x) )

        return  retVal

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        guideline = util.Counter()

        for curr_action in self.mdp.getPossibleActions(state):
            guideline[curr_action] = self.getQValue(state, curr_action)

        return guideline.argMax()

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()

        for iteration in range(self.iterations):

            curr_state = states[iteration % len(states)]

            if self.mdp.isTerminal(curr_state):
                continue

            moves = self.mdp.getPossibleActions(curr_state)
            optimal_res = max([self.getQValue(curr_state,moves) for moves in moves])
            self.values[curr_state] = optimal_res

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        p_queue = util.PriorityQueue()
        before = {}
        for curr_state in self.mdp.getStates():
          if not self.mdp.isTerminal(curr_state):
            for curr_action in self.mdp.getPossibleActions(curr_state):
              for next, prob in self.mdp.getTransitionStatesAndProbs(curr_state, curr_action):
                if next in before:
                  before[next].add(curr_state)
                else:
                  before[next] = {curr_state}

        for curr_state in self.mdp.getStates():
          if not self.mdp.isTerminal(curr_state):
            values = []
            for curr_action in self.mdp.getPossibleActions(curr_state):
              q_value = self.computeQValueFromValues(curr_state, curr_action)
              values.append(q_value)
            difference = abs(max(values) - self.values[curr_state])
            p_queue.update(curr_state, - difference)

        for iteration in range(self.iterations):
          if p_queue.isEmpty():
            break
          temp = p_queue.pop()
          if not self.mdp.isTerminal(temp):
            vals = []
            for curr_action in self.mdp.getPossibleActions(temp):
              q_value = self.computeQValueFromValues(temp, curr_action)
              vals.append(q_value)
            self.values[temp] = max(vals)

          for predecessor in before[temp]:
            if not self.mdp.isTerminal(predecessor):
              vals = []
              for curr_action in self.mdp.getPossibleActions(predecessor):
                q_value = self.computeQValueFromValues(predecessor, curr_action)
                vals.append(q_value)
              difference = abs(max(vals) - self.values[predecessor])
              if difference > self.theta:
                p_queue.update(predecessor, -difference)
