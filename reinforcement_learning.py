import random
from collections import defaultdict

class ValueIterationSolver:
    """
    Implements the Value Iteration algorithm for solving Markov Decision Processes (MDPs).
    """
    def __init__(self, mdp, discount_rate=0.9, iterations=100):
        self.mdp = mdp
        self.discount_rate = discount_rate
        self.iterations = iterations
        self.values = {state: 0.0 for state in mdp.getStates()}

        for _ in range(self.iterations):
            new_values = {}
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    new_values[state] = 0.0
                else:
                    action = self.get_policy(state)
                    if action:
                        new_values[state] = self.get_q_value(state, action)
                    else:
                        new_values[state] = 0.0
            self.values = new_values

    def get_q_value(self, state, action):
        q_val = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            reward = self.mdp.getReward(state, action, next_state)
            q_val += prob * (reward + self.discount_rate * self.values.get(next_state, 0.0))
        return q_val

    def get_policy(self, state):
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        if not actions:
            return None
        return max(actions, key=lambda a: self.get_q_value(state, a))

    def get_value(self, state):
        return self.values.get(state, 0.0)

class TabularQLearner:
    """
    Standard Q-Learning agent utilizing an epsilon-greedy strategy.
    """
    def __init__(self, alpha=0.5, epsilon=0.1, discount=0.9, actions_func=None):
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.get_legal_actions = actions_func
        self.q_values = defaultdict(float)

    def get_q_value(self, state, action):
        return self.q_values[(state, action)]

    def compute_value_from_q_values(self, state):
        actions = self.get_legal_actions(state)
        return max([self.get_q_value(state, a) for a in actions]) if actions else 0.0

    def compute_action_from_q_values(self, state):
        actions = self.get_legal_actions(state)
        if not actions: return None
        return max(actions, key=lambda a: self.get_q_value(state, a))

    def get_action(self, state):
        actions = self.get_legal_actions(state)
        if not actions: return None
        if random.random() < self.epsilon:
            return random.choice(actions)
        return self.compute_action_from_q_values(state)

    def update(self, state, action, next_state, reward):
        old_q = self.get_q_value(state, action)
        sample = reward + self.discount * self.compute_value_from_q_values(next_state)
        self.q_values[(state, action)] = (1 - self.alpha) * old_q + self.alpha * sample

class FeatureBasedQLearner(TabularQLearner):
    """
    Approximate Q-Learning agent using linear function approximation to generalize across states.
    """
    def __init__(self, feature_extractor, **kwargs):
        super().__init__(**kwargs)
        self.feat_extractor = feature_extractor
        self.weights = defaultdict(float)

    def get_q_value(self, state, action):
        features = self.feat_extractor(state, action)
        return sum(self.weights[f] * val for f, val in features.items())

    def update(self, state, action, next_state, reward):
        diff = (reward + self.discount * self.compute_value_from_q_values(next_state)) - self.get_q_value(state, action)
        features = self.feat_extractor(state, action)
        for f, val in features.items():
            self.weights[f] += self.alpha * diff * val
