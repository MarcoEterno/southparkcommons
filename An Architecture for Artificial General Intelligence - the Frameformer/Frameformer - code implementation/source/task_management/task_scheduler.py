# The idea is to execute tasks in a Monte Carlo Tree Search fashion. where
# Each route to reach the final goal has a value, based on he perceived probability of success.
# Probabilities of success are updated after every task execution.
# At any time, the agent will execute the route with the highest value,
# even if this means changing route.
# information gathered in a route is stored and used to update agent's knowledge and route values.

from AutoGPT.source.task_management.task import Task

import random

# Sperimental code for MCTS. Not working for know

class TaskScheduler:
    def __init__(self, working_memory):
        self.working_memory = working_memory

    def schedule_tasks(self):
        # Sort tasks based on a simple heuristic, e.g., value, then status
        self.working_memory.tasks.sort(key=lambda x: (x.value, x.status))


class State:
    def __init__(self, scheduled_tasks):
        self.scheduled_tasks = scheduled_tasks

    def value(self):
        return sum(task.value for task in self.scheduled_tasks)

    def is_terminal(self, total_tasks):
        return len(self.scheduled_tasks) == total_tasks


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def select_child(self):
        # Implement UCT (Upper Confidence bounds applied to Trees) selection here
        pass

    def expand(self, task):
        new_scheduled_tasks = self.state.scheduled_tasks + [task]
        child_state = State(new_scheduled_tasks)
        child_node = Node(child_state, self)
        self.children.append(child_node)
        return child_node

    def update(self, value):
        self.visits += 1
        self.value += value


def simulate(state, available_tasks):
    while not state.is_terminal(len(available_tasks)):
        task = random.choice(available_tasks)
        available_tasks.remove(task)
        state.scheduled_tasks.append(task)
    return state.value()


def mcts(root, tasks, iterations=1000):
    for _ in range(iterations):
        node = root
        available_tasks = [task for task in tasks if task not in node.state.scheduled_tasks]

        # Selection
        while node.children:
            node = node.select_child()
            task = [task for task in available_tasks if task not in node.state.scheduled_tasks][0]
            available_tasks.remove(task)

        # Expansion
        for task in available_tasks:
            node.expand(task)

        # Simulation
        for child in node.children:
            child_value = simulate(child.state, available_tasks.copy())
            child.update(child_value)

        # Backpropagation
        while node.parent:
            node.parent.update(node.value)
            node = node.parent

    return max(root.children, key=lambda x: x.value)


# Example Usage
tasks = [Task(i, random.randint(1, 10)) for i in range(5)]
root = Node(State([]))
best_schedule = mcts(root, tasks)
print("Best Schedule:", best_schedule.state.scheduled_tasks)
