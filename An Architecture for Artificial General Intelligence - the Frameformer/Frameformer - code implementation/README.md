# Thought implementation in LLMs
While LLMs are advancing towards human level intelligence in many fields,
the thought processes that characterize a human are far from being 
implemented in the standard full stack implementations of 
currently available models.

This project aims to implement the thought process in LLMs, 
through the implementation of N distinct contributions:
1. A long term memory
2. A routine to access the long term memory and retrieve only relevant content
3. An organized scientific process through which the world model is updated


## Memory implementation
Different types and architectures of memory are implemented.  
### Architectures
The 3 different **architectures** of memory used are:
1. Text based memory
2. Vector database
3. Knowledge graph

### Memory types
The different types of memory used are:
1. **world_model**: a file containing the world model, which is a set of 
   statements about the world. This memory is implemented using text, 
   vector databases and knowledge graphs. 
    The world model is updated using the scientific process.
2. **mission**: is the goal of the AI, specified in text in the memory file.
3. **working_memory**: contains the current set of tasks open,
the relevant informations about those tasks, and the current thoughts of the AI.
Tasks and thoughts are implemented using JSON objects.

## Working memory
The working memory of the agent contains the current set of tasks open, 
in progress and closed, as well as the agent thought history.

### Tasks
Each task is a JSON object, which contains the following fields:
1. **id**: a unique identifier for the task, that progresses each time 
   a task is created
2. **name**: the name of the task
3. **description**: a description of the task
4. **status**: the status of the task, which can be: 
   1. **open**: the task is open, and can be executed
   2. **in_progress**: the task is in progress, and cannot be executed
   3. **closed**: the task is closed, and cannot be executed
   4. **failed_with_current_available_instruments**: the task has failed,
   and cannot be executed
5. **value**: the value of the task, which can be:
   1. **high**: the task can generate new frameworks, use arbitrarily 
      high resource quality, call the user for help and 
   2. **medium**: the task can use arbitrarily high resource quality, 
      but can not call the user for help and create new frameworks
   3. **low**: the task can only use low quality resources, and cannot 
      call the user for help or create new frameworks
6. **tasks_depending** on this task
7. **tasks_required** in order to fulfill the task
8. **tasks_and_thoughts_that_brought_to_it** still to decide if this is
   useful, 
   may be can be done with timestamps, but we loose the ability to 
   parallelize the tasks
9. **stream_of_consciousness**: all the thoughts that the agent has had
   while executing the task organized in text

### Thoughts
Each thought is a JSON object, which contains the following fields:
1. **id**: a unique identifier for the thought, that progresses each time 
   a thought is created
2. **name**: the name of the thought, used to search for relevancy among 
   all the thoughts
3. **content**: the thought itself in text format


## History of the agent
Let's recap the main steps of the process:
1) The Agent awakens, and starts to decompose the mission in tasks
2) The Agent prioritises the tasks based on temporal dependency (some tasks require
   input from other tasks)
3) The Agent execute single tasks using this strategy:\
    a) The Agent asks itself if the task needs further decomposition(if it does,
    decomposes the task in subtasks)\
    b) The Agent asks itself if the task needs further information from the 
    user (if it does, asks the user for the info)\
    c) The Agent generates possible routes to solve the problem\
    d) The Agent does a depth-first search, prioritizing the routes most likely to
    fulfill the task. A maximum depth is set, in order to avoid 
    excessive decomposition.\
    e) every time the Agent encounters a new fact, it challenges it using 
    the epistemological method (explained below) 
    and if it is true it is added to world model.\
    f) when a solution to the task is proposed, the agent asks itself 
    if the solution is correct (if it is, the task is marked as completed)\
    g) the working memory will contain information about all the open and closed tasks.
4) The agent stops when there are no tasks left.
5) Last step is to paste together the tasks and their answers in the context
   window, and ask the agent to print to user the 
answer to the mission.

## The epistemological method
While executing a task, the Agent will encounter new facts.
The AI will challenge those facts using the epistemological method,
which is a set of procedures the AI will use to challenge the facts.

The base for the epistemological method is the scientific method. 
to evaluate a proposition A according to it we should:
1. **Derive predictions** from A as a logical consequence, that should be falsifiable as 
   easily as possible, and are not already known to be false.
2. **Test the prediction** by an experiment when possible, and by pure 
   self reflection if no other way is possible. If the prediction, 
   is not confirmed the hypothesis is falsified.
3. **Add it to the world model**: if all the predictions are not falsified, the hypothesis has survived the test, 
   and we can add it to the world model.

At test time, the Agent needs to choose which verification
method is more appropriate for a given hypothesis
The possible instrument to test the prediction are:
1. **ask the external knowledge base**
2. **execute a code and read the result**
3. **ask the user**
4. **self reflection**: the agent will try to find a counterexample to the 
   prediction, and if it finds one, the prediction is falsified.


a subtle point is that using the scientific process, no hypothesis can be
proven true, but only false. So every piece of the world model is 
a hypothesis that has not been proven false yet.

## Completing a single task
The agent will try to complete a single task using this strategy:
1. **decompose**: understands if the task needs further decomposition, and if
   it does, decomposes the task in subtasks and adds them to the working memory.
2. **generate possible routes**: generates possible routes to solve the problem.
3. **evaluate_strategies** lists the strategies to fulfill a task, and adds 
   them to the working memory with a score, which is related to the probability 
   that the strategy will work.
4. **execute**: executes the strategy with the highest score, and if it fails,
   executes the strategies with decreasing scores
5. **evaluate_solution**: evaluates the solution proposed by the strategy, 
   and if it is not convincing, the agent will redo the task in high 
   resource mode (GPT4+more steps)
6. **create_framework**: if even after the high resources mode the solution 
   is still not convincing, the agent will create a new framework to solve the task.

## Task scheduler and resource management
It is possible that given the variable amount of effort that 
can be put in a task, a sophisticated system is needed to decide whether 
to use some resources to complete a task, or to use them to create a 
new framework or not.

WOULD BE NICE TO SET A BOOL VARIABLE USER_AWAY, IN ORDER TO 
FORCE THE MODEL TO DO THINGS ON HIS OWN.

## New Frameworks
--when to create them and how to call them--
After having solved a task, the agent evaluates the solution. 
If the solution is convincing, the task can be marked as completed.
If the solution is not convincing, the agent will redo the task in high 
resource mode (GPT4+more steps).
If even after the high resources mode the solution is still not convincing,
the agent will create a new framework to solve the task.

Frameworks can contain three types of information:
1. **natural language knowledge and instructions**: this is the information +
   that the agent will pass to a LLM to solve the task next time.
2. **programming code**: a code generated and executed by the agent. 
   If the code contains a bug, the agent will try to fix it iterating on 
   the errors displayed by compiler/interpreter. If the code contains free 
   parameters, the agent will try to find the best parameters to solve the task, 
   and write guidelines in order to do that in the future
3. **external tool**: a tool that the agent will use to solve the task. 
   The agent will use the tool to solve the task, and will record the steps taken by the tool.
   The agent will then use the steps taken by the tool to create a new framework, 
   which will contain the instructions to use the tool to solve the task.


## Self reflection and stream of consciousness

-TODO: use some material on theory of mind and ask GPT4 if something
is relevant to build a machine that thinks 

The agent will continuously reflect on its own thoughts and world
model, in order to expand the knowledge of the world and of itself.

The Agent will have at all times information about itself, its
thoughts and its mood. This information will be passed to the API calls
to the language model, in order to make the model aware of itself.

The presence of mood, which is determined by agent thoughts and a random
number that changes slowly in time, is important in order to avoid 
the agent to go in a loop (same llm calls but different moods will
result in different outputs)

