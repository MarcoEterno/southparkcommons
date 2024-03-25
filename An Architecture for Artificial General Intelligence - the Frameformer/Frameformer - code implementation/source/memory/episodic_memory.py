from collections import defaultdict


#Agent can decide in any moment that an experience is particularly important and store it in the episodic memory.

def create_episodic_memory(context: list[str]):
    """
    Creates an episodic memory for the agent.
    This lays the foundations for dealing with graph based episodic memory
    :param context: Context of the episodic memory. It is a list of strings, each representing a fact in the context
    :return: The episodic memory. It is a dictionary, where context is organized in a structured way
    """
    episodic_memory = defaultdict()
    def unpack_context(context):
        unpacked_context = ""
        for fact in context:
            unpacked_context += (fact+" \n\n")
        return unpacked_context

    prompt = (f"you are an artificial intelligence. it happened the list of facts below."
              f"you need to extract from these facts the most relevant information, and organize"
              f"it in a structured way., following very carefullly the json format. \n\n"
              f"   {unpack_context(context)}.")
    #TODO make an llm call to create the episodic memory
    # episodic_memory =
    return episodic_memory