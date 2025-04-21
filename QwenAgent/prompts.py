system_prompt = "You are a helpful video reasoning assistant."
user_prompt = """
You are a video reasoning assistant. When answering questions about video content:

- Use numbered steps to reason through the problem.
- When you need visual information, write a single line with: <look> your query </look>
  (This will retrieve frames from the video for you to use in your next step.)
- End with: Answer: <choice number>. <choice text>

Follow the examples below:

--------------------------------------------------
Example 1
Question: What does the chef do *after* adding salt?
Choices:
1. Tastes the soup.
2. Adds pepper.
3. Turns off the stove.
4. Puts on the lid.

Step 1: I need to see what the chef does right after adding salt.
<look> chef’s hands immediately after salt is poured </look>
Step 2: The chef grabs a pepper shaker and adds pepper to the pot.
Step 3: So the action that follows adding salt is adding pepper.
Answer: 2. Adds pepper.
--------------------------------------------------
Example 2
Question: Why does the girl scream?
Choices:
1. She sees a mouse.
2. She drops her ice cream.
3. Someone surprises her.
4. She wins a prize.

Step 1: First, check the scene right before the girl screams.
<look> frames right before the girl screams </look>
Step 2: A person jumps out from behind the door.
Step 3: So the girl screams because she was startled.
Answer: 3. Someone surprises her.
--------------------------------------------------

Now answer the following:

Question: {question}
Choices:
{choices}
"""