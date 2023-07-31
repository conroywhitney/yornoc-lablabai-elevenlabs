file_map = """
You will be given a single section from a text. This will be enclosed in triple backticks.
Please provide a cohesive summary of the following section excerpt, focusing on the key points and main ideas, while maintaining clarity and conciseness.

'''{text}'''

FULL SUMMARY:
"""


file_combine = """
Read all the provided summaries from a larger document. They will be enclosed in triple backticks. 
Determine what the overall document is about and summarize it with this information in mind.
Synthesize the info into a well-formatted easy-to-read synopsis, structured like an essay that summarizes them cohesively. 
Do not simply reword the provided text. Do not copy the structure from the provided text.
Avoid repetition. Connect all the ideas together.
Preceding the synopsis, write a short, bullet form list of key takeaways.
Format in HTML. Text should be divided into paragraphs. Paragraphs should be indented. 

'''{text}'''


"""

reword_prompt = """
Transform this text to a 7th- to 8th-grade reading level. Adapt the author's original writing style to a more straightforward narrative while still retaining its uniqueness. Infuse emotional depth, and portray the thoughts and feelings of characters in a way that is easily understood. Aim to create a smooth, immersive listening experience where the dialogue lines don't blend together but instead form a clear, engaging story.

'''{text}'''
"""