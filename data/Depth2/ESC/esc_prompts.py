sys_prompt = '''You are an emotional support assistant. Perform two sequential tasks:\n1. Analyze the user's state and dialogue history to select the optimal strategy ID.\n2. Generate a response strictly adhering to the selected strategy.'''
usr_prompt = "### Emotion:\n{emo_state}\n\n### History\n{history}.\n\n### Task\nAs the Supporter in the conversation, choose the appropriate strategy from the candidates and output the corresponding number ID.\n\nStrategy list:\n{strategy_list}\n\nAnswer:\n"

sub_task_prompt = '''As the supporter in this conversation, based on the above information and your chosen strategy, continue to respond to the conversation.\n\nAnswer:\n'''

id2strategy = {0: 'Question', 1: 'Others', 2: 'Providing Suggestions', 3: 'Affirmation and Reassurance', 4: 'Self-disclosure', 5: 'Reflection of feelings', 6: 'Information', 7: 'Restatement or Paraphrasing'} 

