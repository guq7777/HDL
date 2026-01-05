from tqdm import tqdm

def extract_content(data):
    return [message["content"] for message in data["messages"] if message["role"] == "assistant"]


def extract_content_and_old_content(data):
    result = []
    for message in data["messages"]:
        if message["role"] == "assistant":
            try:
                if 'content' in message:
                    content = message['content']
                else:
                    content = None
                if 'old_content' in message:
                    old_content = message['old_content']
                else:
                    old_content = None
                result.append((content, old_content))
            except KeyError as e:
                raise KeyError(f"Missing key: {e} in message: {message}")
    return result


def process_sharegpt_data(data):
    candidate = {}
    references = {}
    for idx, entry in enumerate(tqdm(data)):
        key = f'res{idx + 1}'
        if entry[0] != None and entry[1] != None:
            candidate[key] = [entry[0]]
            references[key] = [entry[1]]

    return candidate, references



def process_data(data, key_new_content, key_old_content):
    candidate = {}
    references = {}
    for idx, entry in enumerate(data):
        key = f'res{idx + 1}'
        if key_new_content in entry and key_old_content in entry:
            if entry[key_new_content] != 'None' and entry[key_new_content] is not None:
                candidate[key] = [entry[key_new_content]]
                references[key] = [entry[key_old_content]]

    return candidate, references

