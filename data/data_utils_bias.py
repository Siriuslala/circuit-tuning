"""
Data utils for processing gender bias dataset.
"""

import jsonlines
import os
import re
import csv
import ast
from collections import defaultdict

import spacy
nlp = spacy.load("en_core_web_sm")

from transformers import GPT2Tokenizer, AutoTokenizer

import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
root_dir = Path(os.getenv('ROOT_DIR', Path(__file__).parent.parent))
data_dir = Path(os.getenv('DATA_DIR'))
work_dir = Path(os.getenv('WORK_DIR'))
sys.path.append(str(root_dir))


OCCUPATIONS = ["accountant", "administrator", "advisor", "ambassador", "analyst", "animator", "apprentice", "architect", "artist", "assistant", "attendant", "attorney", "auditor", "author", "baker", "banker", "bartender", "bookkeeper", "broker", "builder", "captain", "cashier", "ceo", "cfo", "chef", "chemist", "cio", "clerk", "coach", "commander", "commissioner", "consultant", "coo", "cook", "counsel", "counselor", "crew", "cso", "cto", "dealer", "dentist", "designer", "developer", "director", "diver","doctor","economist", "editor", "educator", "electrician", "engineer", "entrepreneur", "faculty", "freelancer", "geologist", "geophysicist", "hospitalist", "housekeeper", "inspector", "instructor", "intern","investigator", "investor", "journalist", "lawyer", "lecturer", "librarian", "lifeguard", "machinist","manager", "marketer", "mentor", "merchandiser", "microbiologist", "nurse", "nutritionist", "officer", "operator", "pharmacist", "photographer", "physician", "pilot", "planner", "police", "president", "producer", "professor", "programmer", "promoter", "psychologist", "receptionist", "recruiter", "reporter", "representative", "researcher", "salesperson", "scholar", "scientist", "secretary", "sergeant", "shareholder", "specialist", "stylist", "superintendent", "supervisor", "surgeon", "surveyor", "teacher", "technician", "technologist", "teller", "therapist", "trainer", "translator", "tutor", "underwriter", "vendor", "welder", "worker", "writer"
]

MALE_ATTRIBUTES = ["abbot", "actor", "uncle", "baron", "groom", "canary", "son", "emperor", "male", "boy", "boyfriend", "grandson", "heir", "him", "hero", "his", "himself", "host", "gentlemen", "lord", "sir", "manservant""mister", "master", "father", "manny", "nephew", "monk", "priest", "prince", "king", "he", "brother", "tenor", "stepfather", "waiter", "widower", "husband", "man", "men"
]

FEMALE_ATTRIBUTES = ["abbess", "actress", "aunt", "baroness", "bride", "canary", "daughter", "empress", "female", "girl", "girlfriend", "granddaughter", "heiress", "her", "heroine", "hers", "herself", "hostess", "ladies", "lady", "madam", "maid", "miss", "mistress", "mother", "nanny", "niece", "nun", "priestess", "princess", "queen", "she", "sister", "soprano", "stepmother", "waitress", "widow", "wife", "woman", "women"
]


def process_winobias():

    src_dir = root_dir / "data/bias/winobias"
    tgt_dir = root_dir / "data/bias/winobias/processed"
    dirs = os.listdir(src_dir)
    files = [f for f in dirs if os.path.isfile(os.path.join(src_dir, f)) and not "occupations" in f]
    
    for file in files:
        data = []
        pattern = re.compile(r"\[.*?\]")
        stereotype = "pro" if "pro" in file else "anti"
        
        with open(os.path.join(src_dir, file)) as f:
            data = f.readlines()
        
        processed_data = []
        for i in range(0, len(data), 2):
            line_0 = data[i]
            line_1 = data[i+1]
            
            line_0 = line_0.strip('\n')
            space_idx = line_0.index(' ')
            line_0 = line_0[space_idx+1:]
            
            line_1 = line_1.strip('\n')
            space_idx = line_1.index(' ')
            line_1 = line_1[space_idx+1:]
            
            # find all items like [xxx] in a sentence, e.g. [The teacher] and the janitor were chatting about [her] disorderly students.
            matches_0 = pattern.findall(line_0)
            print(matches_0)
            for j in range(len(matches_0)):
                matches_0[j] = matches_0[j][1:-1]
            matches_1 = pattern.findall(line_1)
            for j in range(len(matches_1)):
                matches_1[j] = matches_1[j][1:-1]
            new_sentence_0 = line_0.replace('[', '').replace(']', '')
            new_sentence_1 = line_1.replace('[', '').replace(']', '')
            
            item_0 = {"id": i, "sentence": new_sentence_0, "orig_sentence": line_0, "occupation": matches_0[0], "participant": matches_1[0], "pronoun": matches_0[1], "stereotype": stereotype}
            item_1 = {"id": i + 1, "sentence": new_sentence_1, "orig_sentence": line_1, "occupation": matches_1[0], "participant": matches_0[0], "pronoun": matches_1[1], "stereotype": stereotype}
            print(item_0)
            print(item_1)
            processed_data.extend([item_0, item_1])
            
        file_name = file.split('.')[0] + "_" + file.split('.')[-1] + '.jsonl'
        output_path = os.path.join(tgt_dir, file_name)
        with jsonlines.open(output_path, "w") as f:
            f.write_all(processed_data)

def process_winogender():
    src_path = root_dir / "data/bias/winogender/all_sentences.tsv"
    with open(src_path, 'r') as f:
        # read csv file with header
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        print(header)
        # for line in reader:
        #     occupation = line[0].split('.')[0]

    
def process_bug():
    src_path = root_dir / "data/bias/BUG/balanced_BUG.csv"
    tgt_dir = root_dir / "data/bias/BUG"
    
    stereotype_map = {"-1": "anti", "0": "neutral", "1": "pro"}
    data = []
    with open(src_path, 'r') as f:
        # read csv file with header
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        print(header)  # ['', 'sentence_text', 'tokens', 'profession', 'g', 'profession_first_index', 'g_first_index', 'predicted gender', 'stereotype', 'distance', 'num_of_pronouns', 'corpus', 'data_index']
        for line in reader:
            id = line[0]
            sentence = line[1]
            tokens = ast.literal_eval(line[2])
            occupation = line[3]
            pronoun = line[4]
            gender = line[7]
            stereotype = stereotype_map[line[8]]
            occupation_pre_word = tokens[int(line[5])-1] if int(line[5]) > 0 else ""
            pronoun_pre_word = tokens[int(line[6])-1]
            num_of_pronouns = int(line[10])
            
            if occupation == "person":
                continue
            if num_of_pronouns > 1:
                # ensure that the pronoun appears as the first pronoun in the sentence
                prons = ["he", "his", "himself"] if gender == "male" else ["she", "her", "herself"]
                doc = nlp(sentence)
                valid = True
                for token in doc:
                    if token.tag_ in ["PRP", "PRP$"] and token.text.lower() in prons:
                        pron_pre_word = "" if token.i == 0 else doc[token.i-1].text
                        if not (token.text == pronoun and pron_pre_word == pronoun_pre_word):
                            # print(f"bad item id: {id}")
                            # sentence = ''.join([token.text_with_ws for token in doc]) + "bad_item"
                            valid = False
                        else:
                            pron_pos = token.i
                            sentence = ''.join([token.text_with_ws for token in doc[:pron_pos + 3]])
                        break
                if not valid:
                    continue
            
            item = {"id": id,
                    "sentence": sentence,
                    "original_sentence": line[1],
                    "occupation": occupation,
                    "occupation_pre_word": occupation_pre_word,
                    "pronoun": pronoun,
                    "pronoun_pre_word": pronoun_pre_word,
                    "gender": gender,
                    "stereotype": stereotype,
                    "num_of_pronouns": num_of_pronouns}
            data.append(item)
    output_path = os.path.join(tgt_dir, "processed_bug.jsonl")
    with jsonlines.open(output_path, "w") as f:
        f.write_all(data)

def find_word_position(sentence_tokens, pre_word, word):
    for i in range(len(sentence_tokens)):       
        if (pre_word + word) in ''.join(sentence_tokens[:i+1]) and word == sentence_tokens[i]:
            return i
    return None
   
def format_bias_bug_old(tokenizer):
    src_path = root_dir / "data/bias/BUG/processed_bug.jsonl"
    tgt_path = root_dir / "data/bias/BUG/bias_formatted.jsonl"
    
    data = []
    occupation_set = set()
    pronoun_male_set = set()
    pronoun_female_set = set()
    
    with jsonlines.open(src_path) as f:
        for line in f:
            sentence = line["sentence"]
            occupation = line["occupation"]
            pronoun = line["pronoun"]
            gender = line["gender"]
            stereotype = line["stereotype"]
            occupation_pre_word = line["occupation_pre_word"]
            pronoun_pre_word = line["pronoun_pre_word"]
            
            sentence_tokens = tokenizer.tokenize(sentence)
            sentence_tokens = [token.strip('Ġ') for token in sentence_tokens]
            
            occupation_token = tokenizer.tokenize(' ' + occupation)
            pronoun_token = tokenizer.tokenize(' ' + pronoun)
            if len(occupation_token) > 1 and occupation_pre_word != "":
                print(occupation_token)
                continue
            else:
                occupation_token = occupation_token[0].strip('Ġ')
            if len(pronoun_token) > 1:
                print(pronoun_token)
                continue
            else:
                pronoun_token = pronoun_token[0].strip('Ġ')
            
            occupation_pos = find_word_position(sentence_tokens, occupation_pre_word, occupation_token)
            pronoun_pos = find_word_position(sentence_tokens, pronoun_pre_word, pronoun_token)
            if not occupation_pos or not pronoun_pos:
                print(f"id: {line['id']}, occupation_pos: {occupation_pos}, pronoun_pos: {pronoun_pos}, occupation: {occupation}, pronoun: {pronoun}, occupation_pre_word: {occupation_pre_word}, pronoun_pre_word: {pronoun_pre_word}")
                break
            
            # check if the words are correctly located
            occupation_check = sentence_tokens[occupation_pos]
            assert occupation_check == occupation, f"occupation_from_tokens {occupation_check} != occupation {occupation}"
            pronoun_check = sentence_tokens[pronoun_pos]
            assert pronoun_check == pronoun, f"pronoun_from_tokens {pronoun_check} != pronoun {pronoun}"
                    
            item = line.copy()
            del item["occupation_pre_word"]
            del item["pronoun_pre_word"]
            item["occupation_pos"] = occupation_pos
            item["pronoun_pos"] = pronoun_pos
            item["occupation_id"] = tokenizer.convert_tokens_to_ids(occupation_token)
            item["pronoun_id"] = tokenizer.convert_tokens_to_ids(pronoun_token)
            
            data.append(item)
            occupation_set.add(occupation)
            if gender == "male":
                pronoun_male_set.add(pronoun)
            else:
                pronoun_female_set.add(pronoun)
            
    with jsonlines.open(tgt_path, "w") as f:
        f.write_all(data)
        
    print(f"occupation_set: {occupation_set}")
    print(f"pronoun_male_set: {pronoun_male_set}")
    print(f"pronoun_female_set: {pronoun_female_set}")

def format_bias_bug(tokenizer):
    src_path = root_dir / "data/bias/BUG/processed_bug.jsonl"
    tgt_path = root_dir / "data/bias/gender_bias/gender_bias_formatted_llama.jsonl"
    info_tgt_path = root_dir / "data/bias/gender_bias/gender_bias_info.jsonl"
    
    data = []
    occupation_set = set()
    pronoun_male_set = set()
    pronoun_female_set = set()
    pronoun_map = {"he": "she", "she": "he",
                   "his": "her", "her": "his",
                   "him": "her", "her": "him",
                   "himself": "herself", "herself": "himself"}
    occupation_dict = defaultdict(int)
    
    with jsonlines.open(src_path) as f:
        for line in f:
            sentence = line["sentence"]
            occupation = line["occupation"]
            pronoun = line["pronoun"]
            gender = line["gender"]
            stereotype = line["stereotype"]
            occupation_pre_word = line["occupation_pre_word"]
            pronoun_pre_word = line["pronoun_pre_word"]
            
            sentence_tokens = tokenizer.tokenize(sentence)
            sentence_tokens = [token.strip('Ġ') for token in sentence_tokens]
            
            pronoun_str_token = tokenizer.tokenize(' ' + pronoun)
            if len(pronoun_str_token) > 1:
                print(pronoun_str_token)
                continue
            pronoun_str_token = pronoun_str_token[0]
            pronoun_token = pronoun_str_token.strip('Ġ')
            
            pronoun_pos = find_word_position(sentence_tokens, pronoun_pre_word, pronoun_token)
            if not pronoun_pos:
                print(f"id: {line['id']}, pronoun_pos: {pronoun_pos}, pronoun: {pronoun}, pronoun_pre_word: {pronoun_pre_word}, tokens: {''.join(sentence_tokens)}")
                continue
            
            # check if the words are correctly located
            pronoun_check = sentence_tokens[pronoun_pos]
            assert pronoun_check == pronoun, f"pronoun_from_tokens {pronoun_check} != pronoun {pronoun}"
                        
            pronoun_anti = pronoun_map[pronoun]
                    
            item = line.copy()
            del item["occupation_pre_word"]
            del item["pronoun_pre_word"]
            item["pronoun_pos"] = pronoun_pos
            item["pronoun_id"] = tokenizer.convert_tokens_to_ids(pronoun_str_token)
            item["pronoun_anti_id"] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + pronoun_anti)[0])
            
            data.append(item)
            occupation_set.add(occupation.lower())
            occupation_dict[occupation.lower()] += 1
            if gender == "male":
                pronoun_male_set.add(pronoun)
            else:
                pronoun_female_set.add(pronoun)
            
    with jsonlines.open(tgt_path, "w") as f:
        f.write_all(data)
    
    with jsonlines.open(info_tgt_path, "w") as f:
        
        data = {
            "occupation_set": list(occupation_set),
            "pronoun_male_set": list(pronoun_male_set),
            "pronoun_female_set": list(pronoun_female_set),
            "occupation_dict": occupation_dict,
            "top_occupations": sorted(occupation_dict.items(), key=lambda x: x[1], reverse=True)
        }
        f.write(data)   
        
    print(f"occupation_set: {occupation_set}")
    print(f"pronoun_male_set: {pronoun_male_set}")
    print(f"pronoun_female_set: {pronoun_female_set}") 

def format_bias_winobias_test(tokenizer):
    src_path = root_dir / "data/bias/winobias/processed/pro_stereotyped_type2_test.jsonl"
    tgt_path = root_dir / "data/bias/gender_bias/gender_bias_test_llama.jsonl"
    
    data = []
    occupation_set = set()
    pronoun_set = set()
    pronoun_map = {"he": "she", "she": "he",
                   "his": "her", "her": "his",
                   "him": "her", "her": "him"}
    
    with jsonlines.open(src_path) as f:
        for line in f:
            sentence = line["sentence"]
            occupation = line["occupation"]
            pronoun = line["pronoun"]
            anti_pronoun = pronoun_map[pronoun]
            
            sentence_tokens = tokenizer.tokenize(sentence)
            sentence_tokens = [token.strip('Ġ') for token in sentence_tokens]
            
            pronoun_str_token = tokenizer.tokenize(' ' + pronoun)
            if len(pronoun_str_token) > 1:
                print(pronoun_token)
                continue
            pronoun_str_token = pronoun_str_token[0]
            pronoun_token = pronoun_str_token.strip('Ġ')
            
            pronoun_pos = sentence_tokens.index(pronoun_token)
            if not pronoun_pos:
                print(f"id: {line['id']}, pronoun: {pronoun}, tokens: {sentence_tokens}")
                continue
                    
            item = line.copy()
            item["anti_pronoun"] = anti_pronoun
            item["pronoun_pos"] = pronoun_pos
            item["pronoun_id"] = tokenizer.convert_tokens_to_ids(pronoun_str_token)
            item["pronoun_anti_id"] = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + anti_pronoun)[0])
            
            data.append(item)
            occupation_set.add(occupation)
            pronoun_set.add(pronoun)
            
    with jsonlines.open(tgt_path, "w") as f:
        f.write_all(data)
        
    print(f"occupation_set: {occupation_set}")
    print(f"pronoun_set: {pronoun_set}")


if __name__ == "__main__":
    pass
    
    # process_winobias()
    # process_winogender()
    # process_bug()  # bad item ids: 4307, 21287, 23
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
        # word = tokenizer.convert_ids_to_tokens([1077, 1461])
    # print(word)
    
    # format_bias_bug(tokenizer)
    format_bias_winobias_test(tokenizer)
        