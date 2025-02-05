"""
Data processing.
"""

import pandas as pd
from huggingface_hub import snapshot_download
from langdetect import detect
import spacy
from tqdm import tqdm

import os
import jsonlines
import string

from transformer_lens import HookedTransformer

API_KEY = "f81e2c2bebbba47fe24a1372dcebec19.kDnJrGIFsdmfkath"


nlp = spacy.load("en_core_web_sm")

def is_english(sentence):
    try:
        language = detect(sentence)
        if language == 'en':
            return True
        else:
            return False
    except:
        return False
    
def is_complete(sentence):
    if sentence[-1] in ['.', '!', '?']:
        return True
    else:
        return False

def is_present_tense(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.tag_ == "VBP" or token.tag_ == "VBZ":  # VBP: base form, VBZ: 3rd person sing. present
            return True
    return False

def get_verb_third_person(verb):
    """
    VBP -> VBZ
    """
    # special cases
    if verb in ["am", "'m"]:
        return "is" if verb == "am" else "'s"
    if verb in ["are", "'re"]:
        return "is" if verb == "are" else "'s"
    if verb in ["have", "'ve"]:
        return "has"
    
    # end with 'y'
    vowels = ['a', 'e', 'i', 'o', 'u']
    if verb[-1] == 'y':
        if verb[-2] in vowels:
            return verb + "s"
        else:
            return verb[:-1] + "ies"
    
    # s, x, sh, ch, o, ss
    suffixes = ['s', 'x', 'sh', 'ch', 'o', 'ss']
    if verb[-1] in suffixes:
        return verb + "es"
    
    # normal case
    return verb + "s"

def check_verbs(sentence):
    sentence = sentence.replace("’", "'")
    doc = nlp(sentence)
    for token in doc:
        print(token.text, token.tag_)
        if token.tag_ == "VBP":  # base form
            pass
        elif token.tag_ == "VBZ":  # 3rd person sing. present
            pass
        else:
            pass
    return sentence

def annotate_verbs(sentence):
    doc = nlp(sentence)
    verbs = []
    corr_text = ""
    for token in doc:
        # find verbs in ["VBP", "VBZ"] and flip them
        if token.tag_ in ["VBP", "VBZ"]:
            # bad cases
            # 1. France's _defeat_
            conditions = [token.i - 1 >= 0, token.i + 1 < len(doc)]
            if all(conditions) and doc[token.i - 1].text == "_" and doc[token.i + 1].text == "_":
                corr_text += (token.text + ' ')
                continue
            # 2. ... you'.
            stripped = token.text.translate(str.maketrans('', '', string.punctuation))
            if not stripped:  # contains only punctuation
                corr_text += (token.text + ' ')
                continue
            # 3. must-sees (-verb)
            if token.i - 1 >= 0 and "-" in doc[token.i - 1].text:
                corr_text += (token.text + ' ')
                continue
            # 4. criss-cross (verb-)
            if token.i + 1 < len(doc) and "-" in doc[token.i + 1].text:
                corr_text += (token.text + ' ')
                continue
            
            verb_flip = ""
            if token.tag_ == "VBP":  # base form
                verb_third_person = get_verb_third_person(token.text)
                verb_flip = verb_third_person
            elif token.tag_ == "VBZ":  # 3rd person sing. present
                verb_lemma = token.lemma_
                if verb_lemma == "be":
                    verb_flip = "are" if token.text == "is" else "'re"
                else:
                    verb_flip = verb_lemma
            # n't      
            neg = False
            if token.i < len(doc) - 1 and doc[token.i + 1].text == "n't":
                neg = True
            # pre_word
            if token.i == 0:  # no pre_word, verb as the beginning
                pre_word = None
            # elif token.i - 1 == 0:  # pre_word as the beginning (row 33)
            #     pre_word = doc[token.i - 1].text
            # elif doc[token.i - 2].text == "-":  #  the pre-pre_word is '-' (row 50)
            #     pre_word = '-' + doc[token.i - 1].text
            
            # elif not doc[token.i - 1].text.translate(str.maketrans('', '', string.punctuation)):
            #     pre_word = ' ' + doc[token.i - 2].text + ' ' + doc[token.i - 1].text
            elif doc[token.i - 1].tag_ in ["VBP", "VBZ"] and len(verbs) > 0:  # pre_word is also a verb and has been changed (row 143)
                pre_word = [doc[token.i - 1].text , verbs[-1]["flip"]]
            else:
                pre_word = doc[token.i - 1].text
                
            verbs.append({"id": len(verbs), 
                          "clean": token.text, 
                          "flip": verb_flip, 
                          "tag": token.tag_,
                          "pre_word": pre_word,  # used for locate the verb in the tokens from the model's tokenizer
                          "neg": neg
                        })  
                          
            # replace the verb with the annotated one in corr_text
            if "'" in verb_flip:
                corr_text = corr_text[:-1] + verb_flip + ' '
            else:
                corr_text += (verb_flip + ' ')
        else:  # no-verb token
            del_punc = token.text.translate(str.maketrans('', '', string.punctuation))
            if not del_punc:  # token contains only punctuation  ([Can you ?] v.s. [Can you?])
                if (doc[token.i - 1].text + token.text) in sentence:  # no space between the word and the punctuation
                    if token.i == len(doc) - 1:  # end of the sentence
                        corr_text = corr_text[:-1] + token.text
                    else:
                        corr_text = corr_text[:-1] + token.text + ' '
                else:
                    if token.i == len(doc) - 1:  # end of the sentence
                        corr_text += token.text
                    else:
                        corr_text += (token.text + ' ')
            else:  # word token
                if token.i == 0:
                    corr_text += (token.text + ' ')
                elif token.i == len(doc) - 1:  # e.g. NNP (proper noun)
                    corr_text += token.text
                else:
                    if token.text == ' ':  # # e.g. St Margaret's Church (  MAP   GOOGLE MAP ) ;
                        corr_text += (token.text + ' ')
                    elif (doc[token.i - 1].text + token.text) in sentence:  # e.g. everyone's = everyone + 's
                        corr_text = corr_text[:-1] + token.text + ' '
                    else:
                        corr_text += (token.text + ' ')
                        
    corr_text = corr_text.replace(" n't", "n't")
    
    return verbs, corr_text

def get_verb_info(clean_tokens, corr_tokens, verbs, tokenizer):
        """
        Return:
        the position of the verb in the sentence
        the id of the clean verb in the vocab
        the id of the corr verb in the vocab
        """
        # verb: id, clean, flip, tag, pre_word, neg
        clean_verb_pos = []
        corr_verb_pos = []
        clean_verb_ids = []
        corr_verb_ids = []
        clean_verbs = []
        corr_verbs = []
        
        def find_verb_position(sentence_tokens, verb, pre_word):
            
            # print(x, y)
            try:
                sentence_tokens.index(verb)
            except:
                return None

            for i in range(len(sentence_tokens)):       
                if (pre_word + verb) in ''.join(sentence_tokens[:i+1]) and verb == sentence_tokens[i]:
                    return i
            return None
        
        clean_tokens = [token.strip('Ġ') for token in clean_tokens]
        corr_tokens = [token.strip('Ġ') for token in corr_tokens]
        
        for verb in verbs:
            # remember to add ' ' to the beginning of the verb (verb in sentences, not alone!)
            clean_v_tokens = tokenizer.tokenize(' ' + verb["clean"]) if not "'" in verb["clean"] else [verb["clean"]]  # special case: 's 
            corr_v_tokens = tokenizer.tokenize(' ' + verb["flip"]) if not "'" in verb["flip"] else [verb["flip"]]
            
            # verb is cut in subwords
            if len(clean_v_tokens) > 1 or len(corr_v_tokens) > 1:
                continue
            clean_v_str_token = clean_v_tokens[0]
            corr_v_str_token = corr_v_tokens[0]
            
            clean_v_token = clean_v_str_token.strip('Ġ')
            corr_v_token = corr_v_str_token.strip('Ġ')
            
            # n't
            if verb["neg"]:
                clean_v_token += "n"
                corr_v_token += "n"
                clean_v_str_token += "n"
                corr_v_str_token += "n"
                if clean_v_token not in ["don", "doesn", "haven", "hasn", "isn", "aren"]:
                    continue
            
            # discard the first
            if verb["pre_word"] is None:
                continue
            
            # model str v.s. spacy
            else:
                # get pre_word and combine it with the verb to better locate the verb
                if type(verb["pre_word"]) == list:
                    clean_pre_word = verb["pre_word"][0]
                    corr_pre_word = verb["pre_word"][1]
                else:
                    clean_pre_word = verb["pre_word"]
                    corr_pre_word = verb["pre_word"]
                
                # get the pos of the verb
                ret1 = find_verb_position(clean_tokens, clean_v_token, clean_pre_word)
                ret2 = find_verb_position(corr_tokens, corr_v_token, corr_pre_word)
                
                if ret1 is None or ret2 is None:
                    continue
                clean_v_pos = ret1
                corr_v_pos = ret2
                
                # check if the verb is correctly located
                clean_v_check = clean_tokens[clean_v_pos] if not verb["neg"] else clean_tokens[clean_v_pos][:-1]
                corr_v_check = corr_tokens[corr_v_pos] if not verb["neg"] else corr_tokens[corr_v_pos][:-1]                                                                      
                assert clean_v_check == verb["clean"], f"clean_v {clean_v_check} != clean verb {verb['clean']} {clean_tokens}"
                assert corr_v_check == verb["flip"], f"corr_v {corr_v_check} != corr verb {verb['flip']} {corr_tokens}"
                # assert clean_v_pos == corr_v_pos, f"clean_v_pos {clean_v_pos} != corr_v_pos {corr_v_pos}"
                
                # get the id of the verb
                clean_v_id = tokenizer.convert_tokens_to_ids(clean_v_str_token)
                corr_v_id = tokenizer.convert_tokens_to_ids(corr_v_str_token)
                clean_verb_pos.append(clean_v_pos)
                corr_verb_pos.append(corr_v_pos)
                clean_verb_ids.append(clean_v_id)
                corr_verb_ids.append(corr_v_id)
                clean_verbs.append(clean_v_token)
                corr_verbs.append(corr_v_token)

        verb_info = {"clean_verb_pos": clean_verb_pos, 
                     "corr_verb_pos": corr_verb_pos,
                     "clean_verb_ids": clean_verb_ids, 
                     "corr_verb_ids": corr_verb_ids,
                     "clean_verbs": clean_verbs,
                     "corr_verbs": corr_verbs}
        
        return None if not clean_verb_pos else verb_info
    
def process_data_pile():
    data_path = "/raid_sdd/lyy/dataset/datasets--NeelNanda--pile-10k/snapshots/127bfedcd5047750df5ccf3a12979a47bfa0bafa/data/train-00000-of-00001-4746b8785c874cc7.parquet"
    df = pd.read_parquet(data_path)
    dataset = df.to_dict(orient='records')
    
    raw_pile_data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_pile_10k.jsonl"
    with jsonlines.open(raw_pile_data_path, "w") as f:
        for data in dataset:
            if data["meta"]["pile_set_name"] in ["Github", "ArXiv", "PubMed Abstracts", "PubMed Central", "StackExchange", "USPTO Backgrounds", "Pile-CC", "DM Mathematics", "FreeLaw"]:   
                continue
            lines = data["text"].split("\n")
            for line in lines:
                words = line.split(" ")
                if len(words) <= 5:  # '\n' or too short
                    continue
                sublines = line.split(". ")
                for i, subline in enumerate(sublines):
                    # bad case
                    if len(subline) <= 25:
                        continue
                    if any(word in subline for word in ["http", "www.", "png", "jpg", "gif"]):
                        continue
                    
                    # mend
                    if i < len(sublines) - 1:
                        subline += "."
                        
                    # language detection -> sentence completeness -> tense detection 
                    conditions = [is_english(subline), is_complete(subline), is_present_tense(subline)]
                    if not all(conditions):
                        continue
                    
                    # normal case
                    f.write({"raw_text": subline, "meta": data["meta"]})

def annotate_data_pile(src_path, tgt_path):
    data = []
    with jsonlines.open(src_path) as f:
        data = list(f)
    with jsonlines.open(tgt_path, "w") as f:
        for d in tqdm(data):
            d["raw_text"] = d["raw_text"].replace("’", "'").replace("‘，", ",")  # mend "You’re" ...
            verbs, corr_text = annotate_verbs(d["raw_text"])
            f.write({"raw_text": d["raw_text"], 
                     "corr_text": corr_text,
                     "verbs": verbs, 
                     "meta": d["meta"]})

def format_data_pile(src_path, tgt_path, tokenizer):
    """
    Find verb pos and ids using the model's tokenizer.
    """
    dataset = []
    with jsonlines.open(src_path, "r") as f:
        for line in f:
            dataset.append(line)
    
    with jsonlines.open(tgt_path, "w") as f:
        for line in tqdm(dataset):
            # {"raw_text": , "corr_text": , "verbs": , "meta": }
            clean_tokens = tokenizer.tokenize(line["raw_text"])
            corr_tokens = tokenizer.tokenize(line["corr_text"])
            if len(clean_tokens) > 100:
                continue
            else:
                # {"clean_text": , "corr_text": , "verb_pos", "clean_verb_ids": , "corr_verb_ids": }
                new_line = {"clean_text": line["raw_text"],
                            "corr_text": line["corr_text"]}
                verb_info = get_verb_info(clean_tokens, corr_tokens, line["verbs"], tokenizer)
                if verb_info is None:
                    continue
                new_line["clean_verb_pos"] = verb_info["clean_verb_pos"]
                new_line["corr_verb_pos"] = verb_info["corr_verb_pos"]
                new_line["clean_verb_ids"] = verb_info["clean_verb_ids"]
                new_line["corr_verb_ids"] = verb_info["corr_verb_ids"]
                new_line["clean_verbs"] = verb_info["clean_verbs"]
                new_line["corr_verbs"] = verb_info["corr_verbs"]
                
                f.write(new_line)


def split_dataset(src_path, tgt_dir, data_num, ratio):
    """
    Args:
    ratio: [train, dev, test] e.g. [8, 1, 1]
    """
    # split the dataset into train, dev, test
    with jsonlines.open(src_path, "r") as f:
        data = list(f)
    data = data[:data_num]
    train_num = int(data_num * ratio[0] / sum(ratio))
    dev_num = int(data_num * ratio[1] / sum(ratio))
    test_num = data_num - train_num - dev_num
    
    train_data = data[:train_num]
    dev_data = data[train_num:train_num+dev_num]
    test_data = data[train_num+dev_num:]
    
    train_path = os.path.join(tgt_dir, f"train_{train_num}.jsonl")
    dev_path = os.path.join(tgt_dir, f"dev_{dev_num}.jsonl")
    test_path = os.path.join(tgt_dir, f"test_{test_num}.jsonl")
    
    with jsonlines.open(train_path, "w") as f:
        for d in train_data:
            f.write(d)
    with jsonlines.open(dev_path, "w") as f:
        for d in dev_data:
            f.write(d)
    with jsonlines.open(test_path, "w") as f:
        for d in test_data:
            f.write(d)


def get_subj():
    
    data_path = "/raid_sdi/home/lyy/lyy/Interpretability/circuit-tuning/data/sv_dataset/test_3000_single_verb.jsonl"
    tgt_path = "/raid_sdi/home/lyy/lyy/Interpretability/circuit-tuning/data/sv_dataset/test_3000_single_verb_new.jsonl"
    with jsonlines.open(data_path, "r") as f, jsonlines.open(tgt_path, "w") as f1:
        for line in f:
            sentence = line["clean_text"]
            doc = nlp(sentence)
            verb = line["clean_verbs"][0]
            subj = None
            subj_next_word = None
            for token in doc:
                if token.text == verb:
                    children = list(token.children)
                    for child in children:
                        if child.dep_ == "nsubj":
                            subj = child.text
                            subj_next_word = doc[child.i + 1].text
                            print(f"Verb: {token.text}, Subject: {child.text}")
                            break
                    if subj is None:
                        token_to_check = doc[token.i + 1]
                        print(token_to_check)
                        children = list(token_to_check.children)
                        for child in children:
                            if child.dep_ == "nsubj":
                                subj = child.text
                                subj_next_word = doc[child.i + 1].text
                                print(f"Verb: {token.text}, Subject: {child.text}")
                    break
            if subj is not None:
                data = line.copy()
                data["subj"] = subj
                data["subj_next_word"] = subj_next_word
                f1.write(data)

def format_subj(tokenizer):
    
    data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_single_verb_with_subj.jsonl"
    tgt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/test_3000_single_verb_with_subj_formatted.jsonl"
    with jsonlines.open(data_path, "r") as f, jsonlines.open(tgt_path, "w") as f1:
        for line in f:
            sentence = line["clean_text"]
            subj_token = line["subj"]
            sentence_tokens = tokenizer.tokenize(sentence)
            sentence_tokens = [token.strip('Ġ') for token in sentence_tokens]
            
            # if not ' ' + subj_token in sentence:  # Jed, Bailey...
            #     subj_tokens = tokenizer.tokenize(subj_token + ' ')
            #     subj_tokens = [token for token in subj_tokens if token not in ['Ġ', '']]
            #     subj_tokens = [token.strip('Ġ') for token in subj_tokens]
            # else:
            #     subj_tokens = tokenizer.tokenize(' ' + subj_token)
            #     subj_tokens = [token.strip('Ġ') for token in subj_tokens]
            
            
            # def find_word_position(sentence_tokens, next_word, word_tokens):
            #     word_len = len(word_tokens)
            #     word = ''.join(word_tokens)
            #     for i in range(len(sentence_tokens)):
            #         if word == ''.join(sentence_tokens[i:i+word_len]) and ''.join(sentence_tokens[i+word_len:]).startswith(next_word):
            #             return (i, i+word_len)
            #     return None
            
            def find_word_position(sentence_tokens, next_word, word):
                word_group = word + next_word
                for i in range(len(sentence_tokens)):
                    if ''.join(sentence_tokens[i:]).startswith(word_group):
                        j=1
                        word_to_match = ''.join(sentence_tokens[i:i+j])
                        while not word_to_match == word:
                            print(word_to_match, word)
                            j+=1
                            word_to_match = ''.join(sentence_tokens[i:i+j])
                            if j > 10:
                                return None
                        return (i, i+j)
            
            subj_pos = find_word_position(sentence_tokens, line["subj_next_word"], subj_token)
            
            if not subj_pos:
                print("subj_tokens", subj_token, "sentence_tokens", sentence_tokens, "next_word", line["subj_next_word"], "subj_pos", subj_pos)
                # import pdb; pdb.set_trace()
                continue
            else:
                subj_start, subj_end = subj_pos
                subj_pos = [i for i in range(subj_start, subj_end)]
            
            # check if the words are correctly located
            subj_check = ''.join(sentence_tokens[subj_start:subj_end])
            assert subj_check == subj_token, f"subj_check {subj_check} != subj_token {subj_token}"
                    
            item = line.copy()
            del item["subj_next_word"]
            item["subj_pos"] = subj_pos
            
            f1.write(item)


if __name__ == '__main__':
    
    # model = HookedTransformer.from_pretrained(
    #     'gpt2-small',
    #     center_writing_weights=False,
    #     center_unembed=False,
    #     fold_ln=False,
    # )
    
    # # annotate and format data
    # src_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_pile_10k.jsonl"
    # tgt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_pile_10k_annotated.jsonl"
    # annotate_data_pile(src_path=src_path, tgt_path=tgt_path)
    
    # src_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_pile_10k_annotated.jsonl"
    # tgt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_pile_10k_formatted.jsonl"
    # format_data_pile(src_path=src_path, tgt_path=tgt_path, tokenizer=model.tokenizer)
    
    
    # sentence = "Bath's classical trappings aren't to everyone's taste, so things are kept strictly modern at this city-centre crash pad, with searingly bright colour schemes and clashing wallpapers giving it a fun, young vibe."
    # verbs, corr_text = annotate_verbs(sentence)
    # print(verbs, corr_text)
    
    # clean_tokens = model.tokenizer.tokenize(sentence)
    # corr_tokens = model.tokenizer.tokenize(corr_text)
    # verb_info = get_verb_info(clean_tokens, corr_tokens, verbs, model.tokenizer)
    
    # print(verb_info)
    
    # src_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_pile_10k_formatted.jsonl"
    # tgt_dir = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset"
    # data_num = 30000 
    # ratio = [8, 1, 1]
    # split_dataset(src_path, tgt_dir, data_num, ratio)
    
    from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("/home/lyy/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
    # format_subj(tokenizer)

    data_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted.jsonl"
    tgt_path = "/raid_sdd/lyy/Interpretability/lyy/circuit-tuning/data/sv_dataset/sv/test_3000_single_verb_with_subj_formatted_type0.jsonl"
    with jsonlines.open(data_path, "r") as f, jsonlines.open(tgt_path, "w") as f1:
        for line in f:
            subj_pos = line["subj_pos"][0]
            verb_pos = line["clean_verb_pos"][0]
            if verb_pos - subj_pos == 1:
                item = line.copy()
                f1.write(item)