import transformers
import torch
from tqdm import tqdm
import json
from clip import clip
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# For coarse-grained description generation
templates = [
    "What visual cue is unique to {} among all {}?",
    "What are the differences between {} and other {} in appearance?",
    "What physical attributes define the appearance of {} among all {}?",
    "How is {} distinguished from other {} visually?",
    "What textures or surface characteristics are typical of {} among all {}?",
]

# For fine-grained description generation
templates_ = [
    "What visual cue is unique to {} different from {}?",
    "What are the differences between {} and {} in appearance?",
    "What physical attributes define the appearance of {} different from {}?",
    "How is {} distinguished from {} visually?",
    "What textures or surface characteristics are typical of {} different from {}?",
]

infos = {
    'EuroSAT':              ["{}",                "types of land in a centered satellite photo", 2], # 10
    'OxfordPets':           ["a pet {}",          "types of pets", 2], # 37
    'DescribableTextures':  ["a {} texture",      "types of texture", 2], # 47
    'FGVCAircraft':         ["a {} aircraft",     "types of aircraft", 3], # 100
    'Caltech101':           ["{}",                "objects", 3], # 101
    'Food101':              ["{}",                "types of food", 3], # 101
    'UCF101':               ["a person doing {}", "types of action", 3], # 101
    'OxfordFlowers':        ["a flower {}",       "types of flowers", 3], # 102
    'StanfordCars':         ["a {} car",          "types of car", 3], # 196
    'SUN397':               ["a {} scene",        "types of scenes", 3], # 397
    'ImageNet':             ["{}",                "objects", 4], # 1000
}


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device=1,
    batch_size=10
)

pipeline.tokenizer.pad_token_id = pipeline.tokenizer.eos_token_id
pipeline.tokenizer.padding_side = 'left'
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

backbone_name = "ViT-B/16"
url = clip._MODELS[backbone_name]
model_path = clip._download(url)

try:
    # loading JIT archive
    model = torch.jit.load(model_path, map_location="cpu").eval()
    state_dict = None

except RuntimeError:
    state_dict = torch.load(model_path, map_location="cpu")

clip_model = clip.build_model(state_dict or model.state_dict()).cuda()

for dataset_name in infos.keys():
    print(dataset_name)
    info = infos[dataset_name]
    with open('../data/gpt_data/classname/'+dataset_name+'.txt', 'r') as f:
        classnames = f.read().split("\n")[:-1]
        result = {}
        feat_buffer = []
        #################################
        #       1. Coarse-grained       #
        #################################
        for classname in tqdm(classnames):
            prompts = [template.format(info[0], info[1]).format(classname) for template in templates] #  Do not response any prefix like Sure!.
            messages = []
            suffix = "The answer should be a descriptive sentence of no more than 20 words."
            for p in prompts:
                dialog = [{"role": "system", "content": "You are a chatbot, which is expert in describing and classifying objects via their features!"}, {"role": "user", "content": " ".join([p, suffix])}]
                messages.append(dialog)

            outputs = pipeline(
                messages,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=pipeline.tokenizer.eos_token_id
            )
            output_desp = [output[0]['generated_text'][-1]['content'] for output in outputs]
            result[classname] = output_desp
            with torch.no_grad():
                prompts = clip.tokenize(output_desp, truncate=True).cuda()
                text_features = clip_model.encode_text(prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                text_feature = text_features.mean(dim=0)
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
                feat_buffer.append(text_feature)
        feat_buffer = torch.stack(feat_buffer, dim=0)
        
        k = info[2] + 1
        logits = feat_buffer @ feat_buffer.t()
        all_idx = torch.argsort(logits, descending=True, dim=-1)[:, 1:k]

        #################################
        #        2. Fine-grained        #
        #################################
        result_new, result_final, result_json = {}, {}, {}
        for idx, classname in enumerate(tqdm(classnames)):
            cls_idx = all_idx[idx]
            new_names = [classnames[index] for index in cls_idx]
            prompts = [template.format(info[0], ", ".join(new_names)).format(classname) for template in templates_] #  Do not response any prefix like Sure!.
            
            messages = []
            suffix = "The answer should be a descriptive sentence only about {} of no more than 20 words, avoiding the appearance of other categories in the sentence.".format(classname)
            for p in prompts:
                dialog = [{"role": "system", "content": "You are a chatbot, which is expert in describing and classifying objects via their features!"}, {"role": "user", "content": " ".join([p, suffix])}]
                messages.append(dialog)

            outputs = pipeline(
                messages,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=pipeline.tokenizer.eos_token_id
            )
            output_desp = [output[0]['generated_text'][-1]['content'] for output in outputs]
            result_new[classname] = output_desp

            #################################
            #          3. Overall           #
            #################################
            prompt_1, prompt_2 = result[classname], output_desp
            template_final = "Please summarize the following two descriptions as an overall description of {} encompassing all relevant features within these descriptions: 1. {}; 2. {}"
            prompts = [template_final.format(classname, prompt_1[i], prompt_2[i]) for i in range(len(prompt_1))] #  Do not response any prefix like Sure!.
            
            messages = []
            suffix = "The answer should be a descriptive sentence of no more than 30 words without outputing other content."
            for p in prompts:
                dialog = [{"role": "system", "content": "You are a chatbot, which is expert in describing and classifying objects via their features!"}, {"role": "user", "content": " ".join([p, suffix])}]
                messages.append(dialog)

            outputs = pipeline(
                messages,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=pipeline.tokenizer.eos_token_id
            )
            output_desp = [output[0]['generated_text'][-1]['content'] for output in outputs]
            result_final[classname] = output_desp
            
            #################################
            #         4. Structured         #
            #################################
            instruction = "Please reconstruct the following sentence and find relationships within the sentence, and only return a JSON object. \
            The target JSON object contains a *list of dictionaries* with the following keys: 'A', 'R', 'B', \
            For a sentence of ''' In a centered satellite photo, annual crop land appears as a mosaic of varying shades of green and gold.''', \
            here is an ouput json examples: [{{\"A\": \"satellite\",\"R\": \"contains\",\"B\": \"crop land\"}}, {{\"A\": \"crop land\",\"R\": \"appears\",\"B\": \"mosaic\"}}]. \
            The value of 'A', 'R', and 'B' should be linked into one sentence.\
            Do not output anything other than the JSON object. \
            Sentence: '''{}'''"
            
            messages = []
            for p in output_desp:
                dialog = [{"role": "system", "content": "You are a chatbot, which is expert in describing and classifying objects via their features!"}, {"role": "user", "content": instruction.format(p)}]
                messages.append(dialog)
            
            outputs = pipeline(
                messages,
                max_new_tokens=512,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=pipeline.tokenizer.eos_token_id
            )
            output_stru = [output[0]['generated_text'][-1]['content'] for output in outputs]
            responses = []
            for after_colon in output_stru:
                try:
                    after_colon = after_colon.replace("\n", "")  # .replace("'", "\"")
                    json_dict = json.loads(after_colon)
                    responses.append(json_dict)
                except json.JSONDecodeError as e:
                    logging.error(f"JSONDecodeError: {str(e)}")
                    responses.append([])
                    
            result_json[classname] = [[item for item in stru if 'A' in item and 'B' in item and 'R' in item and isinstance(item['A'], str) and isinstance(item['R'], str) and isinstance(item['B'], str)] for stru in responses]
            
        #################################
        #           5. Merged           #
        #################################
        result_all = {}
        for classname in tqdm(classnames):
            result_all[classname] = {"coarse_description":result[classname], "fine_description":result_new[classname], "overall_description":result_final[classname], "structure":result_json[classname]}
        with open('../data/corpus/'+dataset_name+'.json','w') as f:
            json.dump(result_all, f, indent=4)