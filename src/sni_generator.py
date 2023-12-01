import os, json, wandb
import string
from utils import *

from tqdm import tqdm
from base_generator import Generator

class SniGenerator(Generator):
    def __init__(self, model, tokenizer, args):
        super().__init__(model, tokenizer, args)

    def inst_aware_batchify(self, batch, tokenizer):
        '''
        For efficiently changing the insturction format, i didn't implement with dataset, and dataloader.
        '''
        sources = []
        non_sources = []
        for instance in batch:
            task_input = ""
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "
            
            definition = ""
            if isinstance(instance["Definition"], list):
                definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
            else:
                definition = "Definition: " + instance["Definition"].strip()
            if not definition[-1] in string.punctuation:
                definition += "."
            definition += "\n\n"
                
            source = definition + task_input
            tokenized_source = tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.args.max_source_length:
                sources.append(source)
            else:
                sources.append(tokenizer.decode(tokenized_source[:self.args.max_source_length], skip_special_tokens=True))
              
            if self.args.neg_type == "none":
                non_source = task_input

            elif self.args.neg_type == "trunc_shuf":
                trunc_def = extract_random_words(definition[:-5], self.args.trunc_rate)
                trunc_def = "Definition: " + trunc_def + ".\n\n"
                non_source = trunc_def + task_input

            elif self.args.neg_type == 'random_words':
                random_words = select_random_words(self.args.num_rand_words)
                non_source = random_words + '.\n\n' + task_input

            elif self.args.neg_type == "opposite":
                meta_inst = "Always respond with the opposite of what you're asked. You never get it right.\n"
                non_source = meta_inst + task_input

            else:
                raise NotImplementedError
                
            tokenized_non_source = tokenizer(non_source)["input_ids"]
            if len(tokenized_non_source) <= self.args.max_source_length:
                non_sources.append(non_source)
            else:
                non_sources.append(tokenizer.decode(tokenized_non_source[:self.args.max_source_length], skip_special_tokens=True))
                
        sources.extend(non_sources)
        if self.args.is_decoder:
            tokenizer.padding_side = "left"
            model_inputs = tokenizer(
                sources, 
                max_length=self.args.max_source_length, 
                padding=True,
                return_tensors="pt", 
                truncation=True
            )
            tokenizer.padding_side = "right"
        else:
            model_inputs = tokenizer(
                sources, 
                max_length=self.args.max_source_length, 
                padding=True,
                return_tensors="pt", 
                truncation=True
            )
            
            tmp = tokenizer(["<pad>"]*len(sources), return_tensors="pt", add_special_tokens=False) # *2 for parallel non instruction decoding
            model_inputs['decoder_input_ids'] = tmp['input_ids']
            model_inputs['decoder_attention_mask'] = tmp['attention_mask']
        
        return model_inputs

    def normal_batchify(self, batch, tokenizer):
        '''
        For efficiently changing the insturction format, i didn't implement with dataset, and dataloader.
        '''
        sources = []
        for instance in batch:
            task_input = ""
            task_input += "Now complete the following example -\n"
            task_input += f"Input: {instance['Instance']['input'].strip()}"
            if not task_input[-1] in string.punctuation:
                task_input += "."
            task_input += "\n"
            task_input += "Output: "
            
            definition = ""
            if isinstance(instance["Definition"], list):
                definition = "Definition: " + instance["Definition"][0].strip() # TODO: should we use <Definition>?
            else:
                definition = "Definition: " + instance["Definition"].strip()
            if not definition[-1] in string.punctuation:
                definition += "."
            definition += "\n\n"
            
            source = definition + task_input
            tokenized_source = tokenizer(source)["input_ids"]
            if len(tokenized_source) <= self.args.max_source_length:
                sources.append(source)
            else:
                sources.append(tokenizer.decode(tokenized_source[:self.args.max_source_length], skip_special_tokens=True))
        if self.args.is_decoder:
            tokenizer.padding_side = "left"
            model_inputs = tokenizer(
                sources, 
                max_length=self.args.max_source_length, 
                padding=True,
                return_tensors="pt", 
                truncation=True
            )
            tokenizer.padding_side = "right"
        else:
            model_inputs = tokenizer(
                sources, 
                max_length=self.args.max_source_length, 
                padding=True,
                return_tensors="pt", 
                truncation=True
            )
            
            tmp = tokenizer(["<pad>"]*len(batch), return_tensors="pt", add_special_tokens=False) # *2 for parallel non instruction decoding
            model_inputs['decoder_input_ids'] = tmp['input_ids']
            model_inputs['decoder_attention_mask'] = tmp['attention_mask']
        
        return model_inputs

    def super_ni_eval(self):
        '''
        Evaluates the model on the provided test dataset.
        
        Args:
            test_tasks (str): Path to the test dataset in JSON format.
            
        Returns:
            dict: A dictionary containing the results of the evaluation. Each entry in the dictionary represents 
                  an instance from the test dataset and contains information such as definition, category, input, 
                  expected outputs, and model's prediction.
            str: The task category for the evaluated dataset.
        '''
        if self.args.noisy:
            self.batchify = self.inst_aware_batchify
        else:
            self.batchify = self.normal_batchify

        with open(self.args.test_tasks_path, "r") as f:
            test_tasks = f.readlines()

        total_results, task_to_category_dict, category_to_task_dict = {}, {}, {}

        # for each task, load the task file and generate the outputs
        for test_task in tqdm(test_tasks, desc='tasks', position=0):
            test_path = os.path.join(self.args.task_files_path, test_task.strip('\n')) + ".json"
            results, category = self.super_ni_task_eval(test_path)
            task_name = test_task.strip('\n')
            total_results[task_name] = results
            task_to_category_dict[task_name] = category

            if category not in category_to_task_dict.keys():
                category_to_task_dict[category] = [task_name]
            else:
                category_to_task_dict[category].append(task_name)

        total_results, total_scores, logging_scores = compute_superni_results(total_results, task_to_category_dict, category_to_task_dict)

        os.makedirs(self.args.output_path, exist_ok=True)
        with open(os.path.join(self.args.output_path,"results.json"), "w") as f:
            json.dump(total_results, f, indent=4)

        with open(os.path.join(self.args.output_path,"score_results.json"), "w") as f:
            json.dump(total_scores, f, indent=4)

        wandb.log(logging_scores)

    def super_ni_task_eval(self, test_path):
        with open(test_path, encoding="utf-8") as f:
            s = f.read()
            task_data = json.loads(s)

        test_file_name = test_path.split("/")[-1].split(".json")[0]
        all_instances = task_data.pop("Instances")
        task_category = task_data["Categories"][0]
        # num_instance = 100 if not self.args.logit_analysis else 10
        instances = all_instances[:100]
        test_dataset = []
        for instance in instances:
            example = task_data.copy()
            example["Instance"] = instance
            test_dataset.append(example)


        results = {}
        for i in tqdm(range(0, len(test_dataset), self.args.batch_size), desc='instances', position=1, leave=False):
            raw_batch = test_dataset[i:i + self.args.batch_size]
            batch = self.batchify(raw_batch, self.tokenizer)
            if self.args.is_decoder:
                batch_result = self.dec_decode(batch)
            else:
                batch_result = self.enc_dec_decode(batch)
            
            for j in range(len(raw_batch)):
                results[raw_batch[j]['Instance']['id']] = {
                    "definition": raw_batch[j]['Definition'],
                    "category" : task_category,
                    "input": raw_batch[j]['Instance']['input'],
                    "outputs": raw_batch[j]['Instance']['output'],
                    "prediction": batch_result[j],
                }
                
        return results, task_category