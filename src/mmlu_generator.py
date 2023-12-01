import os, json, wandb, string
import pandas as pd

from utils import *
from tqdm import tqdm
from base_generator import Generator

class MMLUGenerator(Generator):
    def __init__(self, model, tokenizer, args):
        super().__init__(model, tokenizer, args)
        self.choices = ["A", "B", "C", "D"]
        prefix = " " if self.args.is_decoder else ""
        self.answer_choice_ids = [self.tokenizer.encode(prefix + answer_choice, add_special_tokens=False)[-1] for answer_choice in self.choices]

    def inst_aware_batchify(self, batch, tokenizer):
        '''
        For efficiently changing the insturction format, i didn't implement with dataset, and dataloader.
        '''
        sources = []
        noisy_sources = []
        for prompt, options in batch:
            # Each instruction in MMLU consists of a subject-related question can be along with four answer choices (i.e., “Answer with A, B, C or D”).
            prefix = "Answer with A, B, C, or D.\n" if self.args.answer_with else ""
            source = prefix + prompt + options + "\nAnswer:"
            sources.append(source)

            if self.args.neg_type == "none":
                non_source = "Answer:"
            elif self.args.neg_type == "none_options":
                non_source = options + "\nAnswer:"

            elif self.args.neg_type == "opposite":
                non_source = "Always respond with the opposite of what you're asked. You never get it right.\nAnswer:"
            elif self.args.neg_type == "opposite_options":
                non_source = "Always respond with the opposite of what you're asked. You never get it right." + options + "\nAnswer:"
            elif self.args.neg_type == "opposite_full":
                non_source = "Always respond with the opposite of what you're asked. You never get it right.\n" + prompt + options + "\nAnswer:"
            noisy_sources.append(non_source)

        sources.extend(noisy_sources)

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
            
            tmp = tokenizer(["<pad>"]*len(sources), return_tensors="pt", add_special_tokens=False)
            model_inputs['decoder_input_ids'] = tmp['input_ids']
            model_inputs['decoder_attention_mask'] = tmp['attention_mask']
        
        return model_inputs
        

    def normal_batchify(self, batch, tokenizer):
        '''
        For efficiently changing the insturction format, i didn't implement with dataset, and dataloader.
        '''
        sources = []
        for prompt, options in batch:
            prefix = "Answer with A, B, C, or D.\n" if self.args.answer_with else ""
            source = prefix + prompt + options + "\nAnswer:"
            sources.append(source)

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
            
            tmp = tokenizer(["<pad>"]*len(sources), return_tensors="pt", add_special_tokens=False)
            model_inputs['decoder_input_ids'] = tmp['input_ids']
            model_inputs['decoder_attention_mask'] = tmp['attention_mask']

        return model_inputs

    def mmlu_eval(self):
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

        subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(self.args.task_files_path, "test"))
            if "_test.csv" in f
        ])
        results = {}
        for subject in tqdm(subjects, desc='tasks', position=0):
            test_df = pd.read_csv(
                    os.path.join(self.args.task_files_path, "test", subject + "_test.csv"), header=None
                )
            
            subject_result = self.mmlu_task_eval(subject, test_df)
            ground_truth = test_df.iloc[:, -1].values.tolist()
            results[subject] = {
                'input': subject_result['input'],
                'output': ground_truth,
                'prediction': subject_result['prediction']
            }
            
        logging_scores = compute_mmlu_results(results)

        os.makedirs(self.args.output_path, exist_ok=True)
        with open(os.path.join(self.args.output_path,"results.json"), "w") as f:
            json.dump(results, f, indent=4)

        with open(os.path.join(self.args.output_path,"score_results.json"), "w") as f:
            json.dump(logging_scores, f, indent=4)

        wandb.log(logging_scores)
            
    def extract_example(self, df, idx):
        prompt = df.iloc[idx, 0]

        options = ""
        k = df.shape[1] - 2
        for j in range(k):
            options += "\n{}. {}".format(self.choices[j], df.iloc[idx, j + 1])

        return prompt, options

    def mmlu_task_eval(self, subject, test_df):
        prompts = []
        for i in range(0, test_df.shape[0]):
            prompts.append(self.extract_example(test_df, i))

        results = {
            "input": [],
            "prediction": []
        }
        for i in tqdm(range(0, len(prompts), self.args.batch_size), desc='instances', position=1, leave=False):
            raw_batch = prompts[i:i + self.args.batch_size]
            batch = self.batchify(raw_batch, self.tokenizer)
            if self.args.is_decoder:
                if self.args.candidate:
                    batch_result = self.dec_decode(batch, self.answer_choice_ids)
                else:
                    batch_result = self.dec_decode(batch)
            else:
                if self.args.candidate:
                    batch_result = self.enc_dec_decode(batch, self.answer_choice_ids)
                else:
                    batch_result = self.enc_dec_decode(batch)
            
            for j in range(len(raw_batch)):
                results['input'].append(raw_batch[j])
                results['prediction'].append(batch_result[j])

        return results