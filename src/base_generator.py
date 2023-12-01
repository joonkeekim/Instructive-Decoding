from typing import Dict, List, Tuple, Any
from transformers import GenerationConfig

import torch
import torch.nn.functional as F

class Generator():
    '''
    Generator class: designed to handle the generation of text using a given model and tokenizer.
    
    Attributes:
        model: The pre-trained model used for text generation.
        tokenizer: The tokenizer used for converting text to and from token format.
        args: Configuration arguments for the generator.
        generation_config: Configuration for text generation (e.g., top-k, top-p, temperature).
        do_sample: Whether or not to use sampling in text generation.
        logits_warper: Utility for modifying model outputs.
        t5_encoder: Encoder component of a T5 model (if applicable).
    '''
    def __init__(self, model, tokenizer, args):
        '''
        Initializes the Generator with the given model, tokenizer, and configuration arguments.
        '''
        self.model = model.to(device=args.device)
        if args.half:
            self.model = model.half()

        self.tokenizer = tokenizer
        self.device = args.device
        self.args = args
        
        self.generation_config = GenerationConfig()
        # top-k sampling
        if args.top_k > 0:
            self.generation_config.top_k = int(args.top_k)
        # top-p sampling
        if args.top_p > 0:
            self.generation_config.top_p = args.top_p
        # temperature
        if args.temp > 0:
            self.generation_config.temperature = args.temp

        # do sampling
        if args.top_k > 0 or args.top_p > 0 or args.temp > 0:
            self.do_sample = True
        else:
            self.do_sample = False

        self.generation_config.do_sample = self.do_sample
        self.logits_warper = model._get_logits_warper(self.generation_config)
        
        self.model.eval()
        if not self.args.is_decoder:
            self.t5_encoder = self.model.get_encoder()
            self.t5_encoder.eval()
        
        
        
    def enc_dec_decode(self, batch: Dict[str, torch.Tensor], candidate_tokens=None) -> List[str]:
        '''
        Generates predictions for a given batch using the encoder decoder model.
        
        Args:
            batch (dict): A dictionary containing the required inputs for the model, such as input_ids, 
                          attention_mask, decoder_input_ids, and decoder_attention_mask.

        Returns:
            list: A list of generated sequences for each input in the batch.

        '''
        encoder_input_ids = batch['input_ids'].to(self.device)
        encoder_attention_masks = batch['attention_mask'].to(self.device)
        decoder_input_ids = batch['decoder_input_ids'].to(self.device)
        decoder_attention_masks = batch['decoder_attention_mask'].to(self.device)
        
        batch_size = encoder_input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long).to(self.device)
        past_key_values = None
        
        with torch.no_grad():
            encoder_outputs = self.t5_encoder(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_masks,
                    return_dict=True
            )
            
            model_kwargs = {"decoder_attention_mask":decoder_attention_masks,
                        "past_key_values":past_key_values, 
                        "encoder_outputs":encoder_outputs,
                        "attention_mask":encoder_attention_masks}

            for i in range(self.args.max_target_length):
                model_inputs = self.model.prepare_inputs_for_generation(decoder_input_ids, **model_kwargs)
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                )
                # divide
                next_token_logits = outputs.logits[:, -1, :]
                if self.args.noisy:
                    next_token_logits = self.inst_decode(next_token_logits, decoder_input_ids)
                # normal
                if self.do_sample:
                    next_token_scores = self.logits_warper(decoder_input_ids, next_token_logits)
                    probs = F.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                elif candidate_tokens is not None:
                    next_token_logits = next_token_logits[:, candidate_tokens]
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)      

                # update
                next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
                # prepare next forward
                decoder_input_ids = torch.cat([decoder_input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=True)
                # update sequence status
                unfinished_sequences = unfinished_sequences.mul((next_tokens != self.tokenizer.eos_token_id).long())
                if unfinished_sequences.max() == 0:
                    break
                
            if candidate_tokens is not None:
                result = []
                for ids in decoder_input_ids:
                    result.append(self.choices[ids[-1]])
            else:
                result = self.tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)
        return result
    
    def dec_decode(self, batch: Dict[str, torch.Tensor], candidate_tokens=None) -> List[str]:
        '''
        Generates predictions for a given batch using the decoder only model.
        
        Args:
            batch (dict): A dictionary containing the required inputs for the model, such as input_ids, 
                          attention_mask, decoder_input_ids, and decoder_attention_mask.

        Returns:
            list: A list of generated sequences for each input in the batch.

        '''
        input_ids = batch['input_ids'].to(self.device)
        attention_masks = batch['attention_mask'].to(self.device)
        model_kwargs = {'attention_mask':attention_masks}
        batch_size, input_seq_len = input_ids.shape
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            for i in range(self.args.max_target_length):
                model_inputs = self.model.prepare_inputs_for_generation(input_ids=input_ids,**model_kwargs)
                outputs = self.model(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                # divide
                next_token_logits = outputs.logits[:, -1, :]
                
                if self.args.noisy:
                    next_token_logits = self.inst_decode(next_token_logits, input_ids)
                if self.do_sample:
                    next_token_scores = self.logits_warper(input_ids, next_token_logits)
                    probs = F.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                elif candidate_tokens is not None:
                    next_token_logits = next_token_logits[:, candidate_tokens]
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)      

                # update
                next_tokens = next_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
                # prepare next forward
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                model_kwargs = self.model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=False)
                # update sequence status
                unfinished_sequences = unfinished_sequences.mul((next_tokens != self.tokenizer.eos_token_id).long())
                if unfinished_sequences.max() == 0:
                    break
                
            result = input_ids[:,input_seq_len:]
            if candidate_tokens is not None:
                results = []
                for res in result:
                    results.append(self.choices[res])
                result = results
            else:
                result = self.tokenizer.batch_decode(result, skip_special_tokens=True)
        return result

    def inst_decode(self, 
                next_token_logits: torch.Tensor, 
                decoder_input_ids: torch.Tensor) -> torch.Tensor:
        '''
        Decodes tokens based on instance information and adjusts the logits based on different strategies.

        Args:
            next_token_logits (torch.Tensor): Raw logits for the next token prediction.
            decoder_input_ids (torch.Tensor): Input IDs used for decoding so far.
            time_step (int): Current time step in the decoding process.

        Returns:
            torch.Tensor: Decoded token IDs for the next position.

        Raises:
            NotImplementedError: If an unsupported strategy or combination is used.
        '''
        batch_size = next_token_logits.shape[0]//2
        inst_logits, non_inst_logits = next_token_logits[:batch_size, :], next_token_logits[batch_size:, :]

        next_token_logits = inst_logits + self.args.eps * non_inst_logits
        next_token_logits = torch.cat([next_token_logits, next_token_logits], dim=0)
        return next_token_logits