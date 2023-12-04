from transformers import GenerationConfig

import torch
import torch.nn.functional as F

class InlineGenerator():
    def __init__(self, model, tokenizer, args):
        self.device = args.device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.generation_config = GenerationConfig()

        self.model.eval()
        self.encoder = self.model.get_encoder()
        self.encoder.eval()
        
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length
        
    def instructive_decoding(self, base_input, noisy_input, eps=0.3, verbose=True):
        
        baseline_response = self.generate(base_input, noisy_input, eps=0)
        id_response = self.generate(base_input, noisy_input, eps=eps)

        if verbose:
            print("\n============== Base Instruction Example ================")
            print(f"{base_input}\n" + "="*50)

            print("\n============== Noisy Instruction Example ================")
            print(f"{noisy_input}\n" + "="*50)
            
            # Baseline Response
            print(f"[Base Response]: {baseline_response}")

            # Instructive Decoding Response 
            print(f"[ID Response]: \033[32m{id_response}\033[0m")
        
        return id_response

    @torch.no_grad()
    def generate(self, source_text, contrast_text=None, eps=None):
        '''
        Generates prediction by instructive decoding.
        '''
        eps = 0.0 if eps is None else eps
        
        # Preprocess inputs
        batch_size = 1 if contrast_text is None else 2
        batch = self._preprocess_text_inputs(source_text, contrast_text, batch_size)
        
        # Set up inputs
        encoder_input_ids = batch['input_ids'].to(self.device)
        encoder_attention_masks = batch['attention_mask'].to(self.device)
        decoder_input_ids = batch['decoder_input_ids'].to(self.device)
        decoder_attention_masks = batch['decoder_attention_mask'].to(self.device)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long).to(self.device)
        past_key_values = None
        
        encoder_outputs = self.encoder(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_masks,
                return_dict=True
        )
        
        model_kwargs = {
            "decoder_attention_mask":decoder_attention_masks,
            "past_key_values":past_key_values, 
            "encoder_outputs":encoder_outputs,
            "attention_mask":encoder_attention_masks
        }
        
        # Generate
        for _ in range(self.max_target_length):
        
            # Forward pass
            model_inputs = self.model.prepare_inputs_for_generation(decoder_input_ids, **model_kwargs)
            outputs = self.model(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]
            
            source_logits = next_token_logits[0]
            
            # Refine token logits
            if eps == 0:
                contrast_logits = None
                final_token_logits = source_logits
                
            else:
                contrast_logits = next_token_logits[1]
                final_token_logits = source_logits - eps * contrast_logits
                            
            # Finalize next token
            final_tokens = self._generate_next_token(final_token_logits, decoder_input_ids)
            final_tokens = final_tokens * unfinished_sequences + self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            
            # Prepare next forward
            decoder_input_ids = torch.cat([decoder_input_ids, final_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder=True)
            
            unfinished_sequences = unfinished_sequences.mul((final_tokens != self.tokenizer.eos_token_id).long())
            if unfinished_sequences.max() == 0:
                break
            
        final_outputs = self.tokenizer.batch_decode(decoder_input_ids, skip_special_tokens=True)[0]
        
        return final_outputs
    
    def _generate_next_token(self, next_token_logits, decoder_input_ids):
        next_tokens = torch.argmax(next_token_logits, dim=-1)
            
        return next_tokens

    def _preprocess_text_inputs(self, source_text, contrast_text, batch_size=1):
        input_batch = []
        
        # Tokenize source text
        source_text = self._truncate_text(source_text)
        input_batch.append(source_text)
        
        # Tokenize contrast text
        if contrast_text is not None:
            contrast_text = self._truncate_text(contrast_text)
            input_batch.append(contrast_text)

        model_inputs = self.tokenizer(
            input_batch, 
            max_length=self.max_source_length, 
            padding=True,
            return_tensors="pt", 
            truncation=True
        )
        
        tmp = self.tokenizer(["<pad>"]*batch_size, return_tensors="pt", add_special_tokens=False)
        model_inputs['decoder_input_ids'] = tmp['input_ids']
        model_inputs['decoder_attention_mask'] = tmp['attention_mask']
    
        return model_inputs

    def _truncate_text(self, input_text):        
        truncated_text = input_text
        tokenized_text = self.tokenizer(input_text)["input_ids"]
        
        if len(tokenized_text) > self.max_source_length:
            truncated_text = self.tokenizer.decode(tokenized_text[:self.max_source_length], skip_special_tokens=True)
        
        return truncated_text
