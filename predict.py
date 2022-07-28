import json
from config import Args
import torch
import time
import torch.nn.functional as F
from processing.tools import clean, top_k_top_p_filtering


def predict_summary1(text, args, tokenizer, model, device):

    CLS, SEP, PAD, UNK = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[PAD]', '[UNK]'])
    EOS, SOS = tokenizer.convert_tokens_to_ids(['[EOS]', '[SOS]'])

    content = clean(text)
    bart_start_time = time.time()

    content_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(content))
    if len(content_ids) > 512 - 3 - args.summary_max_len:
        content_ids = content_ids[:512 - 3 - args.summary_max_len]

    input_ids = [CLS] + content_ids + [SEP]

    # To tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    decoder_input_ids = torch.tensor([SOS], dtype=torch.long).unsqueeze(0).to(device)
    decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.long)

    attention_add = torch.ones((1, 1)).long().to(device)

    # Inference
    generated = []

    with torch.no_grad():

        past_key_values = None
        for _ in range(args.summary_max_len):

            if input_ids.shape[-1] > 1024:
                input_ids = input_ids[:, -1024:]
                attention_mask = attention_mask[:, -1024:]

            outputs = model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask, past_key_values=past_key_values)

            logits, past_key_values = outputs['logits'], outputs['past_key_values']

            # get the next token
            next_token_logits = logits[:, -1, :]  # [B, V]

            # reduce repetition
            for token_id in generated:
                next_token_logits[:, token_id] /= args.repetition_penalty

            # to prevent to generate the UNK token
            for logit in next_token_logits:
                logit[UNK] = -float('inf')
                logit[PAD] = -float('inf')
                logit[SEP] = -float('inf')
                logit[CLS] = -float('inf')

            # next_token_logits = top_k_p(next_token_logits, k=args.top_k, p=args.top_p)
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
            next_tokens = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)

            if next_tokens[:, 0] == EOS:
                break

            # Add to generated
            generated.append(next_tokens[0][0].item())

            # Add to decoder_input_ids
            decoder_input_ids = next_tokens
            decoder_attention_mask = torch.cat([decoder_attention_mask, attention_add], dim=-1)

    # print(f'摘要 (长度: {len(generated)}): \n  ', end='')
    # print(''.join(tokenizer.convert_ids_to_tokens(generated)).replace("#", "").strip())
    #
    # bart_end_time = time.time()
    # print(f'Take {round(bart_end_time - bart_start_time, 3)} seconds\n')

    return ''.join(tokenizer.convert_ids_to_tokens(generated)).replace("#", "").strip()

def predict_summary(text, args, tokenizer, model, device):
    CLS, SEP, PAD, UNK = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[PAD]', '[UNK]'])
    EOS, SOS = tokenizer.convert_tokens_to_ids(['[EOS]', '[SOS]'])

    content = clean(text)
    bart_start_time = time.time()

    content_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(content))
    if len(content_ids) > args.content_max_len:
        content_ids = content_ids[:args.content_max_len]

    input_ids = [CLS] + content_ids + [SEP]

    # To tensor
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    
    summary = model.generate(input_ids, 
                            attention_mask=attention_mask,
                            min_length=7,
                            max_length=26,
                            do_sample=True,
                            num_beams=5,
                            early_stopping=True,
                            temperature=1.0,
                            top_k=5,
                            top_p=0.95,
                            repetition_penalty=1.2,
                            pad_token_id=PAD,
                            eos_token_id=EOS,
                            bos_token_id=SOS,
                            decoder_start_token_id=SOS,
                            length_penalty=1.2,
                            no_repeat_ngram_size=3,
                            num_return_sequences=2
                        )


    return tokenizer.batch_decode(summary, skip_special_tokens=True)[0].replace(" ", "").\
            replace('#', '').replace('[EOS]', '').replace('[SOS]', '').strip()
    


if __name__ == '__main__':
    pass

