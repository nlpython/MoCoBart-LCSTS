from transformers import BertTokenizer, BartForConditionalGeneration, BartModel
import torch
import json
from tqdm import tqdm
from predict import predict_summary, predict_summary1
from rouge import Rouge
from config import Args
from processing.tools import seed_everything

tokenizer = BertTokenizer.from_pretrained('checkpoints/MoCoBart-2-21-749', do_lower_case=True)
tokenizer.add_tokens('[SOS]', special_tokens=True)
tokenizer.add_tokens('[EOS]', special_tokens=True)
model = BartForConditionalGeneration.from_pretrained('checkpoints/MoCoBart-2-21-749')
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
args = Args()
seed = 2020
seed_everything(seed)


eval_set = json.loads(open('./data/PART_III_clean.json', 'r', encoding='utf-8').read())
def compute_rouge():
    rouge = Rouge()
    result = []
    rouge_1, rouge_2, rouge_l = 0, 0, 0
    for sample in tqdm(eval_set, desc='Computing ROUGE'):
        ref = sample['summary']
        hyp = predict_summary(sample['content'], args, tokenizer, model, device)

        scores = rouge.get_scores(' '.join(list(hyp)), 
                                  ' '.join(list(ref)))
        rouge_1 += scores[0]['rouge-1']['f']
        rouge_2 += scores[0]['rouge-2']['f']
        rouge_l += scores[0]['rouge-l']['f']

        result.append({'content': sample['content'], 'title': ref, 'summary': hyp})

    print('rouge-1: {}'.format(rouge_1 / len(eval_set)))
    print('rouge-2: {}'.format(rouge_2 / len(eval_set)))
    print('rouge-l: {}'.format(rouge_l / len(eval_set)))

    result.append({
        'seed': seed,
        'rouge-1': rouge_1 / len(eval_set),
        'rouge-2': rouge_2 / len(eval_set),
        'rouge-l': rouge_l / len(eval_set)
    })

    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)


if __name__ == '__main__':
    compute_rouge()
