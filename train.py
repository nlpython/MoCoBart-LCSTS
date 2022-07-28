import os
import json
import torch
from loguru import logger
from processing.dataset import BartDataset
from processing.tools import seed_everything, AverageMeter
from config import Args
from transformers import BertTokenizer, BartForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch import nn
from torch.nn.utils import clip_grad_norm_
from model import MoCoBart
from inference import predict_summary
from rouge import Rouge
from tqdm import tqdm

def train():

    args = Args()
    seed_everything(args.seed)
    logger.add('%s/{time}.log' % args.log_path)

    args.print_args(logger)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    tokenizer.add_tokens('[SOS]', special_tokens=True)
    tokenizer.add_tokens('[EOS]', special_tokens=True)

    # define dataloader
    batch_size = int(args.batch_size / args.gradient_accumulation_steps)


    # eval_dataset = BartDataset(tokenizer, args, mode='test')
    # eval_dataloader = DataLoader(
    #     dataset=eval_dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     collate_fn=eval_dataset.collate_fn
    # )
    eval_set = json.loads(open('./data/PART_III_clean.json', 'r', encoding='utf-8').read())

    # define model
    model = MoCoBart(args.pretrained_model_path, args.K, args.m, args.T, args.mlp, args.pooling).to(device)

    # define criterion
    mle_criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    cl_criterion = nn.CrossEntropyLoss()

    # define optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.bart_lr)
    # total_steps = int(len(train_dataset) * args.epochs / args.gradient_accumulation_steps)
    total_steps = int(2400591 / batch_size / args.gradient_accumulation_steps * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)

    # Begin training
    logger.info('\n')
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = {}".format(args.epochs))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(
        args.batch_size))
    logger.info("  Type of optimizer = {}".format(optimizer.__class__.__name__))
    # logger.info("  Total optimization steps = {}".format(len(train_dataloader) * args.epochs))
    logger.info("  Learning rate = {}".format(args.bart_lr))
    logger.info('\n')


    best_eval_loss = {'loss': float('inf'), 'epoch': -1, 'step': -1}
    best_rouge_scores = {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0, 'epoch': -1, 'step': -1}
    loss_record = AverageMeter()
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for chunk in range(args.chunk_nums):
            train_dataset = BartDataset(tokenizer, args, chunk, mode='train')
            train_dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
                collate_fn=train_dataset.collate_fn
            )
            
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                X, y_q, mle_labels = batch

                mle_logits, cl_logits, cl_labels = model(X, y_q)

                mle_loss = mle_criterion(mle_logits.view(-1, mle_logits.shape[-1]), mle_labels.view(-1))
                cl_loss = cl_criterion(cl_logits, cl_labels)
                loss = mle_loss + args.alpha * cl_loss

                # record loss
                loss_record.update(mle_loss, cl_loss, loss)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                global_step += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    clip_grad_norm_(model.parameters(), args.max_clip_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    

                if step % args.log_interval == 0:
                    logger.info(
                        "Epoch: {:2d} | Chunk: {:2d} | step: {:4d} | mle_loss: {:.6f} | cl_loss: {:.6f} | loss: {:.6f}".format(
                            epoch, chunk, step, *loss_record.get_loss()))
                    loss_record.reset()

                if global_step % args.eval_interval == 0:
                    # evaluate on test set
                    eval_loss, rouge_scores = evaluate(model, device, None, args, tokenizer,
                                            cl_criterion, do_generate=True, eval_set=eval_set[:500])

                    if eval_loss < best_eval_loss['loss']:
                        best_eval_loss['loss'] = round(eval_loss, 4)
                        best_eval_loss['epoch'] = epoch
                        best_eval_loss['step'] = step
                    if rouge_scores['rouge-1'] > best_rouge_scores['rouge-1']:
                        best_rouge_scores = rouge_scores
                        best_rouge_scores['epoch'] = epoch
                        best_rouge_scores['step'] = step

                    logger.info("Eval loss: {}, the best loss: {}".format(eval_loss, best_eval_loss))
                    logger.info("Rouge scores: {}, the best scores: {}".format(rouge_scores, best_rouge_scores))

                    output_dir = os.path.join(args.checkpoint_path, 'MoCoBart-{}-{}-{}'.format(epoch, chunk, step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    logger.info("Saving model checkpoint to %s" % output_dir)
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    args.save_settings(output_dir)
                    model.train()
                
                # free memory
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()


def evaluate(model, device, dataloader, args, tokenizer,
             cl_criterion, do_generate=False, eval_set=None):

    model.eval()

    total_loss, total_step = 0.0, 0.0

    with torch.no_grad():
        # for step, batch in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
        #     batch = tuple(t.to(device) for t in batch)
        #     X, y_q, mle_labels = batch

        #     mle_loss, cl_logits, cl_labels = model(X, y_q, mle_labels)

        #     cl_loss = cl_criterion(cl_logits, cl_labels)
        #     loss = mle_loss + args.alpha * cl_loss

        #     total_loss += loss.item()
        #     total_step += 1
        total_step = 1

        if do_generate and eval_set is not None:
            metrics = Rouge()
            rouge_1, rouge_2, rouge_l = 0.0, 0.0, 0.0
            logger.info("Generating...")
            for sample in tqdm(eval_set):
                text = sample['content']
                gold_summary = sample['summary']
                pred_summary = predict_summary(text, args, tokenizer, model, device)

                rouge_scores = metrics.get_scores(' '.join(list(pred_summary)), ' '.join(list(gold_summary)))[0]
                rouge_1 += rouge_scores['rouge-1']['f']
                rouge_2 += rouge_scores['rouge-2']['f']
                rouge_l += rouge_scores['rouge-l']['f']
            logger.info("Generation finished.")

            rouge_scores = {
                'rouge-1': round(rouge_1 / len(eval_set), 4),
                'rouge-2': round(rouge_2 / len(eval_set), 4),
                'rouge-l': round(rouge_l / len(eval_set), 4)
            }
            logger.info("Rouge scores: {}".format(rouge_scores))

    return total_loss / total_step, rouge_scores



if __name__ == '__main__':
    train()












