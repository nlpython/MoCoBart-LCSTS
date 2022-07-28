import os

class Args(object):
    def __init__(self):

        # Data settings
        self.data_dir = 'data/'
        self.pretrained_model_path = 'bart-base-chinese/'
        self.vocab_path = 'bart-base-chinese/'
        self.json_path = 'bart-base-chinese/config.json'
        self.checkpoint_path = 'checkpoints/'
        self.log_path = 'logs/'

        # Training settings
        self.seed = 42
        self.batch_size = 64
        self.max_len = 184
        self.content_max_len = 152
        self.summary_max_len = 32
        self.epochs = 3
        self.bart_lr = 1e-4
        self.warmup_steps = 10000
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-8
        self.max_clip_norm = 1.0

        self.gradient_accumulation_steps = 1
        self.eval_interval = 2000
        self.log_interval = 50

        self.alpha = 5  # rate of cl loss
        self.K = 1600  # number of negative samples
        self.m = 0.999
        self.T = 0.07  # temperature of softmax
        self.mlp = False
        self.pooling = 'last-avg'
        self.chunk_nums = 30
        self.smoothing = 0.1

        # Model settings
        self.generate_max_len = 70
        self.repetition_penalty = 1.2
        self.top_k = 5
        self.top_p = 0.95


    def print_args(self, logger):
        for attr in dir(self):
            if attr != 'print_args' and not attr.startswith('__'):
                logger.info('{} = {}'.format(attr, getattr(self, attr)))

    def save_settings(self, output_dir):
        with open(os.path.join(output_dir, 'settings.txt'), 'w') as f:
            for attr in dir(self):
                if attr != 'print_args' and not attr.startswith('__') and attr != 'save_settings':
                    f.write('{} = {}\n'.format(attr, getattr(self, attr)))


    def __repr__(self):
        return str(self.__dict__)

if __name__ == '__main__':
    args = Args()
    args.save_settings('./logs')


 