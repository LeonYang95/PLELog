import sys

sys.path.extend(["../../", "../", "./"])
import argparse
from approaches.RNN_pipeline import evaluate
from utils.Config import *
from data.DataLoader import *
from model.GRU import AttGRUModel
from utils.ADHelper import AnomalyDetectionBCELoss
import logging


def read_corpus(file, logger):
    data = []
    counter = Counter()
    with open(file, 'r', encoding='utf8') as input_file:
        curtext = []
        for line in input_file.readlines():
            line = line.strip()
            if line == '':
                if len(curtext) == 2:
                    src_events = curtext[0].split()
                    tokens = curtext[1].split(',')
                    if len(tokens) == 3:
                        blk, type, confidence = tokens[0], tokens[1], float(tokens[2])
                    else:
                        blk, type = tokens[0], tokens[1]
                        confidence = None
                    counter[type] += 1
                    instance = parseInstance(src_events, blk, type, confidence)
                    data.append(instance)
                    curtext.clear()
                else:
                    curtext.clear()
            else:
                curtext.append(line)
    slen = len(curtext)
    if slen == 2:
        src_events = curtext[0].split()
        tokens = curtext[1].split(',')
        if len(tokens) == 3:
            blk, type, confidence = tokens[0], tokens[1], float(tokens[2])
        else:
            blk, type = tokens[0], tokens[1]
            confidence = None
        instance = parseInstance(src_events, blk, type, confidence)
        data.append(instance)
        curtext.clear()
    logger.info(file + ":")
    logger.info("Total num: " + str(len(data)))
    for type, count in counter.most_common():
        print(type, count)
    logger.info('****************************')
    return data


if __name__ == '__main__':
    gpu = torch.cuda.is_available()
    # logger.info('GPU Available: %s'%str(gpu))
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='config/BGL.cfg',
                           help='Configuration file for Attention-Based GRU Network.')
    argparser.add_argument('--gpu', default=0, type=int, help='GPU ID if using cuda, -1 if cpu.')
    argparser.add_argument('--input', default=None, type=str, help='Input file for testing.')
    argparser.add_argument('--thread', default=1, type=int, help='Number of thread to use. Default value is 1')
    argparser.add_argument('--threshold', default=0.6, type=float,
                           help='Final threshold of anomaly detection. Any instance with higher score than threshold will be regarded as anomaly. ')
    args, extra_args = argparser.parse_known_args()
    config_file = args.config_file
    input_file = args.input
    threshold = args.threshold
    target_gpu = args.gpu
    thread_num = args.thread
    if input_file is None:
        input_file = 'dataset/BGL/train-6/rd-100_mcs-100_ms-100_random-6/test'
    config = Configurable(config_file, extra_args)
    # Specify logger
    if not os.path.exists('logs'):
        os.makedirs('logs')
    logger_name = 'test.log'
    hdlr = logging.FileHandler(os.path.join('logs', logger_name))
    logger = logging.getLogger('main')
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    test = read_corpus(input_file, logger)
    vocab = Vocab(None)
    logger.info('Load configfile %s' % config_file)
    config = Configurable(config_file, extra_args)
    torch.set_num_threads(thread_num)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)
    config.use_cuda = False
    logger.info('GPU Available: %s' % str(gpu))
    if gpu and target_gpu != -1:
        config.use_cuda = True
        torch.cuda.set_device(target_gpu)
        logger.info('GPU ID:' + str(target_gpu))
    logger.info("GPU using status: " + str(config.use_cuda))
    model = AttGRUModel(vocab, config, vec)
    if config.use_cuda:
        model = model.cuda(target_gpu)
    model.load_state_dict(torch.load(config.load_model_path, map_location=lambda storage, loc: storage))
    classifier = AnomalyDetectionBCELoss(model, vocab)
    evaluate(test, classifier, config, vocab, logger, None, threshold)
