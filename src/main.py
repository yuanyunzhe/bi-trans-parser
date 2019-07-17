from optparse import OptionParser
import pickle, utils, os, time, sys, argparse
import dynet_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bidirectional Transition-based Dependency Parser")
    group = parser.add_argument_group("Dynet")
    group.add_argument("--dynet-seed", type=int, dest="seed", default=1)
    group.add_argument("--dynet-mem", type=int, dest="cnn_mem", default=2048)

    group = parser.add_argument_group("I/O")
    group.add_argument("--outdir", dest="output", type=str, default="results")
    group.add_argument("--train", dest="train", metavar="FILE", default="data/ptb/small.conllu")
    group.add_argument("--dev", dest="dev", metavar="FILE", default="data/ptb/small.conllu")
    group.add_argument("--test", dest="test", metavar="FILE", default="data/ptb/small.conllu")
    group.add_argument("--params", dest="params", metavar="FILE", default="params.pickle")
    group.add_argument("--model", dest="model", metavar="FILE", default="model")
    group.add_argument("--extrn", dest="external_embedding")

    group = parser.add_argument_group("Embedding")
    group.add_argument("--wembedding", dest="wembedding_dims", type=int, default=100)
    group.add_argument("--pembedding", dest="pembedding_dims", type=int, default=25)

    group = parser.add_argument_group("Learning")
    group.add_argument("--epochs", dest="epochs", type=int, default=30)
    group.add_argument("--k", dest="window", type=int, default=3)
    group.add_argument("--lr", dest="learning_rate", type=float, default=0.1)
    group.add_argument("--lstmdims", dest="lstm_dims", type=int, default=200)
    group.add_argument("--hidden", dest="hidden_units", type=int, default=100)
    group.add_argument("--hidden2", dest="hidden2_units", type=int, default=0)

    group = parser.add_argument_group("Flags only for transition-based")
    group.add_argument("--usehead", action="store_true", dest="headFlag", default=False)
    group.add_argument("--userl", action="store_true", dest="rlMostFlag", default=False)
    group.add_argument("--userlmost", action="store_true", dest="rlFlag", default=False)

    group = parser.add_argument_group("Decoding")
    group.add_argument("--predict", action="store_true", dest="predictFlag", default=False)
    group.add_argument("--dd", action="store_true", dest="ddFlag", default=False)
    group.add_argument("--dd_iter", dest="dd_iter", type=int, default=0)
    group.add_argument("--dd_lr", dest="dd_lr", type=float, default=0.01)
    group.add_argument("--do_lr", dest="do_lr", type=float, default=0.8)

    options = parser.parse_args()
    dynet_config.set(mem=options.cnn_mem,random_seed=options.seed)
    print('Using external embedding:', options.external_embedding)

    if not options.predictFlag:
        from arc_hybrid_train import ArcHybridLSTM
        if not (options.rlFlag or options.rlMostFlag or options.headFlag):
            print('You must use either --userlmost or --userl or --usehead (you can use multiple)')
            sys.exit()
        print('Preparing vocab')
        words, w2i, pos, rels = utils.vocab(options.train)
        with open(os.path.join(options.output, options.params), 'wb') as paramsfp:
            pickle.dump((words, w2i, pos, rels, options), paramsfp)
        print('Finished collecting vocab')
        parser = ArcHybridLSTM(words, pos, rels, w2i, options)
        for epoch in range(options.epochs):
            print('Starting epoch', epoch)
            parser.train(options.train, epoch=epoch)
            parser.save(os.path.join(options.output, options.model + str(epoch + 1)))
            parser.predictAndOutput(epoch, 999)
    else:
        from arc_hybrid_rebuttal1 import ArcHybridLSTM
        with open(options.params, 'rb') as paramsfp:
            words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = options.external_embedding
        stored_opt.output = options.output
        stored_opt.dd_iter = options.dd_iter
        stored_opt.dd_lr = options.dd_lr
        stored_opt.do_lr = options.do_lr
        # dynet_config.set(random_seed=options.seed)
        parser = ArcHybridLSTM(words, pos, rels, w2i, stored_opt)
        parser.load(options.model)
        conllu = (os.path.splitext(options.test.lower())[1] == '.conllu')
        testpath = os.path.join(options.output, 'test_pred.conll' if not conllu else 'test_pred.conllu')
        ts = time.time()
        pred = list(parser.predict(options.test))
        te = time.time()
        utils.write_conll(testpath, pred)
        if not conllu:
            os.system('perl src/utils/eval.pl -g ' + options.test + ' -s ' + testpath + ' > ' + testpath + '.txt')
        else:
            os.system(
                'python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + options.test + ' ' + testpath + ' > ' + testpath + '.txt')
        print('Finished predicting test', te - ts)
        with open(testpath + '.txt', 'r') as f:
            for l in f:
                if l.startswith('UAS'):
                    print('UAS:%s' % l.strip().split()[-1])
                elif l.startswith('LAS'):
                    print('LAS:%s' % l.strip().split()[-1])