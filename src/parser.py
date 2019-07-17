class BidirectionalParser:
    def __init__(self, words, pos, rels, w2i, options):
        print(options)
        self.output = options.output
        self.dev = options.dev
        self.test = options.test

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.external_embedding = None
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding, 'r', encoding="UTF-8")
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()
            self.edims = len(list(self.external_embedding.values())[0])
            self.noextrn = [0.0 for _ in range(self.edims)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.elookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edims))
            for word, i in self.extrnd.items():
                self.elookup.init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2
            print('Load external embedding. Vector dimensions', self.edims)
        dims = self.wdims + self.pdims + (self.edims if self.external_embedding is not None else 0)

        self.wordsCount = words
        self.vocab = {word: ind+3 for word, ind in w2i.items()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels
        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2
        self.headFlag = options.headFlag
        self.rlMostFlag = options.rlMostFlag
        self.rlFlag = options.rlFlag
        self.k = options.window
        self.nnvecs = (1 if self.headFlag else 0) + (2 if self.rlFlag or self.rlMostFlag else 0)
        
        self.layer1a = [dy.VanillaLSTMBuilder(1, dims, self.ldims, self.model), dy.VanillaLSTMBuilder(1, dims, self.ldims, self.model)]
        self.layer2a = [dy.VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model), dy.VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]
        self.layer1b = [dy.VanillaLSTMBuilder(1, dims, self.ldims, self.model), dy.VanillaLSTMBuilder(1, dims, self.ldims, self.model)]
        self.layer2b = [dy.VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model), dy.VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]
        
        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units
        self.wlookup = self.model.add_lookup_parameters((len(words) + 3, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))

        self.word2lstm1 = self.model.add_parameters((self.ldims * 2, self.wdims + self.pdims + (self.edims if self.external_embedding is not None else 0)))
        self.word2lstmbias1 = self.model.add_parameters((self.ldims * 2))
        self.word2lstm2 = self.model.add_parameters((self.ldims * 2, self.wdims + self.pdims + (self.edims if self.external_embedding is not None else 0)))
        self.word2lstmbias2 = self.model.add_parameters((self.ldims * 2))
        
        self.hidLayer1 = self.model.add_parameters((self.hidden_units, self.ldims * 2 * self.nnvecs * (self.k + 1)))
        self.hidBias1 = self.model.add_parameters((self.hidden_units))
        self.hid2Layer1 = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.hid2Bias1 = self.model.add_parameters((self.hidden2_units))
        self.outLayer1 = self.model.add_parameters((3, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.outBias1 = self.model.add_parameters((3))
        self.rhidLayer1 = self.model.add_parameters((self.hidden_units, self.ldims * 2 * self.nnvecs * (self.k + 1)))
        self.rhidBias1 = self.model.add_parameters((self.hidden_units))
        self.rhid2Layer1 = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.rhid2Bias1 = self.model.add_parameters((self.hidden2_units))
        self.routLayer1 = self.model.add_parameters((2 * (len(self.irels) + 0) + 1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.routBias1 = self.model.add_parameters((2 * (len(self.irels) + 0) + 1))

        self.hidLayer2 = self.model.add_parameters((self.hidden_units, self.ldims * 2 * self.nnvecs * (self.k + 1)))
        self.hidBias2 = self.model.add_parameters((self.hidden_units))
        self.hid2Layer2 = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.hid2Bias2 = self.model.add_parameters((self.hidden2_units))
        self.outLayer2 = self.model.add_parameters((3, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.outBias2 = self.model.add_parameters((3))
        self.rhidLayer2 = self.model.add_parameters((self.hidden_units, self.ldims * 2 * self.nnvecs * (self.k + 1)))
        self.rhidBias2 = self.model.add_parameters((self.hidden_units))
        self.rhid2Layer2 = self.model.add_parameters((self.hidden2_units, self.hidden_units))
        self.rhid2Bias2 = self.model.add_parameters((self.hidden2_units))
        self.routLayer2 = self.model.add_parameters((2 * (len(self.irels) + 0) + 1, self.hidden2_units if self.hidden2_units > 0 else self.hidden_units))
        self.routBias2 = self.model.add_parameters((2 * (len(self.irels) + 0) + 1))

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.populate(filename)

    def init(self):
        evec = self.elookup[1] if self.external_embedding is not None else None
        paddingWordVec = self.wlookup[1]
        paddingPosVec = self.plookup[1] if self.pdims > 0 else None
        paddingVec = dy.tanh(self.word2lstm1.expr() * dy.concatenate(list(filter(None, [paddingWordVec, paddingPosVec, evec]))) + self.word2lstmbias1.expr())
        self.empty1 = paddingVec if self.nnvecs == 1 else dy.concatenate([paddingVec for _ in range(self.nnvecs)])
        paddingVec = dy.tanh(self.word2lstm2.expr() * dy.concatenate(list(filter(None, [paddingWordVec, paddingPosVec, evec]))) + self.word2lstmbias2.expr())
        self.empty2 = paddingVec if self.nnvecs == 1 else dy.concatenate([paddingVec for _ in range(self.nnvecs)])

    def train(self, path, epoch=0):
        mloss, eloss, etotal = 0, 0, 0
        start = time.time()
        with open(path, 'r', encoding="UTF-8") as conllFP:
            data = list(read_conll(conllFP, True))
            random.shuffle(data)
            errs = []
            self.init()
            for i, sentence in enumerate(data):
                if i % 1000 == 0 and i != 0:
                    print('Sentence:', i, 'Loss:', eloss / etotal, 'Time:', time.time() - start)
                    eloss = 0.0
                    etotal = 0
                # loss = self.trainer(sentence)

                sentence1 = copy.deepcopy(sentence)
                sentence2 = copy.deepcopy(sentence)
                sentence1 = sentence1[1:] + [sentence1[0]]
                sentence2 = reverse(sentence2[1:]) + [sentence2[0]]
                self.embedding(sentence1, True)
                self.embedding(sentence2, True)

                detotal, dloss = self.trainSentence(sentence1, errs, l2r=True)
                etotal += detotal
                eloss += dloss
                mloss += dloss
                detotal, dloss = self.trainSentence(sentence2, errs, l2r=False)
                etotal += detotal
                eloss += dloss
                mloss += dloss

                if len(errs) > 50:
                    eerrs = dy.esum(errs)
                    scalar_loss = eerrs.scalar_value()
                    eerrs.backward()
                    self.trainer.update()
                    errs = []
                    dy.renew_cg()
                    self.init()
        if len(errs) > 0:
            eerrs = dy.esum(errs)
            eerrs.scalar_value()
            eerrs.backward()
            self.trainer.update()
            errs = []
        self.trainer.update()
        print("Loss: ", mloss / i)
