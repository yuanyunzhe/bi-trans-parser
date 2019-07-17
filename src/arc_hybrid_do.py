from utils import ParseForest, read_conll, write_conll
from itertools import chain
import dynet as dy
import utils, time, random, shutil, os, copy, math
import numpy as np
import pickle

ninf = -float('inf')
softmax = lambda x: list(np.exp(x) / np.sum(np.exp(x)))


def reverse(sentence, flag=True):
    if flag:
        l = len(sentence)
        res = []
        for entry in reversed(sentence):
            entry.id = l + 1 - entry.id
            if entry.parent_id != None:
                if entry.parent_id != 0:
                    entry.parent_id = l + 1 - entry.parent_id
            if entry.pred_parent_id != None:
                if entry.pred_parent_id != 0:
                    entry.pred_parent_id = l + 1 - entry.pred_parent_id
            if hasattr(entry, "templete_parent_id"):
                if entry.templete_parent_id != None:
                    if entry.templete_parent_id != 0:
                        entry.templete_parent_id = l + 1 - entry.templete_parent_id
            res.append(entry)
        return res
    else:
        return sentence


def getTree(sentence):
    tree = {}
    for i in range(len(sentence)):
        entry = sentence[i]
        tree[entry.id] = (entry.pred_parent_id, entry.pred_relation, entry.parent_id, entry.relation)
    return tree


def setTree(sentence, tree):
    for i in range(len(sentence)):
        entry = sentence[i]
        pred = tree[entry.id]
        entry.pred_parent_id = pred[0]
        entry.pred_relation = pred[1]


def setDO(sentence, tree):
    if tree:
        for i in range(len(sentence)):
            entry = sentence[i]
            pred = tree[entry.id]
            entry.templete_parent_id = pred[0]


def fscore(tree):
    unlabel = sum([1 for x in tree if tree[x][0] == tree[x][2]])
    label = sum([1 for x in tree if (tree[x][0] == tree[x][2] and tree[x][1] == tree[x][3])])
    return unlabel, label


class ArcHybridLSTM:
    def __init__(self, words, pos, rels, w2i, options):
        print(options)
        self.output = options.output
        self.dev = options.dev
        self.test = options.test
        self.dd_iter = options.dd_iter
        self.dd_lr = options.dd_lr
        self.do_lr = options.do_lr

        self.model = dy.Model()
        self.trainer = dy.AdamTrainer(self.model)
        random.seed(1)

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.pdims = options.pembedding_dims
        self.rdims = options.rembedding_dims
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
        
        self.share = options.share
        self.layer1a = [dy.VanillaLSTMBuilder(1, dims, self.ldims, self.model), dy.VanillaLSTMBuilder(1, dims, self.ldims, self.model)]
        self.layer2a = [dy.VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model), dy.VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]  
        if self.share > 0:
            self.layer1b = self.layer1a
        else:
            self.layer1b = [dy.VanillaLSTMBuilder(1, dims, self.ldims, self.model), dy.VanillaLSTMBuilder(1, dims, self.ldims, self.model)]
        if self.share > 1:
            self.layer2b = self.layer2a
        else:
            self.layer2b = [dy.VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model), dy.VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model)]  
        
        self.hidden_units = options.hidden_units
        self.hidden2_units = options.hidden2_units
        self.wlookup = self.model.add_lookup_parameters((len(words) + 3, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))
        self.rlookup = self.model.add_lookup_parameters((len(rels), self.rdims))

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

    def __evaluate(self, stack, buf, train, l2r=True, do=False):
        if l2r:
            hidLayer, hidBias = self.hidLayer1, self.hidBias1
            hid2Layer, hid2Bias = self.hid2Layer1, self.hid2Bias1
            outLayer, outBias = self.outLayer1, self.outBias1
            rhidLayer, rhidBias = self.rhidLayer1, self.rhidBias1
            rhid2Layer, rhid2Bias = self.rhid2Layer1, self.rhid2Bias1
            routLayer, routBias = self.routLayer1, self.routBias1
            self.empty = self.empty1
        else:
            hidLayer, hidBias = self.hidLayer2, self.hidBias2
            hid2Layer, hid2Bias = self.hid2Layer2, self.hid2Bias2
            outLayer, outBias = self.outLayer2, self.outBias2
            rhidLayer, rhidBias = self.rhidLayer2, self.rhidBias2
            rhid2Layer, rhid2Bias = self.rhid2Layer2, self.rhid2Bias2
            routLayer, routBias = self.routLayer2, self.routBias2
            self.empty = self.empty2
        topStack = [stack.roots[-i - 1].lstms if len(stack) > i else [self.empty] for i in range(self.k)]
        topBuffer = [buf.roots[i].lstms if len(buf) > i else [self.empty] for i in range(1)]
        input = dy.concatenate(list(chain(*(topStack + topBuffer))))
        if self.hidden2_units > 0:
            routput = (routLayer.expr() * dy.tanh(rhid2Bias.expr() + rhid2Layer.expr() * dy.tanh(rhidLayer.expr() * input + rhidBias.expr())) + routBias.expr())
        else:
            routput = (routLayer.expr() * dy.tanh(rhidLayer.expr() * input + rhidBias.expr()) + routBias.expr())

        if self.hidden2_units > 0:
            output = (outLayer.expr() * dy.tanh(hid2Bias.expr() + hid2Layer.expr() * dy.tanh(hidLayer.expr() * input + hidBias.expr())) + outBias.expr())
        else:
            output = (outLayer.expr() * dy.tanh(hidLayer.expr() * input + hidBias.expr()) + outBias.expr())

        scrs, uscrs = routput.value(), output.value()
        left_arc_conditions = len(stack) > 0 and len(buf) > 0
        right_arc_conditions = len(stack) > 1 and stack.roots[-1].id != 0
        shift_conditions = len(buf) >0 and buf.roots[0].id != 0
        uscrs0 = uscrs[0]
        uscrs1 = uscrs[1]
        uscrs2 = uscrs[2]
        if train:
            output0 = output[0]
            output1 = output[1]
            output2 = output[2]
            ret = [[(rel, 0, scrs[1 + j * 2] + uscrs1, routput[1 + j * 2] + output1) for j, rel in enumerate(self.irels)] if left_arc_conditions else [],
                   [(rel, 1, scrs[2 + j * 2] + uscrs2, routput[2 + j * 2] + output2) for j, rel in enumerate(self.irels)] if right_arc_conditions else [],
                   [(None, 2, scrs[0] + uscrs0, routput[0] + output0)] if shift_conditions else []]
        else:
            scrs = [x + uscrs1 if i % 2 == 1 else x + uscrs2 for i, x in enumerate(scrs)]
            scrs[0] = scrs[0] - uscrs2 + uscrs0
            # scrs = [float(x) for x in softmax(scrs)]

            if do:
                ret = [[[rel, 0, scrs[1 + j * 2]] for j, rel in enumerate(self.irels)] if left_arc_conditions else [],
                       [[rel, 1, scrs[2 + j * 2]] for j, rel in enumerate(self.irels)] if right_arc_conditions else [],
                       [[None, 2, scrs[0]]] if shift_conditions else []]
            else:
                s1, r1 = max(zip(scrs[1::2], self.irels))
                s2, r2 = max(zip(scrs[2::2], self.irels))
                ret = [[(r1, 0, s1)] if left_arc_conditions else [],
                       [(r2, 1, s2)] if right_arc_conditions else [],
                       [(None, 2, scrs[0])] if shift_conditions else []]

            # if do:
            #     ret = [[[rel, 0, scrs[1 + j * 2] + uscrs1] for j, rel in enumerate(self.irels)] if left_arc_conditions else [],
            #            [[rel, 1, scrs[2 + j * 2] + uscrs2] for j, rel in enumerate(self.irels)] if right_arc_conditions else [],
            #            [[None, 2, scrs[0] + uscrs0]] if shift_conditions else []]
            # else:
            #     s1, r1 = max(zip(scrs[1::2], self.irels))
            #     s2, r2 = max(zip(scrs[2::2], self.irels))
            #     s1 += uscrs1
            #     s2 += uscrs2
            #     ret = [[(r1, 0, s1)] if left_arc_conditions else [],
            #            [(r2, 1, s2)] if right_arc_conditions else [],
            #            [(None, 2, scrs[0] + uscrs0)] if shift_conditions else []]
        return ret

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

    def embedding(self, sentence, train):
        for root in sentence:
            c = float(self.wordsCount.get(root.norm, 0))
            dropFlag =  not train or (random.random() < (c/(0.25 + c)))
            wordvec = self.wlookup[int(self.vocab.get(root.norm, 0)) if dropFlag else 0]
            posvec = self.plookup[int(self.pos[root.pos])] if self.pdims > 0 else None
            if self.external_embedding is not None:
                if root.form in self.external_embedding:
                    evec = self.elookup[self.extrnd[root.form]]
                elif root.norm in self.external_embedding:
                    evec = self.elookup[self.extrnd[root.norm]]
                else:
                    evec = self.elookup[0]
            else:
                evec = None
            root.ivec = dy.concatenate(list(filter(None, [wordvec, posvec, evec])))

    def buildNet(self, sentence, l2r=True):
        if l2r:
            layer1 = self.layer1a
            layer2 = self.layer2a
        else:
            layer1 = self.layer1b
            layer2 = self.layer2b
        forward = layer1[0].initial_state()
        backward = layer1[1].initial_state()
        for froot, rroot in zip(sentence, reversed(sentence)):
            forward = forward.add_input(froot.ivec)
            backward = backward.add_input(rroot.ivec)
            froot.fvec = forward.output()
            rroot.bvec = backward.output()
        for root in sentence:
            root.vec = dy.concatenate([root.fvec, root.bvec])
        bforward = layer2[0].initial_state()
        bbackward = layer2[1].initial_state()
        for froot, rroot in zip(sentence, reversed(sentence)):
            bforward = bforward.add_input(froot.vec)
            bbackward = bbackward.add_input(rroot.vec)
            froot.bfvec = bforward.output()
            rroot.bbvec = bbackward.output()
        for root in sentence:
            root.vec = dy.concatenate([root.bfvec, root.bbvec])

    def trainSentence(self, sentence, errs, l2r=True):
        detotal, dloss = 0, 0
        self.buildNet(sentence, l2r=l2r)
        stack = ParseForest([])
        buf = ParseForest(sentence)
        for root in sentence:
            root.lstms = [root.vec for _ in range(self.nnvecs)]
        hoffset = 1 if self.headFlag else 0
        while not (len(buf) == 1 and len(stack) == 0):
            scores = self.__evaluate(stack, buf, True, l2r=l2r)
            scores.append([(None, 3, ninf ,None)])
            alpha = stack.roots[:-2] if len(stack) > 2 else []
            s1 = [stack.roots[-2]] if len(stack) > 1 else []
            s0 = [stack.roots[-1]] if len(stack) > 0 else []
            b = [buf.roots[0]] if len(buf) > 0 else []
            beta = buf.roots[1:] if len(buf) > 1 else []
            left_cost = (len([h for h in s1 + beta if h.id == s0[0].parent_id]) +
                         len([d for d in b + beta if d.parent_id == s0[0].id])) if len(scores[0]) > 0 else 1
            right_cost = (len([h for h in b + beta if h.id == s0[0].parent_id]) +
                          len([d for d in b + beta if d.parent_id == s0[0].id])) if len(scores[1]) > 0 else 1
            shift_cost = (len([h for h in s1 + alpha if h.id == b[0].parent_id]) +
                          len([d for d in s0 + s1 + alpha if d.parent_id == b[0].id])) if len(scores[2]) > 0 else 1
            costs = (left_cost, right_cost, shift_cost, 1)
            bestValid = max((s for s in chain(*scores) if costs[s[1]] == 0 and (s[1] == 2 or s[0] == stack.roots[-1].relation)), key=lambda x:x[2])
            bestWrong = max((s for s in chain(*scores) if costs[s[1]] != 0 or (s[1] != 2 and s[0] != stack.roots[-1].relation)), key=lambda x:x[2])
            best = bestValid if ((bestValid[2] - bestWrong[2] > 1.0) or (bestValid[2] > bestWrong[2] and random.random() > 0.1) ) else bestWrong
            if best[1] == 2:
                stack.roots.append(buf.roots[0])
                del buf.roots[0]
            elif best[1] == 0:
                child = stack.roots.pop()
                parent = buf.roots[0]
                child.pred_parent_id = parent.id
                child.pred_relation = best[0]
                bestOp = 0
                if self.rlMostFlag:
                    parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                if self.rlFlag:
                    parent.lstms[bestOp + hoffset] = child.vec
            elif best[1] == 1:
                child = stack.roots.pop()
                parent = stack.roots[-1]
                child.pred_parent_id = parent.id
                child.pred_relation = best[0]
                bestOp = 1
                if self.rlMostFlag:
                    parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                if self.rlFlag:
                    parent.lstms[bestOp + hoffset] = child.vec
            if bestValid[2] < bestWrong[2] + 1.0:
                loss = bestWrong[3] - bestValid[3]
                dloss += 1.0 + bestWrong[2] - bestValid[2]
                errs.append(loss)
            detotal += 1
        return detotal, dloss

    def predictSentence(self, sentence, loss, l2r=True, rev=False):
        sentence = reverse(sentence[:-1], rev) + [sentence[-1]]
        self.buildNet(sentence, l2r=l2r)
        stack = ParseForest([])
        buf = ParseForest(sentence)
        for root in sentence:
            root.lstms = [root.vec for _ in range(self.nnvecs)]
        hoffset = 1 if self.headFlag else 0
        while not (len(buf) == 1 and len(stack) == 0):
            scores = self.__evaluate(stack, buf, False, l2r=l2r)
            best = max(chain(*scores), key=lambda x:x[2])
            loss.append(best[2])
            if best[1] == 2:
                stack.roots.append(buf.roots[0])
                del buf.roots[0]
            elif best[1] == 0:
                child = stack.roots.pop()
                parent = buf.roots[0]
                child.pred_parent_id = parent.id
                child.pred_relation = best[0]
                bestOp = 0
                if self.rlMostFlag:
                    parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                if self.rlFlag:
                    parent.lstms[bestOp + hoffset] = child.vec
            elif best[1] == 1:
                child = stack.roots.pop()
                parent = stack.roots[-1]
                child.pred_parent_id = parent.id
                child.pred_relation = best[0]
                bestOp = 1
                if self.rlMostFlag:
                    parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                if self.rlFlag:
                    parent.lstms[bestOp + hoffset] = child.vec
        sentence = reverse(sentence[:-1], rev) + [sentence[-1]]

    def predictWithDD(self, sentence, loss, rev=False, l2r=True, dd=None, do=None, th=0):
        if dd:
            penalty, selection = dd
        if do:
            tree = do
            setDO(sentence, tree)

        sentence = reverse(sentence[:-1], rev) + [sentence[-1]]
        stack = ParseForest([])
        buf = ParseForest(sentence)
        for root in sentence:
            root.lstms = [root.vec for _ in range(self.nnvecs)]
        hoffset = 1 if self.headFlag else 0
        while not (len(buf) == 1 and len(stack) == 0):
            scores = self.__evaluate(stack, buf, False, l2r=l2r, do=True)
            if dd:
                for score in chain(*scores):
                    if score[1] == 0:
                        child = stack.roots[-1]
                        parent = buf.roots[0]
                    if score[1] == 1:
                        child = stack.roots[-1]
                        parent = stack.roots[-2]
                    if score[1] != 2:
                        id = len(sentence) - child.id if rev and child.id != 0 else child.id
                        head = len(sentence) - parent.id if rev and parent.id != 0 else parent.id
                        score[2] += penalty[id, head] * selection[id, head]
            if do:
                scores.append([(None, 3, ninf ,None)])
                alpha = stack.roots[:-2] if len(stack) > 2 else []
                s1 = [stack.roots[-2]] if len(stack) > 1 else []
                s0 = [stack.roots[-1]] if len(stack) > 0 else []
                b = [buf.roots[0]] if len(buf) > 0 else []
                beta = buf.roots[1:] if len(buf) > 1 else []
                left_cost = (len([h for h in s1 + beta if h.id == s0[0].templete_parent_id]) +
                            len([d for d in b + beta if d.templete_parent_id == s0[0].id])) if len(scores[0]) > 0 else 1
                right_cost = (len([h for h in b + beta if h.id == s0[0].templete_parent_id]) +
                            len([d for d in b + beta if d.templete_parent_id == s0[0].id])) if len(scores[1]) > 0 else 1
                shift_cost = (len([h for h in s1 + alpha if h.id == b[0].templete_parent_id]) +
                            len([d for d in s0 + s1 + alpha if d.templete_parent_id == b[0].id])) if len(scores[2]) > 0 else 1
                costs = (left_cost, right_cost, shift_cost, 1)
                for s in chain(*scores):
                    if (costs[s[1]] == 0) and (s[1] == 2 or s[0] == stack.roots[-1].relation):
                        s[2] += th
                best = max(chain(*scores), key=lambda x:x[2])

                # best = bestValid if ((bestValid[2] - bestWrong[2] > 1.0) or (bestValid[2] > bestWrong[2] and random.random() > th)) else bestWrong
                # if th == 0:
                #     best = bestValid
            else:
                best = max(chain(*scores), key=lambda x:x[2])
            loss.append(best[2])
            if best[1] == 2:
                stack.roots.append(buf.roots[0])
                del buf.roots[0]
            elif best[1] == 0:
                child = stack.roots.pop()
                parent = buf.roots[0]
                child.pred_parent_id = parent.id
                child.pred_relation = best[0]
                bestOp = 0
                if self.rlMostFlag:
                    parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                if self.rlFlag:
                    parent.lstms[bestOp + hoffset] = child.vec
            elif best[1] == 1:
                child = stack.roots.pop()
                parent = stack.roots[-1]
                child.pred_parent_id = parent.id
                child.pred_relation = best[0]
                bestOp = 1
                if self.rlMostFlag:
                    parent.lstms[bestOp + hoffset] = child.lstms[bestOp + hoffset]
                if self.rlFlag:
                    parent.lstms[bestOp + hoffset] = child.vec
        sentence = reverse(sentence[:-1], rev) + [sentence[-1]]

    def train(self, path, epoch=0):
        mloss, eloss, etotal = 0.0, 0.0, 0
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

    def predict(self, path):
        u1, l1, u2, l2, u_dd, l_dd, u_h, l_h, u_b, l_b = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        n1_h, n2_h, n1_b, n2_b = 0, 0, 0, 0
        tot = 0
        scores = np.zeros([self.dd_iter, 2])
        start = time.time()
        with open(path, 'r', encoding="UTF-8") as conllFP:
            for i, sentence in enumerate(read_conll(conllFP, False)):
                if i % 100 == 0 and i > 0:
                    print("---------------------------------------------")
                    print("Sentence:", i, "Time:", time.time() - start)
                    print("Forward: UAS {:.2f} LAS {:.2f}".format(u1 / tot * 100, l1 / tot * 100))
                    print("Backward: UAS {:.2f} LAS {:.2f}".format(u2 / tot * 100, l2 / tot * 100))
                    print("Hybrid: UAS {:.2f} LAS {:.2f}".format(u_h / tot * 100, l_h / tot * 100), n1_h, n2_h)
                    print("DD: UAS {:.2f} LAS {:.2f}".format(u_dd / tot * 100, l_dd / tot * 100))
                    print("Bound: UAS {:.2f} LAS {:.2f}".format(u_b / tot * 100, l_b / tot * 100), n1_b, n2_b)
                    # print("++++++++")
                    # print(s1)
                    # print(s2)
                    # print("Accuracy:", sum([1 for c1, c2 in zip(s1, s2) if c1 == c2])/ len(s1))
                
                loss1, loss1r, loss2, loss2r = [], [], [], []
                sentence = sentence[1:] + [sentence[0]]
                sentence1 = copy.deepcopy(sentence)
                sentence2 = copy.deepcopy(sentence)
                self.dd(sentence, scores)
                tree_dd = getTree(sentence)

                dy.renew_cg()
                self.init()
                sentence2 = reverse(sentence2[:-1]) + [sentence2[-1]]
                self.embedding(sentence1, False)
                self.embedding(sentence2, False)
                sentence2 = reverse(sentence2[:-1]) + [sentence2[-1]]
                self.predictSentence(sentence1, loss1, l2r=True, rev=False)
                self.predictSentence(sentence2, loss2, l2r=False, rev=True)
                tree1 = getTree(sentence1)
                tree2 = getTree(sentence2)
                self.predictWithDD(sentence2, loss1r, l2r=False, rev=True, do=tree1, th=100)
                self.predictWithDD(sentence1, loss2r, l2r=True, rev=False, do=tree2, th=100)

                sentence = [sentence[-1]] + sentence[:-1]
                # print(sum(loss1), sum(loss1r), sum(loss2), sum(loss2r))

                du1, dl1 = fscore(tree1)
                du2, dl2 = fscore(tree2)
                du_dd, dl_dd = fscore(tree_dd)
                u1, l1 = u1 + du1, l1 + dl1
                u2, l2 = u2 + du2, l2 + dl2
                u_dd, l_dd = u_dd + du_dd, l_dd + dl_dd
                if du1 > du2:
                    u_b, l_b = u_b + du1, l_b + dl1
                    n1_b += 1
                else:
                    u_b, l_b = u_b + du2, l_b + dl2
                    n2_b += 1
                if sum(loss1 + loss1r) > sum(loss2 + loss2r):
                    u_h, l_h = u_h + du1, l_h + dl1
                    n1_h += 1
                    setTree(sentence, tree1)
                else:
                    u_h, l_h = u_h + du2, l_h + dl2
                    n2_h += 1
                    setTree(sentence, tree2)
                setTree(sentence, tree_dd)
                tot += len(sentence) - 1

                yield sentence
        print("Forward: UAS {:.2f} LAS {:.2f}".format(u1 / tot * 100, l1 / tot * 100))
        print("Backward: UAS {:.2f} LAS {:.2f}".format(u2 / tot * 100, l2 / tot * 100))
        print("Hybrid: UAS {:.2f} LAS {:.2f}".format(u_h / tot * 100, l_h / tot * 100), n1_h, n2_h)
        print("DD: UAS {:.2f} LAS {:.2f}".format(u_dd / tot * 100, l_dd / tot * 100))
        print("Bound: UAS {:.2f} LAS {:.2f}".format(u_b / tot * 100, l_b / tot * 100), n1_b, n2_b)

        # scores = scores / tot * 100
        # with open(os.path.join(self.output, "dd" + str(self.dd_lr) + "do" + str(self.do_lr)), 'wb') as paramsfp:
        #     pickle.dump(scores, paramsfp)

    def dd(self, sentence, scores):
        trees = []
        u, y, z = {}, {}, {}
        t = 0
        max_uas, max_pos = 0, 0
        for id in range(len(sentence)):
            for head in range(len(sentence)):
                u[id, head] = 0
                y[id, head] = 0
                z[id, head] = 0
        dy.renew_cg()
        self.init()

        sentence1 = copy.deepcopy(sentence)
        sentence2 = copy.deepcopy(sentence)
        sentence2 = reverse(sentence2[:-1]) + [sentence2[-1]]
        self.embedding(sentence1, False)
        self.embedding(sentence2, False)
        self.buildNet(sentence1, l2r=True)
        self.buildNet(sentence2, l2r=False)
        sentence2 = reverse(sentence2[:-1]) + [sentence2[-1]]

        tree1, tree2 = None, None
        for iter in range(self.dd_iter):
            loss1, loss2, loss1r, loss2r = [], [], [], []
            # self.predictWithDD(sentence1, loss1, l2r=True, rev=False, dd=(u, y), th=0.1)
            # self.predictWithDD(sentence2, loss2, l2r=False, rev=True, dd=(u, z), th=0.1)
            self.predictWithDD(sentence1, loss1, l2r=True, rev=False, do=tree2, th=self.do_lr)
            self.predictWithDD(sentence2, loss2, l2r=False, rev=True, do=tree1, th=self.do_lr)
            tree1 = getTree(sentence1)
            tree2 = getTree(sentence2)

            flag = True
            for id in range(len(sentence)):
                for head in range(len(sentence)):
                    y[id, head] = 0
                    z[id, head] = 0
            for k in tree1:
                pred = tree1[k]
                y[k, pred[0]] = 1
            for k in tree2:
                pred = tree2[k]
                z[k, pred[0]] = 1

            pre_loss = sum(loss1 + loss2)
            for id in range(len(sentence)):
                for head in range(len(sentence)):
                    if y[id, head] != z[id, head]:
                        flag = False

                    u[id, head] -= self.dd_lr / (1 + t) * (y[id, head] - z[id, head])

            if sum(loss1 + loss2) > pre_loss:
                t += 1

            self.predictWithDD(sentence2, loss1r, l2r=False, rev=True, do=tree1, th=100)
            self.predictWithDD(sentence1, loss2r, l2r=True, rev=False, do=tree2, th=100)

            trees.append((tree1, sum(loss1) + sum(loss1r)))
            trees.append((tree2, sum(loss2) + sum(loss2r)))
            tree = max(trees, key=lambda x:x[1])[0]
            scores[iter] += fscore(tree)

            uas1, _ = fscore(tree1)
            uas2, _ = fscore(tree2)
            if max(uas1, uas2) > max_uas:
                max_uas = max(uas1, uas2)
                max_pos = iter

            if flag:
                scores[iter + 1:] += fscore(tree)
                break

        # print(flag, iter)
        if flag:
            setTree(sentence, tree1)
        else:
            tree = max(trees, key=lambda x:x[1])[0]
            setTree(sentence, tree)


    def predictAndOutput(self, epoch, batch):
        bestdev = os.path.join(self.output, "best_dev.txt")
        besttest = os.path.join(self.output, "best_test.txt")
        if os.path.exists(bestdev):
            with open(bestdev, 'r') as f:
                for l in f:
                    if l.startswith('UAS'):
                        best_UAS = float(l.strip().split()[-1])
                    elif l.startswith('LAS'):
                        best_LAS = float(l.strip().split()[-1])
        else:
            best_UAS, best_LAS = 0, 0
        conllu = (os.path.splitext(self.dev.lower())[1] == '.conllu')
        devpath = os.path.join(self.output, 'dev_epoch_' + str(epoch + 1) + '.' + str(batch) + '.conllu')
        testpath = os.path.join(self.output, 'test_epoch_' + str(epoch + 1) + '.' + str(batch) + '.conllu')
        utils.write_conll(devpath, self.predict(self.dev))
        utils.write_conll(testpath, self.predict(self.test))
        os.system('python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + self.dev + ' ' + devpath + ' > ' + devpath + '.txt')
        os.system('python src/utils/evaluation_script/conll17_ud_eval.py -v -w src/utils/evaluation_script/weights.clas ' + self.test + ' ' + testpath + ' > ' + testpath + '.txt')
        flag = False
        print("Dev:")
        with open(devpath + '.txt', 'r') as f:
            for l in f:
                if l.startswith('UAS'):
                    print('UAS:%s' % l.strip().split()[-1])
                    local_UAS = float(l.strip().split()[-1])
                elif l.startswith('LAS'):
                    print('LAS:%s' % l.strip().split()[-1])
        print("Test:")
        with open(testpath + '.txt', 'r') as f:
            for l in f:
                if l.startswith('UAS'):
                    print('UAS:%s' % l.strip().split()[-1])
                elif l.startswith('LAS'):
                    print('LAS:%s' % l.strip().split()[-1])

        if local_UAS > best_UAS:
            self.save(os.path.join(self.output, 'best.model'))
            shutil.copyfile(devpath + '.txt', bestdev)
            shutil.copyfile(testpath + '.txt', besttest)