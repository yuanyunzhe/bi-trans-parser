class Model:
    def __init__(self):
        pass
    
    def embedding(self, sentence, train):
        for root in sentence:
            c = float(self.wordsCount.get(root.norm, 0))
            dropFlag = not train or (random.random() < (c / (0.25 + c)))
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

    def evaluate(self, stack, buf, train, l2r=True, do=False):
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
        return ret