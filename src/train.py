class Trainer:
    def __init__(self):
        pass

    def trainSentence(self, sentence, errs, l2r=True):
        detotal, dloss = 0, 0
        self.buildNet(sentence, l2r=l2r)
        stack = ParseForest([])
        buf = ParseForest(sentence)
        for root in sentence:
            root.lstms = [root.vec for _ in range(self.nnvecs)]
        hoffset = 1 if self.headFlag else 0
        while not (len(buf) == 1 and len(stack) == 0):
            scores = self.evaluate(stack, buf, True, l2r=l2r)
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