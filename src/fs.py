with open("try3/test_pred.conllu") as f, open("try2/test_pred.conllu") as g:
    ff = f.readlines()
    gg = g.readlines()
    x, y = 0.0, 0.0
    fs = [] 
    for l1, l2 in zip(ff, gg):
        if l1 != "\n":
            w1 = l1.split("\t")[6]
            w2 = l2.split("\t")[6]
            if w1 == w2:
                x += 1
            y += 1
        else:
            fs.append(x / y)
            x, y = 0.0, 0.0
    print(sum(fs) / len(fs))