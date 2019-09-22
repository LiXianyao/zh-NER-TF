import os


def conlleval(label_predict, label_path, metric_path):
    """
    。。。用conlleval_rev.pl的脚本对上面的三元组存档的文件进行计算，结果写结果记录文件，然后读出来输出
    (我说怎么这么短） 脚本里要求的标注是BIE0所以做了处理，E可以省略
    :param label_predict:
    :param label_path:
    :param metric_path:
    :return:
    """
    eval_perl = "./conlleval_rev.pl"
    with open(label_path, "w", encoding="utf-8") as fw:
        line = []
        for sent_result in label_predict:
            for char, tag, tag_ in sent_result:
                tag = '0' if tag == 'O' else tag
                tag_ = '0' if tag_ == 'O' else tag_
                #char = char.encode("utf-8")
                line.append("%s %s %s\n" % (char, tag, tag_))
            line.append("\n")
        fw.writelines(line)

    os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    while not os.path.exists(metric_path):
        os.system("perl {} < {} > {}".format(eval_perl, label_path, metric_path))
    with open(metric_path) as fr:
        metrics = [line.strip() for line in fr]
    return metrics
    