#-*-encoding:utf8-*-#

def content_completion(content):
    u"""
    实体标注的内容补全，包括:
    1\句首有多余的实体标注的，去掉
    2\句尾没有实体标注的，补上一个</o>
    3\ “发行人”忘记标注成</no>的，补上
    4\句子里有标注写漏了/的，补上
    """
    #content = content.decode("utf-8")
    content = content.replace("<o", "</o")
    content = content.replace('<n', '</n')
    if content[0:4] == "</o>":  # 句首有多余的标注
        content = content[4:]
    if content[-1] != ">":  # 句尾漏了标注
        content += "</o>"

    content = fill_org(content)
    return content


def fill_org(content):
    index = content.find(u"发行人")
    while index != -1:
        #  首先检查后面是否用了</n?>
        #print index
        next_index = lambda idx: idx + 3
        next_tag_index = content.find("</", next_index(index), -1)
        next_tag = content[next_tag_index + 2]
        if next_tag != "n":  # 发行人没有被标识为一个实体，或者被一个别的非O实体所包含
            # 则需要断开，在“发行人”前面加入 </o>， 后面加入</no>
            if next_tag_index == next_index(index):  # 如果是发行人后面刚好跟着一个</o>，就改成</no>
                content = content[:next_tag_index + 2] + "n" + content[next_tag_index + 2:]
            else:  # 否则直接插入一个新的</o>
                #print "%d\ %s \%s " % (next_index(index), content[index], content[next_index(index)])
                #print content[:next_index(index)]
                #print content[next_index(index):]
                content = content[:next_index(index)] + "</no>" + content[next_index(index):]
            #print content, index
            if index:  # 不位于句首，前面可以插入</o>
                if content[index - 1] != ">":  # 如果前面有别的标签就不插
                    content = content[:index] + "</o>" + content[index:]
                    index += 4
        #print content
        index = content.find(u"发行人", next_index(index), -1)
    return content

if __name__=="__main__":
    c = content_completion
    s = u"报告期内，发行人不存在向单一客户销售比例超过 </o>50.00%</np>或严重依赖少数客户的情况"
    print(c(s))