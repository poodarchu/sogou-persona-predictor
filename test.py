# -*- coding=utf-8 -*-

import jieba

if __name__ == '__main__':
    seg = jieba.cut('张翰和古力娜扎分手了吗', cut_all=False)

    for i in seg:
        print i