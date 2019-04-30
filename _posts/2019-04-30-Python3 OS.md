---
layout:     post
title:      Python3 OS
subtitle:   五一节的代码狂欢
date:       2019-04-30
author:     WEW
header-img: img/post-bg-art.jpg
catalog: true
tags:
    - Python3
    - 语言
    - 编程
---

## 文件目录方法

|方法|描述|
|-----|-----|
|os.getcwd()|查看当前工作目录|
|os.chdir()|改变当前工作目录|
|os.chmod(path, mode)|更改目录权限|
|os.close(fd)|关闭当前文件描述符|
|os.dup()|复制当前文件描述符|
|os.listdir(path)|列举当前path路径下所有的文件及目录|
|os.makedirs(path[, mode])|递归式创建path目录|
|os.mkdir(path[, mode])|创建单个目录|
|os.open(file, flags[, mode])|打开文件|
|os.pathconf(path,name)|返回相关路径下文件的系统配置信息|
|os.read(fd, n)|从文件描述符fd中读取n个字符|
|os.remove(path)|删除path路径下的文件，如果path为文件夹则抛出异常|
|os.removedirs(path)|递归删除目录|
|os.rename(src, dst)|重命名文件从src到dst|
|os.renames(old, new)|递归式重命名|
|os.rmdir(path)|删除path指定的空目录，目录不为空抛出异常|
|os.write(fd, str)|写入字符串到文件描述符 fd中. 返回实际写入的字符串长度|
|os.walk(top[, topdown=True[, onerror=None[, followlinks=False]]])|输出在文件夹中的文件名通过在树中游走，向上或者向下。|
