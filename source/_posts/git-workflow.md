---
title: Git 工作流的使用
date: 2022-10-04 11:53:09
categories: [ComputerScience, Git]
tags: [git, github]
---

# Git 三个组成部分

- Remote(repository) - 远程代码仓库。
- Local Git - 本地的 git 仓库。有所有本机告诉 git 的信息。分为本地代码仓库和暂存区。
  - 本地仓库 - 区分分支。
  - 暂存区 - 不区分分支。
- Disk - 本地磁盘。

# 工作流

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202210041528216.png)

> 以下的 source 指的是源分支。一般来讲是 develop 分支，也可能是 master(main) 分支，以下用 source 代替。

### clone

```shell
git clone xxx.git
```

在本地复制一个跟远程仓库一模一样的代码仓库。

### branch and checkout

```shell
git checkout -b feature
```

创建并切换到一个 feature 的分支，这样不会影响主分支。

等同于：

```shell
git branch feature 		# 创建
git checkout feature 	# 切换
```

这样做了之后 Local Git 里存有两个分支的文件，而 Disk 里存的是当前分支的文件。

### code changes

修改代码。这里修改的只是 Disk 里的文件，即使是相同分支的 Local Git 里文件依然没有修改。

### diff

```shell
git diff 
```

这个命令是修改完代码后，可以查看 Disk 与同分支 Local Git 暂存区里文件的差异。

### add

```shell
git add file
```

将 Disk 里修改的某文件添加到 Local Git 暂存区里（而不是本地仓库）。

如果要将所有有改动的文件都添加，可以使用：

```shell
git add .
```

### commit

```shell
git commit -m "xxx"
```

将这次改动内容及备注从 Local Git 暂存区提交到 Local Git 本地仓库中。

如果备注不一样可以：

```shell
git commit file1 -m "xxx1"
git commit file2 -m "xxx2"
```

### push

```shell
git push origin feature
```

将 Local Git 本地仓库的新分支 `xxx` 推送到远程代码仓库中。

### pull request

将 feature branch 的改动更新到 source branch 里，这里的 pull 是针对 source branch 来讲的。

这里会经历合并和删分支的过程。

#### squash and merge

squash 是指 commit 的数量和名字有所变动。比如针对 feature 来说可能有很多的 commit ，但是针对 project 来说一个 feature 就是一个 commit 。 merge 是指将 feature branch的代码改动合并到 source branch 里。

#### delete branch

将 feature branch 删掉。但是这时没结束因为本地的 feature branch 并没有删。接下来：

```shell
git checkout source # 切换到 source
git branch -D feature # 删除 Local Git 的 feature
git pull origin source
```

# 协同处理

## 推送新的 feature branch 后发现源 branch 有更新

这时需要测试一下这个新的 feature 在源 branch 的更新之下是否正常，相当于需要把源 branch 的更新同步到新的 feature branch 里。

### checkout - source

```shell
git checkout source
```

切换到源 branch 上。

### pull

```shell
git pull origin source
```

将远程仓库源 branch 的更新拉取到本地。

### checkout - feature

```shell
git checkout feature
```

回到新分支 feature 上，这时 feature 有我们上面 code changes 的改动但是没有 source 的更新。

### rebase

```shell
git rebase source 	# 注意这时的分支已经在 feature 上了
```

先将 code changes 扔到一边，将 source 的更新弄上，再尝试将 code changes 放到更新后的 feature 上。

这时可能出现的情况是有冲突 `rebase conflict` ，需要手动选择需要哪一段代码。

这个步骤结束后 Disk 的 commit 应该是先是 source 的更新 commit ，再是 code change 的 commit 。

### force push

```shell
git push -f origin feature
```

由于做了 `rebase` ，所以在将 Local Git 本地仓库里的代码推送到远程仓库要使用 `force` ，也就是 `-f` 。
