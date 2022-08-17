---
title: SSH Tunnel 实现 BI 工具连接数据库
date: 2022-06-19 17:57:38
categories: [DataScience, Database]
tags: [ssh tunnel, mysql, powerbi, tableau]
---

# 背景

最近项目需要 BI 工具看板做一个数据可视化的模块，服务器和数据库都是在云端的，每次访问数据库通过 Navicat 、Python 等工具去访问，但是很可惜，power BI 和 tableau 并没有 ssh 访问的入口（ Navicat 是有的）。关于如何通过这种方式访问数据库在网上没有符合预期的解答，后来经过研究，发现了答案。

# 原理

SSH 的理论在此不多赘述。整个连接过程分为两部分：

- 服务器/远程主机某端口与本机某端口通过 ssh tunnel 连接，连接建立后我去访问本机某端口时，相当于访问服务器/远程主机对应端口。
- 在 BI 工具（其实 Navicat 也可以，稍有不同）上访问本机的这个端口，而不是访问之前的 3306（ Mysql 默认的）端口。

# 实操过程

## 隧道搭建

我的系统时 Windows 11，系统本身没有命令直接搭建 ssh 隧道。所以需要选择一款 ssh 客户端软件，我用的是 PuTTY （ 0.77 release 64-bit x86 ），下载地址如下：

[https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)

下载完了之后，我们打开 PuTTY，首先是主页面，也就是左侧 `Session` 的页面，这里有个 Host Name (or IP address) 和 Port。

![配置 Session](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192238821.png)

这里的 **Host Name 与 Port 是指的 ssh 服务器的 IP 地址与端口**，对应着 Navicat 如下位置：

![对应的 ssh 服务器](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192244406.png)

接着在左侧`Connection -> SSH -> Tunnels`这个界面，输入 Source port 与 Destination，输入完之后，点击 Add 添加。

![配置单向隧道的源端口和目的端口](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192255705.png)

这里的 **Source port 是指本机要用来建立隧道通信的端口**，我的建议是不要选太怪的数字，最好 10000 以上，65535 以下，也没有太大讲究，最好别和其它服务重合了。**Destination 则是远程主机/服务器的 IP 和端口**，我在 Google 里其实搜索到过搭隧道 tableau 连接远程主机的，它在演示的时候，自己跟自己建立的隧道，稍微不太方便理解这里的意思。Destination 对应着 Navicat 如下位置：

![对应的目的端口](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192305530.png)

然后在 PuTTY 里还需要配置一个地方，它位于 `Connection -> SSH -> Auth`，这个地方是添加私钥的地方，如下：

![配置私钥](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192309718.png)

它对应着 Navicat 里的：

![对应私钥位置](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192310985.png)

**但是其实并不能直接添加 `pem` 文件**，这个时候我们需要打开 PuTTYgen，点击 load，弹出文件选择页面时选择你的 pem 文件，然后点击 Save private key 保存一个 `ppk` 格式的文件。

![私钥格式转换](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192315126.png)

![image-20220619231701159](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192317180.png)

这个时候就可以在 PuTTY 的页面添加这个 `ppk` 格式文件了，如图：

![成功配置私钥](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192318792.png)

回到 `Session ` 点击 `Save` 保存一下以免之后重新配（只用一次当我没说），配置方面就完成了。点击 `Open` 测试一波：

![建立连接，登录账户](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192321988.png)

输入用户名，这个用户名是 ssh 的用户名，对应着 Navicat 如下位置：

![对应的 ssh 账户名](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192322747.png)

然后其实就连上了。注意连上了之后不要关闭 PuTTY ，关了隧道就断开了。

有时可能会突然挂掉，我采用了每隔一段时间发送空包的方式维持隧道。在 `Connection` 里，我把如下位置的值改成了 10：

![保持隧道连接](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192328488.png)

至此，ssh 客户端（ PuTTY ）这边的工作就结束了，下一步就是测试。

## 连接测试

因为我用 Navicat 是可以连接上的，所以我的选择是先用 Navicat 测试。有过 Navicat 通过 ssh 连接数据库经验的同学应该都知道，用这种方式连接数据库时，需要在 `常规` 和 `SSH` 两个页面分别配置，搭建隧道之后，只用配置 `常规` 页面就可以了。

![Navicat 测试](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192349956.png)

这里：

- 主机填写 `localhost` 或是 `127.0.0.1`  都可以。
- 端口填写之前**在 PuTTY 里填写的 Source port**。
- 用户名填写目的主机/服务器的用户名。
- 密码填写目的主机/服务器的密码。

![连接成功](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192349512.png)

接着在 power BI 上也测试一下。选择 MySQL 数据库。

![power BI 选择数据库](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192355774.png)

**服务器填写 `127.0.0.1:xxxx`，就是本机的 IP 地址加用于搭隧道的端口，也就是 PuTTY 里配置的 Source port 。**数据库填写你要连接的库名。

![power BI 连接数据库](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206192357463.png)

高级选项下可以写 SQL 查询语句。然后在下一步选择 `数据库`，用户名填写目的主机/服务器的用户名，密码填写目的主机/服务器的密码，如下：

![power BI 登录数据库账户](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202206200000501.png)



点击连接就可以了。

