---
title: FastAPI
date: 2023-05-12 05:57:22
categories: [ComputerScience, Python, Backend]
tags: [python, fastapi]
---

# 基础

## WEB 基础

| 请求方法 | 说明                                                  |
| -------- | ----------------------------------------------------- |
| GET      | 请求 URL 的网页信息，并返回实体数据。一般用于查询数据 |
| POST     | 向 URL 提交数据进行处理请求。比如提交表单或是上传文件 |
| PUT      | 向 URL 上传数据内容                                   |
| DELETE   | 向 URL 发送删除资源请求                               |
| HEAD     | 与 GET 请求类似，只返回响应头                         |
| CONNECT  | HTTP/1.1 预留                                         |
| OPTIONS  | 获取服务器特定资源的 HTML 请求方法                    |
| TRACE    | 回复并显示服务器收到的请求                            |
| PATCH    | 对 PUT 方法的补充，用来对已知资源进行局部更新         |

URL （ Uniform Resource Locator，资源定位符）代表一个网站上资源的详细地址。完整的 URL 由以下部分依序组成：

1. 协议，如 HTTPS 协议。
2. 主机名，可能是域名，如 `www.google.com` ，也可能是 IP 地址加端口号，如 `192.168.1.1:80` 。
3. 资源相对路径，指网站上资源相对的地址，可能还带有参数。

HTTP 请求与响应详细实现步骤如下：

1. 客户端 TCP 连接到 Web 服务器。
2. 客户端发送 HTTP 请求。
3. Web 服务器接受请求并返回 HTTP 响应。
4. 释放 TCP 连接。
5. 客户端解析响应数据。

HTTP 请求大致组成如下：

- 请求行（请求方法，URL，协议版本）
- 请求头
- 空行
- 请求数据

HTTP 响应大致组成如下：

- 状态行（协议版本，状态码，状态码描述）
- 响应头
- 空行
- 响应收据

| 状态码 | 说明                                                 |
| ------ | ---------------------------------------------------- |
| 100    | 不带有响应对象的业务数据，需要请求后执行后续相关操作 |
| 200    | 成功                                                 |
| 201    | 也是成功，一般用于表示在数据库中创建了一条新的记录   |
| 204    | 提示客户端服务端成功处理，但没有返回内容             |
| 300    | 重定向                                               |
| 400    | 客户端异常                                           |
| 500    | 服务器异常                                           |

## FastAPI 框架组成

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202305120725715.png)

FastAPI 功能建立在 Python 类型提示（3.7, 3.9, 3.10 可能写法不一样）、Pydantic 框架、 Starlette 框架基础上。

### Python 类型提示

#### 使用方式

```python
def func(data: int = 0) -> str:
    res: str = str(data)
    return res
```

变量或参数后面使用冒号指定类型，不过运行时没用，也是防君子不防小人的一种体现。

#### 基础数据类型

- int - 整型
- float - 浮点型
- str - 字符串型
- bool - 逻辑型
- bytes - 字节型

#### 泛型

- list - 列表泛型

- tuple - 元组泛型
- set - 集合泛型
- dict - 字典泛型

```python
from typing import List, Optional
from datetime import datetime

# 3.7
my_list: List[str] = []
timestamp: Optional[datetime] = None

# 3.9
my_list: list[str] = []
timestamp: Optional[datetime] = None

# 3.10
my_list: list[str] = []
timestamp: datetime | None = None
```

#### 自定义类

```python
class Person:
    def __init__(self, name: str):
        self.name = name


# output: movis
print(Person('movis').name)
```

### Pydantic 框架

[Pydantic](https://docs.pydantic.dev/latest/) 是一套基于 Python 类型提示的数据模型定义及验证的框架。

这个框架在**运行时**强制执行类型提示。

### Starlette 框架

Starlette 是一个轻量级高性能的异步服务网关接口框架（ASGI）。

# 请求

1. 用户向浏览器提交请求数据。
2. 浏览器封装请求数据冰箱 Web 服务器提交。
3. Web 服务器处理请求数据。

## 路径参数

```python
@app.get('/file/{file_name}')
async def read_file_path(file_name: str):
    print(file_name)
    return {'file_name': file_name}
```

写在 `get` 请求路径里的 `{file_name}` 这部分就是路径参数，可以定义类型，有类型路径参数的数据会有验证。

### 路由访问顺序

与代码里路由顺序有关，具体为从上到下。

```python
@app.get('/file/test')
async def read_test_file():
    return {'file_name': 'this is test'}


@app.get('/file/{file_name}')
async def read_file_path(file_name: str):
    print(file_name)
    return {'file_name': file_name}
```

如上代码的访问顺序是如果 `/file/test` 会执行 `read_test_file()` 而不是 `read_csv_file_path()` 。

### 枚举的应用

```python
from enum import Enum

class FileEnum(Enum):
    file1 = "file_one"
    file2 = "file_two"


@app.get('/file/{file_name}')
async def read_csv_file_path(file_name: FileEnum):
    print(file_name)
    return {'file_name': file_name}
```

### 路径参数类

当想要给路径参数添加约束条件时（校验路径参数）可以使用 `Path` 这个查询参数类。这样代码更加规范。

```python
from fastapi import Path

@app.get('/item/{item_id}')
async def main(item_id: int = Path(..., gt=3)):
    return item_id
```

这样可以指定，`item_id` 必传并且必须大于 3 。

具体用法可以看 [Path Parameters and Numeric Validations - FastAPI](https://fastapi.tiangolo.com/tutorial/path-params-numeric-validations/) 。

## 查询参数

指 `path?p1=v1&p2=v2` 这种形式。

```python
@app.get('/file/{file_name}')
async def read_csv_file_path(file_name: FileEnum, q: str):
    print('path parameter:', file_name)
    print('query parameter', q)
    return {'file_name': file_name}
```

在这个示例中，`file_name` 是路径参数，`q` 这是查询参数，可以看到没有出现在 `app.get` 后的路径里。

那么请求 `URL` 是 `xxx/file/file_one?q=v` 这种形式。

如果 `read_csv_file_path` 这个函数声明里写了 `q` 的默认值，那么这个查询参数就是一个可选的查询参数，否则就是必选查询参数。将这个示例改写为可选查询参数为：

```python
async def read_csv_file_path(file_name: FileEnum, q: Optional[str] = None):
# 其实 q: str = None 这样写也可以
```

### 参数类型转换

通过 `URL` 传递的查询参数，参数值的原始类型是字符串，如果这个函数定义的是 `int` 或者是 `bool` 那么 FastAPI 会验证参数并转换为对应的类型。

其中 `bool` 类型可以将如下字符串转义为 `True` 或者 `False` ：

- True
  - `q=True` , `q=true` , `q=1` , `q=yes` ，注意这里 1 这个值是可以的，123 这种是不行的。
- False
  - `q=False` , `q=false` , `q=0` , `q=no`

对于多路径参数和多查询参数的情况，路径参数的顺序必须是正确的，查询参数的顺序可以随意调换。

### 查询参数类

当想要给查询参数添加约束条件时（校验查询参数）可以使用 `Query` 这个查询参数类。这样代码更加规范。

比如：

```python
from fastapi import Query

@app.get('/')
async def main(q: Optional[str] = Query(None, min_length=3)):
    return q
```

具体用法可以看 [Query Parameters and String Validations - FastAPI](https://fastapi.tiangolo.com/tutorial/query-params-str-validations/) 。

## 请求体

客户端发送给服务端的受数据模型约束的请求数据称为请求体（ Request Body ），默认是 JSON 形式，**请求体不能使用 `GET` 方法**，需要使用 `POST` 、`PUT` 、`DELETE` 、`PATCH` 方法之一。

### 对象请求体

在 FastAPI 中，所有请求体实现对象，通过创建类继承 Pydantic 的 BaseModel 类实现定义。

假设我需要一个路径为 `/items/` 的接口，请求体格式为：

```json
{
    name: 'xxx',
    num: 1
}
```

这两个参数并将其返回给客户端，那么写法为：

```python
class Item(BaseModel):
    name: str
    num: int


@app.post('/items/')
async def created_item(item: Item):
    return item
```

值得注意的是，由于 `Item` 类已经定义了数据类型，比如 `num` 为 `int` ，假设 `num` 没传数字是会报错的。

### 常规数据类型请求体

上面定义了继承 Pydantic 的 `schema` 类，因为请求体里是个对象，但是请求体也可能不是个对象，是基础数据类型，可以这样写：

```python
from fastapi import Body

@app.post('/items/')
async def created_item(item: str = Body()):
    return item
```

这里主要体现了和查询参数的区别，如果不写这个 `Body` ，很显然这个 `item` 就是个查询参数。

### 表单和文件

表单和文件实际上也是在请求体中传给服务端的，都依赖第三方库 `python-multipart` 。

表单比较常见的应用就是登录时的提交。

写法为：

```python
@app.post('/login/')
async def login(username: str = Form(), password: str = Form()):
    return {'username': username, 'password': password}
```

官网教程有用过 `username: Annotated[str, Form()]` 这种写法， 这样的话 `username` 就不再是请求体里的内容而是查询参数。

从上面可以看到，其实请求体实质上是以一个对象的方式传递的，包括自定义继承 `Pydantic` 的 `BaseModel` 的对象，`Body` 对象，`Form` 对象。

上传文件也是用类似的方式，即 `File` 对象。这里有两种方式接收文件。

第一种的参数类型为 bytes，用来接收 HTTP 上传的文件流。

```python
@app.post('/upload/')
async def upload_file(file: bytes = File()):
    return file[:20]
```

第二种的参数类型为 UploadFile ，这种情况下，当内存中的数据尺寸超过最大限制后，会将部分数据存储在磁盘中。这种方式比较适合处理大文件，比如图片，视频等。还可以获得文件名。

```python
from fastapi import File, UploadFile

@app.post('/upload/')
async def upload_file(file: UploadFile = File()):
    return file.filename
```

## Cookie 参数

HTTP Cookie 是服务器发送到客户端并保存在本地的一小块数据，一般用来记录用户登录状态和浏览器行为。在 FastAPI 中可以通过 `Cookie` 参数处理 `Cookie` 。

```python
from fastapi import Cookie

@app.get('/item/')
async def main(c: Optional[str] = Cookie(None)):
    return c
```

然后需要在浏览器设置 `Cookie` ：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306081045370.png)

返回内容如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306081046482.png)

URL 里是没有传 Cookie 的信息的。

具体可以见 [Cookie Parameters - FastAPI](https://fastapi.tiangolo.com/tutorial/cookie-params/) 。

## 请求头

在 FastAPI 中可以通过 `Header` 参数处理请求头（响应头其实也可以）。

```python
from fastapi import Header

@app.get('/item/')
async def main(user_agent: Optional[str] = Header()):
    return user_agent
```

这样就可以拿到 `User-Agent` 。

具体可以见 [Header Parameters - FastAPI](https://fastapi.tiangolo.com/tutorial/header-params/) 。

## 请求类

在某些情况下，不需要对数据进行校验和转换，可以使用 FastAPI 中的 `Request` 类。

比如需要返回客户端的 IP ：

```python
from fastapi import Request

@app.get('/')
async def main(request: Request):
    return request.client.host
```

# 响应

1. Web 服务器解析请求数据。
2. 数据逻辑处理。
3. 服务器将响应数据发送回浏览器。
4. 浏览器渲染响应数据。

## 响应模型

在请求中，请求体中的数据以数据模型形式传递给服务端，好处是可以自动做数据验证。在响应中其实也有同样的方式传递给客户端。

```python
class UserIn(BaseModel):
    name: str
    password: str


class UserOut(BaseModel):
    name: str


@app.post('/signup/', response_model=UserOut)
async def create_customer(user: UserIn):
    return user
```

只需要在 `app.post` 或者 `app.get` 装饰器中添加参数 `response_model = YOUR_SCHEMA` 就可以了。

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306080709699.png)

当然，请求模型和响应模型可以是同一个模型。

在上述例子中，可以通过继承的方式简化数据模型的定义（比如 `UserIn` 和 `UserOut` 继承同一个基类 `UserBase` ），也可以通过`reponse_model` 使用 `typing` 里的 `Union` 使用多种响应模型，按顺序匹配直至匹配为止。

一般来讲，数据模型可以分为三类，请求数据模型，响应数据模型，业务数据模型（跟数据库有关）。

## 响应类

路径操作函数在返回响应数据时，可以返回基础数据类型、泛型、数据模型的数据，FastAPI 会将不同类型的数据都转换成兼容 JSON 格式的字符串，再通过响应对象返回给客户端。除此之外，也可以再路径操作函数中直接使用 FastAPI 的内置响应类，返回特殊类型的数据，比如 XML、HTML、文件等。具体响应类型如下：

### 纯文本响应

```python
from fastapi.responses import PlainTextResponse

@app.get('/', response_class=PlainTextResponse)
async def main():
    return "Response"
```

返回的内容为：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306080823675.png)

可以看到这里浏览器是没有渲染的。

### HTML 响应

```python
from fastapi.responses import HTMLResponse

@app.get('/', response_class=HTMLResponse)
async def main():
    return "Response"
```

返回的内容为：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306080824082.png)

可以看到这里浏览器已经渲染了。

可以打开开发者工具观察和上一个的不同，体现在：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306080825151.png)

另外，这样写也是可以的：

```python
from fastapi.responses import HTMLResponse

@app.get('/')
async def main():
    return HTMLResponse("Response")
```

### 重定向响应

HTTP 重定向也是 Web 服务中常见的响应方式，此方式不返回数据，仅返回一个新的 URL 地址，比如百度：

```python
from fastapi.responses import RedirectResponse

@app.get('/', response_class=RedirectResponse)
async def main():
    return RedirectResponse('https://www.baidu.com/')
```

返回的响应就变成了百度的页面。但是这里更重要的是这个 `return` 后这个网址要写全，这里写 `www.baidu.com` 不加协议是不行的。

### JSON 响应

```python
from fastapi.responses import JSONResponse

@app.get('/', response_class=JSONResponse)
async def main():
    return JSONResponse(content={1: 'xx'}, status_code=404)
```

可以看到 `JSONResponse` 这个类可以更改 `status_code` （其实 `XXResponse` 类都可以），不同公司可能对不同错误要求返回不同 `status_code` ，这里就可以派上用场。

### 通用响应

根据自己需要的 `media_type` 自己写，比如 `application/json` 或 `application/xml` 等 :

```python
from fastapi.responses import Response

@app.get('/', response_class=Response)
async def main():
    return Response(content='{"1": "xx"}', media_type='application/json')
```

其实**剩下的响应都是继承这个响应的**。这里有几个参数可以了解：

- `content` ：要响应的内容，可以是 `str` 或者是 `bytes` 类型。
- `status_code` ：HTTP 状态码，`int` 类型。
- `headers` ：响应头，`dict` 类型。
- `media_type` ：媒体类型的文本，`str` 类型，可以在 IANA 查看 [Media Types](https://www.iana.org/assignments/media-types/media-types.xhtml) 。

### 流响应

指使用字节流进行传输。字节流响应的内容是二进制格式，这里以 `csv` 文件举例（可以说明和文件响应的不同之处）。

```python
from fastapi.responses import StreamingResponse

@app.get('/', response_class=StreamingResponse)
async def main():
    f = open('file_like.csv', mode='rb')
    return StreamingResponse(f, media_type='text/csv') # 这里需要指定媒体文件类型
```

这样下载的文件不会带有文件名的信息。

### 文件响应

```python
from fastapi.responses import FileResponse

@app.get('/', response_class=FileResponse)
async def main():
    return FileResponse('file_like.csv', media_type='text/csv', filename='file_like.csv')
```

与流响应相比，文件响应可以接收更多的参数：

- `path` ：路径，而非文件对象。
- `filename` ：文件名，会包含在响应头的 `Content-Disposition` 中。

使用 `StreamingResponse` 类时，需要先将文件打开，载入文件对象进行返回，文件内容是一次性读取的，如果文件很大，就会占用很大的内存。使用 `FileResponse` 类时，通过文件路径指定生成了一个 `FileRespnse` 类实例，文件是异步读取的，会占用更少的内存。所以推荐使用文件响应。

## 自定义 Cookie

服务器设置 Cookie 的方式是先创建响应实例，再调用响应实例的 `set_cookie` 方法。

比如：

```python
@app.get('/item/', response_class=JSONResponse)
async def main(request: Request):
    response = JSONResponse(content={'res': '1'})
    response.set_cookie(key='used_id', value='123')
    return response
```

调用后可以看到，在浏览器已经返回 `cookie` ：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306081222439.png)

## 响应头

和自定义 Cookie 类似，如果想自定义响应头，需要在实例化 `Response` 类时传入 `header` 这个参数，如：

```python
@app.get('/item/', response_class=JSONResponse)
async def main(request: Request):
    response = JSONResponse(content={'res': '1'}, headers={'User-Agent': 'Test Server'})
    return response
```

可以看到：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306081225963.png)

在设置响应头时，有三个规定：

1. 设置服务器内置的 Header 的名称可以直接设置，比如这里的 `User-Agent` 。
2. 设置自定义的 Header 的名称要以 **`X-`** 开头。
3. 自定义 Header 的名称和内容都不要包含下划线 `_` ，因为很多服务器会默认过滤下划线。

## 响应码

自定义响应状态码可以在实例化 `Response` 类时传入，比如：

```python
@app.get('/item/', response_class=JSONResponse)
async def main(request: Request):
    response = JSONResponse(content={'res': '1'}, status_code=203)
    return response
```

当然可以通过导入 `status` 这个模块，在写代码时能更清楚想向客户端传递什么信息，比如上述改写成：

```python
from fastapi import status

@app.get('/item/', response_class=JSONResponse)
async def main(request: Request):
    response = JSONResponse(content={'res': '1'}, status_code=status.HTTP_203_NON_AUTHORITATIVE_INFORMATION)
    return response
```

两者效果都如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306081237061.png)

# 异常处理

FastAPI 提供了异常处理机制，是为了对异常信息的抛出和处理进行统一管理，增加代码可读性。

## HTTPException

使用 `raise` 关键字抛出 `HTTPException` 异常，如下：

```python
from fastapi import HTTPException

@app.get('/item/{item_id}')
async def main(item_id: int):
    if item_id == 3:
        raise HTTPException(status_code=404, detail='Error item_id')
    return {'item_id': item_id}
```

当路径参数中的 `item_id` 为 3 时，客户端会收到一条如下的响应：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306120506790.png)

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306120506339.png)

## 全局异常管理器

用 `HTTPException` 的方式抛出异常，在大型项目中不太方便管理。为了实现逻辑处理与异常处理的分离，FastAPI 提供了一种全局异常处理器的方式。

```python
class TestException(Exception): # 定义异常类
    def __init__(self, name: str) -> None:
        self.name = name


@app.exception_handler(TestException) # 注册全局异常管理器
async def test_exception_handler(request: Request, exc: TestException): # 定义异常处理函数
    return JSONResponse(status_code=404, content={'detail': 'Error item_id.'})


@app.get('/item/{item_id}')
async def main(item_id: int):
    if item_id == 3:
        raise TestException(str(item_id))
    return {'item_id': item_id}
```

通过这种写法，得到的效果和上面 `HTTPException` 的方式是一样的。但是异常处理这部分，也就是 `test_exception_handler` 这个函数已经和逻辑处理分离开，而且异常信息的格式也可以自定义，比如不用 `JSONResponse` 而改用 `PlainTextResponse` 就可以返回一段文本而不是 json 格式的数据。

## RequestValidationError

这里主要是 `RequestValidationError` ，与前面请求模型里的数据格式验证相对应，当数据格式验证失败时，也可以采取自定义的方式返回错误信息。上面这个例子，如果不自定义，当数据格式验证错误时：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306120555362.png)

在某些时候需要自定义这个信息，比如在公司里可能需要约定正确和错误的格式，这时就可以自定义：

```python
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=200, content={"error_code": "-2"})
```

这样的话当数据格式验证错误时，返回的 `Status Code` 仍然是 200，而错误信息变成了：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306120558009.png)

# 中间件

FastAPI 中间件实际上是服务端的一种函数，在每个请求处理之前被调用，又在每个响应返回给客户端之前被调用。（也就是函数内部不再需要自己调用）

## 自定义中间件

如下是一个在 `header` 里添加 `X-Process-Time` 的中间件：

```python
@app.middleware('http')
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    end_time = time.time()
    response.headers['X-Process-Time'] = str(end_time - start_time)
    return response
```

这样在给服务端发请求时：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306120636748.png)

## CORSMiddleware

针对前后端分离的软件项目开发方式，有一种称为 CORS （ Cross-Origin Resource Sharing ，跨域资源共享）的机制，用于**保护后端服务的安全**。

这里所说的**域**指 HTTP 协议、主机名、端口的组合。多见于前端访问后端接口访问不通。

针对这种情况，我们可以如下操作：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:9000'], # 这里写前端访问的网址
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
```

以下是 CORSMiddleware 的配置说明：

- `allow_origins` ：允许跨域请求的源域列表，使用 `['*']` 代表允许任何源。
- `allow_origin_regex` ：使用正则表达式匹配的源允许跨域请求。
- `allow_methods` ：允许跨域请求的 HTTP 方法列表，默认为 `['GET']` ，可以使用 `['*']` 来允许所有标准方法。
- `allow_headers` ：允许跨域请求的 HTTP 请求头列表，默认为空，可以使用 `[*]` 来允许所有请求头。
- `allow_credentials` ：是否支持跨域请求用 cookies，默认 `False` 不支持。如果设置为 `True` ，则 `allow_origins` 不能为 `['*']` 。
- `expose_headers` ：指示可以被浏览器访问的响应信息头，默认为 `[]` 。
- `max_age` ：设定浏览器缓存 CORS 响应的最长事件，单位是秒，默认值为 600 。

在不配置 CORSMiddleware 时，不允许任何跨域的访问。

## HTTPSRedirectMiddleware

HTTPS 的全称为 Hyper Text Transfer Protocol over Secure Socket Layer，通过安全套接字层的超文本传输协议，在 HTTP 上加入了 SSL 。

该中间件的作用是约束传入的请求地址必须是 HTTPS 开头，对于任何传入的以 HTTP 开头的请求地址，都将被重定向到 HTTPS 开头的地址上。

代码只需要一行：

```python
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

app.add_middleware(HTTPSRedirectMiddleware)
```

## TrustedHostMiddleware

这个中间件用来设置域名访问白名单，和 `CORSMiddleware` 类似：

```python
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(TrustedHostMiddleware, allowed_hosts=['baidu.com'])
```

这样设置后，只允许域名为 `baidu.com` 的主机访问，比如本地的 `127.0.0.1` 并不在白名单里，如果访问就会：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202306121020121.png)

## GZipMiddleware

这个一个请求头有 `Accept-Encoding:GZip`  时，对响应数据进行压缩，再发送给客户端，客户端拿到响应，先解压缩的中间件，使用方法也很简单：

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

总的来说，中间件类似于一种通信预处理小工具，可以实现一些基础功能，避免重复造轮子，让开发者更专注于业务逻辑，更多可以看 [Middleware - Starlette](https://www.starlette.io/middleware/) 。

# 依赖注入

# 数据库操作

# 安全机制

# 异步

# 架构

# 测试

# 部署

