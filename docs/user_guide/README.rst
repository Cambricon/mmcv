文档写作规范
--------------------------------

* 用户手册必须用 **rst** 语法写作， **rst** 语法学习参见wiki：http://wiki.cambricon.com/pages/viewpage.action?pageId=12387858。
* 用户手册写作规范参见wiki：http://wiki.cambricon.com/pages/viewpage.action?pageId=12387678
* 执行 ``./makelatexpdf.sh`` 编译文档，编译文档所需环境参见wiki：http://wiki.cambricon.com/pages/viewpage.action?pageId=12387475。
* 文档交付规范参见wiki：http://wiki.cambricon.com/pages/viewpage.action?pageId=27906620。
* 研发提交的文档初稿，以gitlab分支的形式提交，必须有3个赞才能合入。其中必须有文档工程师的赞，和除文档写作人之外的其它研发专家的赞才能合入。

文档写作注意事项
------------------------

* 新增内容建议写到新的文件夹下。文件夹最好以英文命名，以清晰易维护为主，没其它特殊要求。
* 将新增文件添加到工程根目录下的 ``index.rst`` 主索引文件中，否则新增内容将无法包含在最终文档中。
* 文档中如果用到图片，图片的命名必须以英文命名，以兼容不同的服务器环境。
* 如果图片为使用visio、drawio等工具自己画的图片，需要导出pdf和png两种图片，引用图片时无需写具体扩展名，以“.*”代替即可。sphinx当编译pdf文档时自动选择pdf格式图片，当编译html时，自动选择png图片。
  因为pdf格式图片为矢量图，在pdf文件中放大缩小不会模糊。



