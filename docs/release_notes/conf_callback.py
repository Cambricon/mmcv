#!/usr/bin/python
# -*- coding: UTF-8

from __future__ import print_function

# sphinx事件回调函数
#如果copy到conf.py文件中，只copy下面的代码即可。

import os
import os.path as osp
import sys
import importlib
import traceback
import datetime as dt
import locale

#如果文档编译失败，请检查下面默认的路径是否正确，请根据实际路径进行修改

curpath = os.path.dirname(os.path.realpath(__file__))

if osp.exists(osp.join(osp.dirname(__file__), './parsejson.py')):
    sys.path.append(osp.join(osp.dirname(__file__), '.'))
else:
    sys.path.append(osp.join(osp.dirname(__file__), '../'))

cusdirect = None
extpathisexist = False  # ./_ext/customdirective.py是否存在的标志，用于设置指令扩展

extpath = osp.join(osp.dirname(__file__), './_ext')
if osp.exists(extpath):
    # 自定义指令所在路径，加到搜索目录，否则无法使用自定义指令
    sys.path.append(extpath)
    if osp.exists(osp.join(osp.dirname(__file__),r'./_ext/customdirective.py')):
        extpathisexist = True
        cusdirect = importlib.import_module('customdirective')
else:
    extpath = osp.join(osp.dirname(__file__), '../_ext')
    sys.path.append(extpath)
    if osp.exists(osp.join(osp.dirname(__file__),r'../_ext/customdirective.py')):
        extpathisexist = True
        cusdirect = importlib.import_module('customdirective')

import sphinx
import sphinx.errors as sperror
import parsejson

try:
    import breathe # 仅为了判断版本号，根据breathe版本号添加不同配置
    breathe_version = breathe.__version__[:3]
except Exception as e:
    breathe_version = ""

warnfile = ''  # 告警文件，不包含路径
warnfilepath = ''  # 保存告警日志文件，包含完整的路径名

months=['January','February','March','April','May','June','July',
        'August','September','October','Novmber','December']

gmaketitle = r'''
    \pagenumbering{Roman} 
    \begin{titlepage}
        \centering
        \vspace*{40mm} 
        \textbf{\Huge {%(titlename)s}}
        
        \vspace{10mm}
        \textbf{\Large{%(release)s}}
        
        \vfill
        \textbf{\large{%(today)s}}
    \end{titlepage}
   '''

def __ModifyMakeTitle(elementskeys,config):
    '''
    如果有maketitle变量则修改maketitle的内容
    :param config: 配置变量
    :return: 
    '''
    #得到封面名称
    titlename = config.latex_documents[0][2]

    if 'releasename' in elementskeys: 
        releasename = config.latex_elements['releasename']
    else:
        releasename = ''

    if hasattr(config,'version'):
        version = config.version
    else:
        version = ''
    release = releasename +' ' + version
    if hasattr(config,'today') and len(config.today) > 0:
        today = config.today
    else:
        todayft = dt.date.today()
        if hasattr(config,'language') and config.language=='zh_CN':
            today = todayft.strftime('%Y 年 %m 月 %d 日')
        else:
            #除中文外，一律用英文时间格式
            #得到英文月的全称。
            month = months[int(todayft.strftime('%m'))-1]
            today = month + todayft.strftime(" %d, %Y")

    if len(config.latex_elements['maketitle'].strip())==0:
        maketitle = gmaketitle
    else:
        maketitle = config.latex_elements['maketitle']
    #必须转一下，否则赋值不成功
    config.latex_elements['maketitle'] = maketitle % {
        'titlename': titlename,
        'release': release,
        'today': today
    }
def docview_modifyconfig(config):
    """
    以下配置需要在config_init事件中配置，否则有些变量无法读取到，导致配置不生效
    比如：latex_documents的配置必须在config_init中才能读取得到
    :param config: 
    :return: 
    """
    #配置静态路径
    if len(config.html_static_path)==0:
        #如果配置了，就以配置路径为准，否则默认提供两个可选路径
        if osp.exists(curpath + '/_static'):
            config.html_static_path = [curpath + '/_static']
        else:
            config.html_static_path = [curpath + '/../_static']
        
    texfilename = config.latex_documents[0][1]  #得到tex文件名，构建pdf文件名
    filename,fileext = os.path.splitext(texfilename)
    pdfname = filename+'.pdf' 
    #设置js脚本
    config.html_js_files = [('custom.js', {'pdfname': pdfname}),'cndeveloper.js']
    config.html_css_files = ['custom.css']
    config.html_permalinks = True
    
def modifycommonconfig(config):
    """
    以下配置为通用配置，为了配置生效，放在setup函数中调用。
    注意：以下配置因为放在setup函数中调用，因此该配置在conf.py配置之前就会被调用
    因此如果conf.py中又做了相同字段的配置，则以conf.py中的配置为准，以下配置将会被覆盖
    但是像html_theme的配置必须放在setup函数中，因为他会提前被读取用来编译html，在config_init中不再允许被修改
    :param config: 
    :return: 
    """
    #设置html通用配置
    #设置html最新更新时间
    config.html_last_updated_fmt = '%Y-%m-%d %H:%M:%S'
    config.html_theme = 'sphinx_rtd_theme'
    config.html_copy_source = False
    config.html_show_sphinx = False
    
    #设置latex通用配置
    config.latex_use_latex_multicolumn = True
    config.latex_use_xindy = False
    config.latex_engine = 'xelatex'
    #修改通用配置
    config.smartquotes = False
    config.numfig = True
    config.numfig_secnum_depth = 1


    
def config_inited_handler(app, config):
    # 检查年份是否正确，并修改。需配合最新parsejson.py文件使用，否则不支持该接口。
    # 通用配置的修改放在该位置，否则对配置的修改不生效
    try:
        # 该函数的第二个参数可以传入版权声明文件相对于conf.py的相对路径，自动修改版权声明的年份
        # 比如：parsejson.CheckCurrentYearAndModify(app,'../source/copyright/cnperf_conpyright.rst')
        # 默认路径为“./copyright/copyright.rst”和“./copyright/copyright_zh.rst”或者“./copyright/copyright_en.rst”
        # 以上三个路径无需传入
        parsejson.CheckCurrentYearAndModify(app)
    except Exception as e:
        print('------------------')
        print(e)
        # traceback.print_stack()
        traceback.print_exc()
        print('------------------')

    try:
        # print(sphinx.__version__)
        # app.require_sphinx('3.5')
        keys = config.latex_elements.keys()
        if 'sphinxsetup' not in keys:
            config.latex_elements['sphinxsetup'] = ''
        if "3.5" <= sphinx.__display_version__[:3]:
            config.latex_elements['sphinxsetup'] += 'verbatimforcewraps=true,verbatimmaxunderfull=13,'
        #if "3.5" <= sphinx.__display_version__[:3] and \
        #        "7.2" > sphinx.__display_version__[:3]:
        #    config.latex_elements['sphinxsetup'] += 'verbatimmaxunderfull=2,'
        if "5.3" <= sphinx.__display_version__[:3]:
            config.latex_table_style = ['standard', 'nocolorrows']
            config.latex_elements['sphinxsetup'] += 'pre_border-radius=0pt,'
        if len(breathe_version) > 0 and "4.3" <= breathe_version:
            config.latex_elements['preamble'] += '''
                       \\renewenvironment{description}
                         {\\list{}{\\labelwidth=0pt
                          \\let\\makelabel\\descriptionlabel}}
                          {\\endlist}
           '''
        #如果有自定义maketitle，则修改maketitle的内容
        if 'maketitle' in keys:
            __ModifyMakeTitle(keys,config)

        docview_modifyconfig(config)
        # sphinxsetup = config.latex_elements['sphinxsetup']
        # print('sphinxversion:%s;sphinxsetup:%s' % (sphinx.__version__,sphinxsetup))
    except Exception as e:
        # print('sphinxversion:%s %s' % (sphinx.__version__,sphinx.__display_version__[:3]))
        pass


def build_finished_handler(app, exception):
    if exception != None:
        # print(exception)
        return

    # 判断告警文件是否存在，只有无告警或者告警全是忽略告警才允许继续后续的编译
    if warnfilepath != '' and osp.exists(warnfilepath):
        # 判断告警文件中是否全是忽略告警
        iswarn = parsejson.warn_main(warnfilepath, app)
        if iswarn:
            # 如果为True则说明有不可忽略的告警，报sphinxerror异常，停止继续编译
            raise sperror.SphinxError('There are alarms, please check the file of %s for details' % warnfile)
            return

    try:
        if app.builder.name == "latex":
            selffnlst = app.config.latex_documents
            parsejson.Modifylatex_main(app.outdir, selffnlst, app)
    except Exception as e:
        print('------------------')
        print(e)
        traceback.print_exc()

    # 检查html标题一致性。该接口最好放在该位置，主索引文件的标题已经解析出来。
    # if app.builder.name == "html":
    #    result = parsejson.CheckProjectNameIsConsistent(app)
    #    if result != "":
    #        raise sperror.SphinxError(result) #带raise的异常如果用except捕捉，则在终端打印告警将不能显示为红色字体。


def build_inited_handler(app):
    global warnfile
    global warnfilepath

    print(sys.argv)
    args = sys.argv[1:]  # 0为sphinx-build，需忽略掉
    if '-w' in args:
        pos = args.index('-w')  # 找到-w所在的索引位置
        warnfile = args[pos + 1]  # 得到告警保存的文件名
        # print('warnfile=' + warnfile)

        # 根据工作路径，得到文件名的绝对路径
        # 当前在build阶段，因此工作路径为Makefile所在的目录，-w后面的文件保存在基于Makefile的相对路径下
        filepath = osp.join(os.getcwd(), warnfile)
        warnfilepath = osp.abspath(filepath)
        # print('warnfilepath = ' + warnfilepath)

    # try:
    #    # 检查是否有rst_prolog或者rst_epilog替换内容，有的话去掉前后空格
    #    # 仅适用于pdf文件和中文文档，英文文档和html不启作用
    #    checkDocobj = parsejson.clsCheckDocCopyrightYear(app, "")
    #    if app.builder.name == 'latex' or app.builder.name == 'latexpdf':
    #        checkDocobj.CheckReplaceContent()
    # except Exception as e:
    #    print('------------------')
    #    print(e)
    #    traceback.print_exc()
    #    print('------------------')

    # try:
    #    # 检查html配置是否符合规范。必须放在这个位置，否则app没有builder对象。
    #    if app.builder.name == "html":
    #        error = parsejson.CheckHtmlConfigIsCorrect(app)
    #        if error != "":
    #            raise sperror.ConfigError(error)
    # except Exception as e:
    #    print('------------------')
    #    print(e)
    #    traceback.print_stack()
    #    traceback.print_exc()
    #    print('------------------')


def source_read_handler(app, docname, source):
    if cusdirect is not None:
        cnonlyobj = cusdirect.CNOnlyPro(app, docname, source[0])
        source[0] = cnonlyobj.parsecnonlycontent(source[0])

    # try:
    # 自动添加更新历史代码，默认添加“无内容更新”
    #    source[0] = parsejson.CheckUpdateHistory(app, docname, source[0])
    # except Exception as e:
    #    print('------------------')
    #    print(e)
    #    traceback.print_exc()
    #    print('------------------')

    return source


def doctree_resolved_handle(app, doctree, docname):
    """
    删除不需要文件的内容，避免编译html的时候还进行编译，导致内容泄漏
    """
    if not hasattr(app.config, 'enable_exclude_html') or not app.config.enable_exclude_html:
        # 默认按sphinx方式编译所有rst文件到html
        return

    indexname = app.config.master_doc
    if docname == indexname:
        return
    indexdocnames = app.env.toctree_includes[indexname]
    isexist = False
    # 判断文档是否在索引文件里
    if docname not in indexdocnames:
        newtoctree = app.env.toctree_includes.copy()
        del newtoctree[indexname]  # 删除主索引文件，因为无需比较
        for key in newtoctree.keys():
            values = newtoctree[key]
            if docname in values:
                isexist = True
                # 判断该key是否在主索引文件里面
                if key not in indexdocnames:
                    # 如果不在整个的索引文件里面，删除该文件内容
                    # doctree.attributes['source']=''
                    while len(doctree.children) > 0:
                        doctree.remove(doctree.children[0])
                    # 删除标题，否则html页面还有标题，还需要维护标题
                    for node in app.env.titles[docname].children:
                        app.env.titles[docname].remove(node)
        if not isexist:
            # 文档不在toctree字典里，直接删除内容
            while len(doctree.children) > 0:
                doctree.remove(doctree.children[0])
            # 删除标题，否则html页面还有标题，还需要维护标题
            for node in app.env.titles[docname].children:
                app.env.titles[docname].remove(node)


def setup(app):
    """
    该函数中的路径都是相对于工程builder环境的路径而不是相对于conf.py所在的路径，
    该函数外面的相对路径是相对于conf.py的路径没有问题。
    比如./_ext相对路径是相对于工程的路径，因此如果使用abspath得到绝对路径会得到~/m0_docs/_ext/,
    而不是~ / mo_docs / source / _ext。
    因此该路径下对路径的判断最好转化为绝对路径判断，否则可能会判断失败。
    :param app: 
    :return: 
    """
    #将通用配置放在这个位置，因为有些配置比如html_theme，在config_init事件前就已读取，不允许再修改
    #放在setup函数中，提前预配置项才能生效。
    #注意：以下配置因为放在setup函数中调用，因此该配置在conf.py配置之前就会被调用。
    #因此如果conf.py中又做了相同字段的配置，则以conf.py中的配置为准，以下配置将会被覆盖。
    #在这个位置只能设置配置的值，不能读取到配置的值，因为该位置具体配置值还不存在
    modifycommonconfig(app.config)
    
    #事件回调函数
    app.connect('config-inited', config_inited_handler)
    app.connect('build-finished', build_finished_handler)
    app.connect('builder-inited', build_inited_handler)
    app.connect('source-read', source_read_handler)
    app.connect('doctree-resolved', doctree_resolved_handle)

    app.add_config_value('chapterbkpaper_image', '', 'env', [str])
    # 是否排除未添加到index的html文件。在编译html时，sphinx默认会将所有rst文件都生成html，因此可能会造成部分内容的泄露。
    # 为了删除这部分内容，设置该变量。如果置为True的话，将删除不在index.rst里面的html的内容。
    # 默认为false，不做处理，以增加编译效率。
    # 建议还是将不参与编译的rst文件，放到exclude_patterns变量内，放到该变量后，rst不参与编译，能提高编译效率。
    app.add_config_value('enable_exclude_html', False, 'env', [bool])
    #配置版权声明文件完整路径，实现版权声明文件年份的自动修改。
    #parsejson.CheckCurrentYearAndModify如果该函数中传入了版权声明的全路径，则以该函数中传入的为准
    #在该处又增加了一个配置值，是为了配置和代码解偶，为了兼容性，保留了原来的配置方式
    app.add_config_value('copyfile_path', '', 'env', [str])
    #在conf.py中配置的conf.json路径，为了实现conf.json配置的灵活处理
    app.add_config_value('confjson_path', '', 'env', [str])

    if extpathisexist:
        app.setup_extension('customdirective')
