#!/usr/bin/python
# -*- coding: UTF-8
'''
注意：
修改该文件，print打印信息中不能包含中文，否则在不支持中文的操作系统中会出现如下的错误，
导致编译失败或者达不到预期的显示效果。
'ascii' codec can't encode characters in position
'''

from __future__ import print_function

import os
import sys
import json
import re
import codecs
import shutil
import platform
import traceback
import datetime as dt
from subprocess import Popen, PIPE, STDOUT
#import importlib

try:
    import sphinx
    sphinx_version = sphinx.__display_version__[0:3]
except Exception as e:
    sphinx_version = ""

Py_version = sys.version_info
Py_v_info = str(Py_version.major) + '.' + str(Py_version.minor) + '.' + str(Py_version.micro)

#if Py_version >= (3,5,0) and Py_version <(3,7,0):
#    print(False)
#
#if Py_version >= (3,7,0):
#    print(True)
#
#if platform.system().lower() == 'windows':
#    print("windows")
#elif platform.system().lower() == 'linux':
#    print("linux")

filepathlist=os.path.split(os.path.realpath(__file__))
currfilepath=filepathlist[0]
latexfontsize={'tiny','scriptsize','footnotesize','small','normalsize','large','Large','LARGE','huge','Huge'}

def __dictIsHasKey__(dict,key):
    if Py_version >= (3,0,0):
        return dict.__contains__(key)
    else:
        return dict.has_key(key)

def __openconfjsonfile__(filepath = ''):
    jsonfile = filepath
    if filepath == '':
        jsonfile = currfilepath+'/conf.json'
        
    with codecs.open(jsonfile,"r+",encoding = 'utf-8') as load_f:
        load_dict = json.load(load_f)
        return load_dict

def getignorewarning():
    return __load_dict__["ignorewarnkey"]

def GetOtherIncludePackage():
#    packagearr = __load_dict__['package']
#    for package in packagearr:
#        print(package)
    return __load_dict__['package']

def GetReplacePackage():
    return __load_dict__['replacepackage']

def GetCustomOptions():
    return __load_dict__['customoptions']

def GetIsTocContents():
    return __load_dict__['isfiguretabletoc']

def GetSensitiveword():
    #得到敏感词数组，便于搜索文档中是否有敏感词存在
    return __load_dict__['sensitivewords']

def GetTablesContent():
    return __load_dict__['tables']

def GetTablerowtype():
#    packagearr = __load_dict__['tables']['rowtype']
#    print(packagearr)
    return __load_dict__['tables']['rowtype']

def GetTableheadtype():
#    packagearr = __load_dict__['tables']['headtype']
#    print(packagearr)
    return __load_dict__['tables']['headtype']

def GetTableHeadFontColor():
    return __load_dict__['tables']['headfontcolor']

def GetTableStylesArr():
#    packagearr = __load_dict__['tables']['styles']
#    for package in packagearr:
#        print(package)
    return __load_dict__['tables']['styles']

def GetImageStyleArr():
#    packagearr = __load_dict__['image']['styles']
#    for package in packagearr:
#        print(package)
    return __load_dict__['image']['styles']

#判断是否有忽略告警的类
class clsIgnoreWarn:

    def __init__(self,warnfile):
        self.warnfile = warnfile  #保存告警文件

    #判断该关键字是否在告警中，在告警中则返回true，不在告警中则返回fasle
    def __JudgeWarnKey(self,keystr,warnstr):

        #先判断是否为组合关键字
        if "&&" in keystr:
            #解析组合关键字为字符串列表
            keyls = keystr.split("&&")
            isignore = True  #默认该告警为忽略，如果组合关键字其中一个关键字没匹配上，则该告警不能忽略。
            for i in range(0,len(keyls)):
                if keyls[i].strip().lower() not in warnstr.lower():
                    isignore =False
                    break
            return isignore
        else:
            if keystr.lower() in warnstr.lower():
                return True  #忽略告警在告警字符串里面，则返回true，表示该告警可以忽略
            else:
                return False

    #解析告警文件
    def parsewarnfile(self,ignorekey):
        '''
        #make latex命令产生的告警都会保存在stderr文件中，只要判断该文件中的告警是否在忽略告警里就可以了，不在忽略告警里，则肯定有错误，停止执行
        '''
        if not os.path.exists(self.warnfile):
            return True

        fs = codecs.open(self.warnfile,"r",encoding = 'utf-8')
        fstr = fs.read()   #先将所有内容读取到字符串中，方便正则表达式查找
        fs.close()

        #查找带warning的内容
        pattern = re.compile(r"([\s\S].*)[WARNING:|ERROR:]([\s\S].*)", re.I | re.U | re.M)
        mobjarr = pattern.finditer(fstr)

        errstr = '' #保存不能被忽略的告警
        isNotError = True
        for mobj in mobjarr:
            amarr = mobj.group()

            if amarr == '':
                    continue

            amarr = amarr.strip()
            amarr = amarr.strip('\r')
            amarr = amarr.strip('\n')
            amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现

            #判断该告警是否在忽略列表里，不在忽略列表里，保存在errstr字符串中

            if ("WARNING:" not in amarr) and ("ERROR:" not in amarr): #直接忽略
                continue

            isWrite = False  #默认不能忽略
            for igkey in ignorekey:
               isWrite = self.__JudgeWarnKey(igkey,amarr)
               if isWrite:
                    break
                     
            if isWrite is False:
                 #写入stderr文件
                 isNotError = False
                 #fe.writelines(amarr+'\n')
                 errstr += amarr

        if errstr != '':
            #如果有不能被忽略的告警，则将不能忽略的告警重新写入warnfile文件中
            #先删除源文件，避免原文件无法写入
            fw = codecs.open(self.warnfile, "w",encoding = 'utf-8')
            fw.write(errstr)
            fw.close()
        else:
            #如果所有的告警都能忽略则删除之前的告警文件
            #该功能暂时不实现，还是保存原始的告警文件，方便查找。
            #os.remove(self.warnfile)
            pass

        return isNotError

    #该函数暂时未用
    def __parsepdfarg(stdout,stderr,ignorelist,ignorekey):
        '''
        make all-pdf命令产生的所有输出都只会保存在stdout中，因此从stdout中判断是否有告警内容
        '''
        stdoutfilepath = currfilepath+'/'+stdout
        stderrfilepath = currfilepath+'/'+stderr

        if not os.path.exists(stdoutfilepath):
            return True

        fs = codecs.open(stdoutfilepath, "r+",encoding = 'utf-8')
        fstr = fs.read()   #先将所有内容读取到字符串中，方便正则表达式查找
        fs.close
        #查找latexmk的位置，latexmk位置的开始即make all-pdf打印输出的开始
        searchstr = r"latexmk"
        m = re.search(searchstr, fstr, re.I|re.U )
        if m == None:
            return True

        spos = m.span()  #获取位置
        latexcont = fstr[spos[0]:len(fstr)]  #获取到make all-pdf产生的内容

        #查找带warning的内容
        pattern = re.compile(r'([\s\S].*)Warning([\s\S].*)',  re.I|re.U)
        mobjarr = pattern.finditer(latexcont)

        #打开stderr文件，方便后续写
        fe = codecs.open(stderrfilepath, "a+",encoding = 'utf-8')
        isNotError = True
        for mobj in mobjarr:
            amarr = mobj.group()
            if amarr == '':
                    continue
            amarr = amarr.strip()
            amarr = amarr.strip('\r')
            amarr = amarr.strip('\n')
            amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现

            #判断该告警是否在忽略列表里，不在忽略列表里，要写入stderr文件
            if amarr.lower() not in [elem.lower() for elem in ignorelist]: #如果错误不在忽略告警里面
                #如果不能整行匹配，再判断该行是否包含需忽略的告警的关键字
                isWrite = False  #默认不能忽略
                for igkey in ignorekey:

                    isWrite = self.__JudgeWarnKey(igkey,amarr)
                    if isWrite:
                        break;

                if isWrite == False:
                    #写入stderr文件
                    isNotError = False
                    fe.writelines(amarr)
        fe.close

        return isNotError

def warn_main(warnfile,app=None):
    
    global __load_dict__

    if len(__load_dict__) ==0:
        GetConfjsondict(app)

    ignorekey = getignorewarning()
    clswarn = clsIgnoreWarn(warnfile)

    if not clswarn.parsewarnfile(ignorekey):
        #如果存在不可忽略的告警，则返回True，否则返回False
        return True

    return False


class clsTableattr:

    def __init__(self, tables):
        self.rowtype = tables['rowtype']
        if __dictIsHasKey__(tables,'headtype'):
            self.headtype = tables['headtype']
        else:
            self.headtype = ""
        if __dictIsHasKey__(tables,'headfontcolor'):
            self.headfontcolor = tables['headfontcolor']
        else:
            self.headfontcolor = ""
        if __dictIsHasKey__(tables,'styles'):
            self.tablestyles = tables['styles']
        else:
            self.tablestyles = None
        if __dictIsHasKey__(tables,'isname'):
            self.isname = tables['isname']
        else:
            self.isname = False
        if __dictIsHasKey__(tables,'fontsize'):
            self.fontsize = tables['fontsize']
        else:
            self.fontsize = ""
            
        if __dictIsHasKey__(tables,'ismulticolleftalign'):
            self.ismulticolleftalign = tables['ismulticolleftalign']
        else:
            self.ismulticolleftalign = False
        
        if __dictIsHasKey__(tables,'ismultirowautowrap'):
            self.ismultirowautowrap = tables['ismultirowautowrap']
        else:
            self.ismultirowautowrap = True
            
        if __dictIsHasKey__(tables,'ismultirowatupcell'):
            self.ismultirowatupcell = tables['ismultirowatupcell']
        else:
            self.ismultirowatupcell = False

        if __dictIsHasKey__(tables,'ismultirowatlowcell'):
            self.ismultirowatlowcell = tables['ismultirowatlowcell']
        else:
            self.ismultirowatlowcell = False

class clsModifyTex:

    def __init__(self, content,app=None):
        self.content = content
        self.app = app
        self.tablesattrobj = clsTableattr(GetTablesContent())
        
        #构建模板字符串
        #自动换行模板
        self.autolinetemplat =r'''\renewcommand{\arraystretch}{0.8}
        \begin{tabular}[c]{@{}l@{}}%(autocontent)s\end{tabular}'''
        
        fontcolor = self.tablesattrobj.headfontcolor
        fontcolor = fontcolor.replace('{}', '{%(content)s}', 1)
        self.commonstr = r'''%(sphinxstyletheadfamily)s\cellcolor''' + \
                         self.tablesattrobj.headtype + "\n{" + fontcolor + "}"

        # multirow模板
        self.multirowstr = r'''\multirow{-%(count)s}{%(coluwidth)s}%(raggedright)s{
        ''' + self.commonstr + "}"

        self.simplemultirow = r'''\multirow{%(count)s}{%(coluwidth)s}{\cellcolor''' + \
                              self.tablesattrobj.headtype + "}"

        # multicolumn模板
        self.multicolumnstr = r'''\multicolumn{%(count)s}{%(flag)s}{%(prefix)s''' + \
                              self.commonstr + r'''%(suffix)s'''

        # 为了普通标题文本看起来和使用了multicolum你的位置一致，对于非multirow和multicolumn单元格的内容，都用multicolumn{1}{|l|}标志
        self.normalstr = r'''\multicolumn{1}{%(flag)s}{
                \begin{varwidth}[c]{%(colwidth)s}
                 ''' + self.commonstr + r'''
\par
\vskip-\baselineskip\vbox{\hbox{\strut}}\end{varwidth}}'''
        self.comcolstr = r'''\multicolumn{%(count)s}{%(flag)s}{''' + self.commonstr +"}"
        # 空的单元格内容,multirow位置变换后，会有空的单元格
        self.emptystr = r"\cellcolor" + self.tablesattrobj.headtype+"{}"

        #删除字符串列表。有时候sphinx生成的文本内容里会带有特殊的latex指令，该指令不支持设置字体颜色，需要删除。删除不影响内容显示。
        self.delstrlst=[r'\begin{quote}',r'\end{quote}']

    #加入其它包
    def AddPackageToTex(self):
        #得到需要包的数组
        packarr = GetOtherIncludePackage()
        if len(packarr)==0:
            return  False

        #如果数组有内容，就需要将包添加到latex文件的导言区
        #搜索\usepackage{sphinx}，将包加在它的前面，用正则表达式搜索它的位置
        #采用正则表达式替换的方式，替换搜索到的字符串，因此需要先构建字符串
        #python认为\u后面为unicode字符，因此需要多加一个转义字符\，python才认为是整个的字符串
        #searchstr = r'\\usepackage\[dontkeepoldnames\]{sphinx}'
        searchstr = r'\\usepackage(\[\S*\]*)?{sphinx}'
        matchstr = re.search(searchstr,self.content)

        replacestr=""
        for package in packarr:
            replacestr  += package+'\n'

        if Py_version >= (3,7,0):
            replacestr += "\\" + matchstr.group(0)
        else:
            replacestr += matchstr.group(0)


        self.content = re.sub(searchstr, replacestr, self.content, 1, re.M | re.I|re.U)

        return True

    #加入自定义选项,包放在了sphinx包的前面，因此选项放在sphinx包的后面
    def AddCustormOptionsToTex(self):
        #得到需要包的数组
        packarr = GetCustomOptions()
        if len(packarr)==0:
            return  False

        #如果数组有内容，就需要将包添加到latex文件的导言区
        #搜索\usepackage{sphinx}，将自定义参数放在它的后面，用正则表达式搜索它的位置
        #采用正则表达式替换的方式，替换搜索到的字符串，因此需要先构建字符串
        #python认为\u后面为unicode字符，因此需要多加一个转义字符\，python才认为是整个的字符串
        searchstr = r'\\usepackage(\[\S*\]*)?{sphinx}'
        matchstr = re.search(searchstr,self.content)

        replacestr=""
        for package in packarr:
             replacestr  += package+'\n'

        if Py_version >= (3,7,0):
            replacestr = "\\" + matchstr.group(0)+'\n'+replacestr
        else:
            replacestr = matchstr.group(0)+'\n'+replacestr

        self.content = re.sub(searchstr, replacestr, self.content, 1, re.M | re.I|re.U)
        return True

    #增加figure和table toc到tex
    def AddOtherTocToTex(self):
        #得到需要包的数组
        packarr = GetIsTocContents()
        if len(packarr)==0:
            return  False

        replacestr = ""
        if packarr['isfigurestoc']:
           figlst = packarr['figurestoc']
           for figstr in figlst:
               replacestr += figstr + '\n'

        if packarr['istablestoc']:
           figlst = packarr['tablestoc']
           for figstr in figlst:
               replacestr += figstr + '\n'
        if replacestr == "":
           return

        #如果数组有内容，就需要将包添加到latex文件的导言区
        #搜索\usepackage{sphinx}，将包加在它的前面，用正则表达式搜索它的位置
        #采用正则表达式替换的方式，替换搜索到的字符串，因此需要先构建字符串
        #python认为\u后面为unicode字符，因此需要多加一个转义字符\，python才认为是整个的字符串
        searchstr = r'\\sphinxtableofcontents'
        matchstr = re.search(searchstr,self.content)

        if Py_version >= (3,7,0):
            replacestr = "\\" + matchstr.group(0) + '\n' + replacestr
        else:
            replacestr = matchstr.group(0) + '\n' + replacestr

        self.content = re.sub(searchstr, replacestr, self.content, 1, re.M | re.I|re.U)
        return True

    def __IsRequireReplace(self, value):
        """
        根据sphinx版本判断是否需要替换,有两种写法：
        1. {sphinx:5.3.0}:这种写法是当前sphinx版本大于等于5.3.0版本时，才执行替换。
        2. {5.3.0:sphinx}:这种写法是当前sphinx版本小于5.3.0时，才进行替换
        :return: 返回True需要替换，和需要替换的实际value;或者返回fasle，不需要替换
        """
        #根据正则表达式得到需要替换的sphinx的最小版本号
        if len(sphinx_version)==0:
            return True,value
        #按大于等于版本号进行匹配查找
        searchstr = r"\{sphinx:([0-9\.]+)\}([\s\S]+)"
        match = re.match(searchstr,value,re.I | re.U)
        if match is not None:
            #print(match.group(1),match.group(2))
            try:
                require_version = match.group(1)[:3]
                if sphinx_version >= require_version:
                    return True, match.group(2)
                else:
                    return False, ""
            except Exception as e:
                print(e)
                return True,value
            
        else:
            #按小于版本号进行匹配查找
            searchstr = r"\{([0-9\.]+):sphinx\}([\s\S]+)"
            match = re.match(searchstr, value, re.I | re.U)
            if match is not None:
                try:
                    require_version = match.group(1)[:3]
                    if sphinx_version < require_version:
                        return True, match.group(2)
                    else:
                        return False, ""
                except Exception as e:
                    print(e)
                    return True, value
            else:
                return True,value
        
    
    def ModifyReplacePackage(self):
        #得到字典值
        #得到需要替换的包，用正则表达式替换
        redict = GetReplacePackage()
        if len(redict) ==0:
           return

        #返回字典中所有键值的列表
        keylst = list(redict)
        for key in keylst:
            if key == 'comment' :
                continue
            keyvalue = redict[key]   #得到键对应的值
            result,keyvalue = self.__IsRequireReplace(keyvalue)
            if result :
                #对键值进行替换
                self.content = re.sub(key, keyvalue, self.content, 0, re.I|re.U)
        return
    
    def __GetReplacedContent(self,afterstr,replacelst):
        """
        对需要替换的内容进行替换
        :param afterstr: 需要替换的字符串
        :return: 
        """
        if len(afterstr) ==0 or len(replacelst)==0:
            return afterstr
        for replacelist in replacelst:
            if len(replacelist) !=2:
                continue
            replacedstr = replacelist[0]
            replacestr = replacelist[1]
            if len(replacedstr) ==0 or len(replacestr)==0:
                continue
            afterstr = re.sub(replacedstr, replacestr, afterstr, 0, re.U | re.M)
            
        return afterstr
    
    def __ParseStyleForSpecialFunction(self,isshaded,apidict,linecontent):
        
        if isshaded:
            envstartstr = "\n\\begin{shaded}\n"  # 环境起始字符串
            envendstr = "\n\\end{shaded}\n"  # 环境结束字符串
        else:
            envstartstr = ""  # 环境起始字符串
            envendstr = ""  # 环境结束字符串
            
        afterstr = linecontent
        
        if apidict is None or len(apidict)==0:
            return envstartstr,envendstr,afterstr
        
        if not __dictIsHasKey__(apidict, "isenter") or \
                (__dictIsHasKey__(apidict, "isenter") and \
                 apidict['isenter'] is True):
            afterstr = self.__strreplacepos(afterstr,r"pysiglinewithargsret",r",",r",\\")
        if __dictIsHasKey__(apidict, "istcolorbox") and \
                apidict['istcolorbox'] is True:
            envstartstr, envendstr = self.__GettcolorboxForFunction(apidict)
        elif __dictIsHasKey__(apidict, "fontsize") and \
                apidict['fontsize'] in latexfontsize:
            envstartstr += "\\" + apidict['fontsize'] + "\n"
        
        #检查是否有替换，有替换的话，对内容进行替换
        if __dictIsHasKey__(apidict, "replace"):
            replacelst = apidict['replace']
            if len(replacelst) > 0:
                afterstr = self.__GetReplacedContent(afterstr,replacelst)
        
        return envstartstr,envendstr,afterstr
        
    def __ModifyFunctionBkColorByPython(self,isshaded,apilst,newcontent,linecontent,curpos,prepos):
        #根据python生成的格式为类和函数声明添加灰底，并按逗号换行

        '''
        根据正则表达式获得以\pysiglinewithargsret开头，以“{}”结尾的中间字符串
        对中间字符串添加灰底和换行
        '''
        if isshaded:
            envstartstr = "\n\\begin{shaded}\n"  # 环境起始字符串
            envendstr = "\n\\end{shaded}\n"  # 环境结束字符串
        else:
            envstartstr = ""  # 环境起始字符串
            envendstr = ""  # 环境结束字符串
            
        startmultiline = '\\pysigstartmultiline\n'
        stopmultiline = '\n\\pysigstopmultiline'
        #searchstr = r'(?=\\pysiglinewithargsret).*(?<={})'
        searchstr=r'(?=\\pysiglinewithargsret).*'
        match = re.search(searchstr, linecontent, re.I|re.U)
        if match is not None:
           #重新组合linecontent
            #pos = match.span()
            isapi, apidict = self.__IsSpecialFunction(apilst, linecontent)
            if  isapi is True and apidict is not None:
                envstartstr,envendstr,afterstr = self.__ParseStyleForSpecialFunction(isshaded,apidict,match.group())
                newstr = '\n' + linecontent[:match.start()] + envstartstr + startmultiline + afterstr+ stopmultiline + envendstr+ linecontent[match.end():len(linecontent)]
            else:
                newstr = '\n' + linecontent[:match.start()] + envstartstr+ startmultiline + match.group().replace(r",",r",\\") + stopmultiline + envendstr + linecontent[match.end():len(linecontent)]
            #计算替换前的内容
            if len(prepos)==0:
                newcontent = self.content[:curpos[0]-1] + newstr
            else:
                newcontent += self.content[prepos[1]:curpos[0]-1] + newstr

        return newcontent

    def __strreplacepos(self,srcstr,posstr,oldstr,newstr):

        # 从指定字符串开始对行内字符串进行替换
        '''
        srcstr：原始字符串
        posstr：从该字符串开始进行替换
        oldstr：被替换字符串
        newstr：替换字符串
        '''
        #查找字符串的起始位置
        pos = srcstr.find(posstr)
        if pos == -1:
            return ""  #如果查找的字符串没有找到，则返回空。

        #根据找到的位置进行字符串拆分
        startstr = srcstr[0:pos]
        beforestr = srcstr[pos:len(srcstr)]
        #对字符串进行替换
        afterstr = beforestr.replace(oldstr,newstr)
        return startstr+afterstr
    
    def __GetleftrightValueForSpecialapi(self,str):
        """
        得到conf.json specialapi配置中left和right的值
        字符串格式为“[-][数字][单位]”，根据这个格式查找值和单位
        :param str: 
        :return: 
        """
        if len(str) ==0:
            return 0,""

        searchstr = r'[-]{0,1}([0-9]{1,2})([A-Za-z]{1,2})'
        match = re.search(searchstr, str, re.I)
        if match is not None:
            value = int(match.group(1))
            unit = match.group(2)
            return value,unit
        else:
            return 0,""
        
    def __GettcolorboxForFunction(self,apidict):
        #判断有没有设置页面左右边距,的左右边距
        #必须通过\geometry指令设置，否则按默认处理
        
        leftvalue = 0
        leftunit = "mm"
        left = "0mm"
        
        rightvalue = 0
        rightunit = "mm"
        right="0mm"

        isleftset = False
        isrightset = False
        
        if __dictIsHasKey__(apidict, "left") and \
            len(apidict['left']) > 0:
            leftvalue,leftunit=self.__GetleftrightValueForSpecialapi(apidict['left'])
            if len(leftunit) > 0:
                left = str(leftvalue)+leftunit
                isleftset = True
                
        if isleftset is not True:
            leftvalue = 9 #默认值
            leftunit="mm"
            left = str(leftvalue)+leftunit 

            searchstr = r'\\geometry{[\s\S]*left=([0-9]{1,2})([A-Za-z]{1,2})[\s\S]*}'
            match = re.search(searchstr, self.content, re.I | re.M)
            if match is not None:
                left = match.group(1) + match.group(2)
                leftvalue = int(match.group(1))
                leftunit = match.group(2)
                
            #print(left)
                
        if __dictIsHasKey__(apidict,"right") and \
            len(apidict['right']) >0 :
            rightvalue, rightunit = self.__GetleftrightValueForSpecialapi(apidict['right'])
            if len(rightunit) > 0:
                right = str(rightvalue) + rightunit
                isrightset = True
                
        if isrightset is not True:
            rightvalue = leftvalue - 1  # 为了使显示效果更好看
            right = str(rightvalue)+leftunit
            
            searchstr = r'\\geometry{[\s\S]*right=([0-9]{1,2})([A-Za-z]{1,2})[\s\S]*}'
            match =re.search(searchstr,self.content,re.I|re.M)
            if match is not None:
                rightvalue = int(match.group(1)) -2
                rightunit = match.group(2)
                right = str(rightvalue) + rightunit
            #print(right)
            
        fontsize = ""
        if __dictIsHasKey__(apidict, "fontsize") and \
                apidict['fontsize'] in latexfontsize:
            fontsize = "fontupper=\\" + apidict['fontsize']
            
        envstartstr = "\n\\begin{tcolorbox}[arc=0mm,boxrule=-1mm,left skip=-" \
                       + str(leftvalue+1) + leftunit +",left="+left +",right=-" +right \
                       + ",colback=shadecolor," + fontsize +"]\n"
        envendstr = "\n\\end{tcolorbox}\n"
        #print("=====================")
        #print(envstartstr)
        #print(envendstr)
        #print("=====================")
        return envstartstr,envendstr

    
    def __IsSpecialFunction(self,apilst,linecontent):
        '''
        判断是否为特殊api
        :param apilst: 
        :param linecontent: 
        :return: 
        '''
        if len(apilst)==0:
            return False,None

        for apidict in apilst:
            if __dictIsHasKey__(apidict,"latexapiname"):
                apistr=apidict['latexapiname'] #有可能以逗号分割的多个函数名称
                apiarr=apistr.split(",")   #以逗号分割函数名称列表
                for api in apiarr:
                    if api in linecontent:
                        return True,apidict
        return False,None
    
    def __RemoveLastSpace(self,headstr):
        """
        删除最后的空格，避免函数含义和关键字间有大的空行
        :param headstr: 
        :return: 
        """
        headlist = headstr.split('\n')
        index = 0
        for i in range(len(headlist)-1,0,-1):
            str = headlist[i].strip('\r')
            if len(str) == 0:
                index = i
                break
        if index > 0:
            headlist.pop(index)
            return '\n'.join(headlist)
        else:
            return headstr
        
    def __ModifyParamPos(self,paramcontent):
        """
        调整参数位置在最前面，为了适配breathe 4.24.1之后版本
        :param paramcontent: 
        :return: 
        """
        #查找第一个\begin{description}\n\sphinxlineitem{}字符串
        searchstr = r"(\\begin\{quote\})?(\r\n|\n)?\\begin\{description\}(\r\n|\n)?\\sphinxlineitem\{(\sphinxstylestrong\{)?(?P<paramkey>[\s\S].+)?(\})?\}(\r\n|\n)?[\s\S]+?(\\end\{description\}){1}(\r\n|\n)?(\\end\{quote\})?"
        match = re.search(searchstr,paramcontent,re.I|re.M|re.U)
        if match is not None:
            
            if "\\begin{quote}" in match.group():
                #删掉缩进
                otherstr = match.group()[len('\\begin{quote}'):len(match.group())-len('\\end{quote}')]
            else:
                otherstr = match.group()
                
            paramkey = match.group('paramkey')
            headstr = paramcontent[0:match.start()]
            headstr = self.__RemoveLastSpace(headstr)
            if "Parameters" in paramkey:
                #说明参数本身在第一个位置，不做处理
                return headstr + '\n' + otherstr + paramcontent[match.end():len(paramcontent)]
            
            #如果参数不在第一个位置，查找参数的位置并将第一个位置记录下来
            searchstr = r"(\\begin\{quote\})?(\r\n|\n)?\\begin\{description\}(\r\n|\n)?\\sphinxlineitem\{(\sphinxstylestrong\{)?(?P<paramkey>Parameters)?(\})?\}(\r\n|\n)?[\s\S]+?(\\end\{description\}){1}(\r\n|\n)?(\\end\{quote\})?"
            matchparam = re.search(searchstr,paramcontent,re.I|re.M|re.U)
            if matchparam is None:
                return headstr + '\n' + otherstr + paramcontent[match.end():len(paramcontent)]

            if "\\begin{quote}" in matchparam.group():
                #删掉缩进
                paramstr = matchparam.group()[len('\\begin{quote}'):len(matchparam.group())-len('\\end{quote}')]
            else:
                paramstr = matchparam.group()
                
            #重新组合字符串,将参数放在第一位
            newparamcontent = headstr + '\n' +paramstr + '\n' +otherstr \
                              + paramcontent[match.end():matchparam.start()] \
                              + paramcontent[matchparam.end():len(paramcontent)]

            return newparamcontent
        else:
            return paramcontent
        
    def ModifyFunctionBackColor(self):
        '''
        该函数用来修改函数的背景色，并根据参数换行
        * c/c++ sphinx生成的函数都被以下三行包围：
          \pysigstartmultiline
          \pysiglinewithargsret{xxxxx}
          \pysigstopmultiline
          因此利用正则表达式查找这三行所在的位置进行替换。
          注意：
          要实现该功能，latex必须包含framed和color包，同时定义以下颜色：
          \definecolor{shadecolor}{RGB}{220,220,220}
          如果shadecolor颜色未定义，则该函数执行失败。
        * python生成的latex文件中函数或者类的声明，有\pysiglinewithargsret指令，但是没有pysigstartmultiline指令。
          因此需要在\pysiglinewithargsret指令句子的结束，前后先添加pysigstartmultiline和pysigstopmultiline。再添加\begin{shaded}和\end{shaded}
          一句的结束根据"{}"来判断，遇到"{}"即为句末。
        '''
        #判断是否有特殊API需要处理
        apilst = []
        isshaded = True
        if __dictIsHasKey__(__load_dict__,"specialapi"):
            if __dictIsHasKey__(__load_dict__["specialapi"],"styles"):
                apilst=list(__load_dict__["specialapi"]["styles"])
            if __dictIsHasKey__(__load_dict__["specialapi"],"isshaded"):
                isshaded = __load_dict__["specialapi"]["isshaded"]
            
        pythontype = False
        newcontent = ""
        prepos=[]     #定义需要替换的字符串列表，该列表中的每一个元素都包含上面提到的三行。
        #searchstr = r'^(?=.*\\pysiglinewithargsret).*$'
        #该正则表达式可以把pysiglinewithargsret或者带template的识别出来。有template的会被pysigline标签标识
        #该修改也兼容sphinx 5.3.0及以后版本。否则5.3.0版本之后将无法使用。
        #searchstr=r'(\\pysigline[\s\S].*){0,1}[\n]{0,1}(\\pysiglinewithargsret[\s\S].*)[\n]'
        searchstr=r'(\\pysigstartsignatures){0,1}[\r\n|\n]{0,1}(\\pysigstartmultiline){0,1}(\r\n|\n){0,1}((\\pysigline[\s\S].*){0,1}[\n]{0,1}(\\pysiglinewithargsret[\s\S].*))[\n](\\pysigstopmultiline){0,1}[\r\n|\n]{0,1}(\\pysigstopsignatures){0,1}'
        m = re.finditer(searchstr, self.content, re.M|re.I|re.U)
        for match in m:

            linecontent = match.group()
            #判断是否添加了pysigstartmultiline和pysigstopmultiline，没有添加的话需要添加才能添加灰色背景
            #multilen=len(r'\pysigstartmultiline')
            #startmultistr = self.content[match.start()-multilen-1:match.start()]
            #如果上一行不是\pysigstartmultiline则需要按python风格修改，添加该标志
            #print(startmultistr.strip())
            if match.group(2) is None:
                #print('is python')
                # 计算替换前的内容
                newcontent = self.\
                    __ModifyFunctionBkColorByPython(isshaded,apilst,newcontent,linecontent,match.span(),prepos)
                prepos = match.span()
                pythontype = True
            else:
                #tablestr = match.groups()
                #计算替换前的内容
                if len(prepos)==0:
                    newcontent = self.content[0:match.start()]
                else:
                    # 得到参数信息内容，方便后面调整位置，此时调整的参数为上一个函数的参数位置
                    paramcontent = self.content[prepos[1]:match.start()]
                    # breathe 4.24.1之后版本将parameters放在了最后面，
                    # 为了适配breathe 4.24.1之后版本对格式的调整
                    # 检查参数是否在后面，是的话，需要将参数放到最前面
                    # breathe 4.24.1之后版本将doxygen的标签都放在\begin{description}\n\sphinxlineitem{}标志对里面
                    # 因此对该标志对进行搜索
                    newparamcontent = self.__ModifyParamPos(paramcontent)
                    newcontent += newparamcontent
                    

                #当有template的时候，pysiglinewithargsret不一定在行首，因此需要从pysiglinewithargsret开始替换逗号，加强制换行符
                if isshaded:
                    envstartstr = "\n\\begin{shaded}\n"  # 环境起始字符串
                    envendstr = "\n\\end{shaded}\n"  # 环境结束字符串
                else:
                    envstartstr = ""  # 环境起始字符串
                    envendstr = ""  # 环境结束字符串
                    
                isapi,apidict = self.__IsSpecialFunction(apilst,linecontent)
                if  isapi is True and apidict is not None:
                    envstartstr,envendstr,afterstr=self.__ParseStyleForSpecialFunction(isshaded,apidict,linecontent)
                    newstr = envstartstr + afterstr + envendstr
                else:
                    afterstr=self.__strreplacepos(linecontent,r"pysiglinewithargsret",r",",r",\\")
                    #得到替换后的字符串
                    newstr = envstartstr + afterstr + envendstr
                #得到替换后的内容
                newcontent += newstr
                prepos = match.span()
                pythontype = False

        if len(prepos) > 0:
            paramcontent = self.content[prepos[1]:len(self.content)]
            newparamcontent = self.__ModifyParamPos(paramcontent)
            self.content = newcontent + newparamcontent
            #self.content = newcontent + self.content[prepos[1]:len(self.content)]
        #print('===============================================')

    def __GetTableCaption(self,tablecontent):
        '''
        得到表格的caption，方便输出打印
        :param tablecontent: 表格内容
        :return: 返回表格的caption
        '''
        tablecaption = ""
        searchstr = r'(\\sphinxcaption|\\caption)\{(?P<caption>[\s\S]*?)\}(?=\\label|\\\\)'
        matchcaption = re.search(searchstr, tablecontent, re.M | re.I | re.U)
        if matchcaption is not None:
            tablecaption = matchcaption.group('caption') #得到caption的值
            #长表格caption后面会添加\struct的后缀，去掉
            tablecaption = tablecaption.replace(r"\strut","")

        return tablecaption
    
    def ModifyTablesAttributes(self):
        #修改表格属性
        newcontent = self.content
        searchstr = r'(\\begin{savenotes})([\s\S]*?)(\\sphinxattablestart|\\sphinxatlongtablestart)([\s\S]*?)(\\sphinxattableend|\\sphinxatlongtableend)([\s\S]*?)(\\end{savenotes})'
        matchobj = re.finditer(searchstr, self.content, re.M|re.I|re.U)
        #outputstr = "\033[34mModifying Table:\033[0m"
        #print(outputstr, end="")
        icount = 0
        for match in matchobj:
            try:
                icount = icount + 1
                tablestr = match.groups()
                tablecontent = tablestr[0]+tablestr[1]+tablestr[2]+tablestr[3]+tablestr[4]
                columnwidthlst,newtablecontent = self.__GetColumnParameters(tablecontent)
                #得到表格caption
                tablecaption=self.__GetTableCaption(tablecontent)
                #print(tablecaption)
                #outputtable="\033[34m[" + tablecaption +"];\033[0m"
                #print(outputtable, end="")
                #为了兼容纯英文linux系统，在打印信息中不能包含中文，因此按自动计数累加表示表格，
                #避免表格标题中有中文时导致修改表格失败。
                prstr = "\033[34m[Modifying table"+str(icount)+":... ...];\033[0m"
                print(prstr, end="\r")
                if self.tablesattrobj.tablestyles is None:
                    #先修改表格的通用属性
                    newtableattr,islongtable = self.__StartModifyTableAttr(newtablecontent,columnwidthlst, False)
                    newcontent = newcontent.replace(tablecontent, newtableattr)
                else:
                    caption_dict = self.__CreateTableCaptionDict(self.tablesattrobj.tablestyles)
                    if len(caption_dict ) > 0 :
                        newtableattr = self.__ModifySingleTableattr(newtablecontent, columnwidthlst, caption_dict ) #tablestr也是3个内容的数组，因为正则表达式被分为了3组，只取中间分组的内容。
                        #重新组成新的字符串
                        newcontent = newcontent.replace(tablecontent, newtableattr)

            except Exception as e:
                print(e)
                traceback.print_exc()

        print("\r\n")
        self.content = newcontent

    #替换表格中字体大小
    def __ModifyTableFontSize(self,singletablecontent,fontsize):
        
        searchstr = r'(\\begin{savenotes})([\s\S]*?)(\\sphinxattablestart|\\sphinxatlongtablestart)'
        m = re.search(searchstr,singletablecontent, re.M|re.I|re.U)
        tablestr = m.groups()
        #替换表格中的字体
        #同时也要使表格的caption字体大小保持一致，修改表格的caption需要添加如下指令：\renewcommand{\tablename}{\scriptsize{表} }
        captionfont = "\n\\renewcommand{\\tablename}{\\" + fontsize + "{Table} }\n"
        if doc_language=='zh_CN':
            captionfont="\n\\renewcommand{\\tablename}{\\" + fontsize + "{表} }\n"
            
        tablefontsize = '\\'+fontsize + captionfont
        
        return tablestr[0]+tablefontsize+tablestr[2]+singletablecontent[m.end():len(singletablecontent)]
    
    def __CreateTableCaptionDict(self, tablestylesarr):
        #根据caption生成表格字典，key=caption，value=属性数组
        cap_dict = {}

        for tablestyle_dict in tablestylesarr:
            captionarr = tablestyle_dict['caption']
            #该caption可能是一个逗号分隔的字符串数组，因此需要做拆分
            captionlist = captionarr.split(",")
            for caption in captionlist:
                cap_dict[caption] = tablestyle_dict  #以caption为key重新生成字典，便于查找
        return cap_dict
    
    #查找表格caption是否被匹配
    def __JudgeIsCaption(self,caption_dict,tablecaption):
        
        captionlst=caption_dict.keys() #得到json配置的所有caption
        for caption in captionlst:
            tablestyle_dict = caption_dict[caption]
            if __dictIsHasKey__(tablestyle_dict, 'ispartial') and tablestyle_dict['ispartial']==True:
                if caption in tablecaption:
                    return True,tablestyle_dict
            else:
                if caption == tablecaption:
                   return True,tablestyle_dict
        return False,None
    
    def  __ModifySingleTableattr(self, singletablecontent, columnwidthlst, caption_dict):
        #修改单个表格属性
        #从单个表格里用正则表达式找caption
        #定义正则表达式,查找caption内容
        tablecaption = ''
        new_singletablecontent = singletablecontent
        if self.tablesattrobj.isname:
            searchstr = r'.*\\label.*?:(?P<caption>[\s\S].*)}}.*'
            matchcaption = re.search(searchstr, singletablecontent, re.M | re.I | re.U)
            if matchcaption is not None:
                tablecaption = matchcaption.group('caption')  # 得到caption的值
            else:
                tablecaption = ''
        else:
            tablecaption = self.__GetTableCaption(singletablecontent)
        
        #stylefontsize = False
        iscaption = False
        iscaption,tablestyle_dict=self.__JudgeIsCaption(caption_dict,tablecaption)
        if iscaption:
            
            #if __dictIsHasKey__(tablestyle_dict,'fontsize') and (tablestyle_dict['fontsize'] in latexfontsize):
            #    stylefontsize = True
            #    tablestyle_dict['isLongTable'] = True  #只有长表格的\caption指令，设置的字体才能生效，所以如果要设置表格字体，需要强制设置为长表格。
            jsonmultirow = []  #保存需要设置自动换行的合并单元格的列号。
            if __dictIsHasKey__(tablestyle_dict, 'multirowcolumn'):
                jsonmultirow = tablestyle_dict['multirowcolumn']
                
            if __dictIsHasKey__(tablestyle_dict, 'isLongTable') is not True:
                tablestyle_dict['isLongTable'] = False
            
            if __dictIsHasKey__(tablestyle_dict, 'isCusHead') is not True:
                tablestyle_dict['isCusHead'] = False
                
            #修改表格通用属性
            new_singletablecontent,islongtable= self.__StartModifyTableAttr(new_singletablecontent,
                                                                columnwidthlst,
                                                                 tablestyle_dict['isLongTable'],
                                                                 tablestyle_dict['isCusHead'],
                                                                 jsonmultirow,
                                                                 tablestyle_dict)
            #设置该表格定制字体
            if __dictIsHasKey__(tablestyle_dict,'fontsize') and (tablestyle_dict['fontsize'] in latexfontsize):
                new_singletablecontent=self.__ModifyTableFontSize(new_singletablecontent,tablestyle_dict['fontsize'])
                
            #渲染竖型表格的第一列
            if __dictIsHasKey__(tablestyle_dict,'isVertical') and tablestyle_dict['isVertical']==True:
                if islongtable:
                    # 长表格自定义sphinxcolwidth的第2个参数默认为5，否则无法实现自动换行
                    #self.normalstr = self.normalstr % {
                    #    'flag': '%(flag)s',
                    #    'colwidth': '5',
                    #    'sphinxstyletheadfamily': '%(sphinxstyletheadfamily)s',
                    #    'content': '%(content)s'
                    #}
                    new_singletablecontent = self.__ModifyVerticalLongTable(new_singletablecontent,tablestyle_dict,columnwidthlst)
                else:
                    # 普通表格自定义sphinxcolwidth的第2个参数默认为3，否则无法实现自动换行
                    #self.normalstr = self.normalstr % {
                    #    'flag': '%(flag)s',
                    #    'colwidth': '3',
                    #    'sphinxstyletheadfamily': '%(sphinxstyletheadfamily)s',
                    #    'content': '%(content)s'
                    #}
                    new_singletablecontent = self.__ModifyTableByLine(new_singletablecontent,True,False,tablestyle_dict,columnwidthlst)
                    
            if __dictIsHasKey__(tablestyle_dict,'isnewpage') and tablestyle_dict['isnewpage']==True:
                new_singletablecontent = '\\newpage\n' + new_singletablecontent
                
            #最后修改表格的行高
            if __dictIsHasKey__(tablestyle_dict, 'ht_rowno') and tablestyle_dict['ht_rowno'] is not None:
                new_singletablecontent = self.__ModifyTableRowHeight(islongtable,new_singletablecontent,tablestyle_dict['ht_rowno'])
        else:
            #修改表格的通用属性
            new_singletablecontent,islongtable = self.__StartModifyTableAttr(singletablecontent, columnwidthlst, False)
        if new_singletablecontent == '':
           new_singletablecontent = singletablecontent

        return new_singletablecontent
    
    def __NormaltableRowHeight(self,rowno,rowheightinfo,singletablecontent):
        #自定义长表格，第一行以“\hline”开头，因此从第一个“\hline”开头进行替换。
        searchstr = r'\\hline'
        match = re.search(searchstr,singletablecontent,re.I | re.U)
        if match is None:
            return singletablecontent
        
        #latex表格属性内容
        tablehead = singletablecontent[0:match.start()]
        #表格内容
        tablecontent = singletablecontent[match.start():len(singletablecontent)]
        tablecontent = self.__ModifyNormalRowHeight(rowno,rowheightinfo,0,tablecontent)
        
        return tablehead + tablecontent

    def __ModifyNormalRowHeight(self, rowno, rowheightinfo, startline, tablecontent):
        # 修改除表格外其它行的行高，并返回修改后的内容
        searchstr = r'\\\\'
        pattern = re.compile(searchstr, re.I | re.U)
        matchiter = pattern.finditer(tablecontent)
        i = startline
        for m in matchiter:
            i += 1
            if rowno == i:
                # 开始设置该行的行高
                # 获取起始和结束位置
                posstart = m.start()
                posend = m.end()
                tablecontent = tablecontent[0:posstart] + rowheightinfo + tablecontent[posend:len(tablecontent)]
                break

        return tablecontent
    
    def __JudgeRowlistRule(self,rowlist):
        #判断行号和行高信息是否符合规则

        if len(rowlist)!=2:
            return False
        rowno = rowlist[0]     #得到行号
        rowhtstr=rowlist[1]    #得到行高信息

        #判断是否为异常信息
        #设置的最大行高不能超过10，因为10的行高太高，正常不需要这么高的行高。而且sphinx会根据换行自动调整行高，该方法只用户行高的微调。
        if rowno <= 0 or len(rowhtstr) == 0 or not rowhtstr.isdigit() or (int(rowhtstr) > 99):
            return False
        return True
    
    def __ModifyLongTableRowHeight(self,rowno,rowheightinfo,singletablecontent):

        # 判断是自定义长表格还是sphinx生成的长表格。
        # sphinx自动生成的长表格有两个表头，第一个表头以“\endfirsthead”结尾。
        # 自定义的长表格没有“\endfirsthead”标志
        searchstr = r'\\endfirsthead'
        matchobj = re.search(searchstr, singletablecontent, re.I | re.U)
        if matchobj is None:
            # 可能是自定义长表格，自定义长表格没有endfirsthead和endhead需要单独设置
            singletablecontent = self.__NormaltableRowHeight(rowno,rowheightinfo,singletablecontent)
            return singletablecontent

        if rowno==1: #因为长表格有两个表头，因此两个表头都要设置，避免换页后表头格式不一致。
            #修改长表格的表头
            firsthead = singletablecontent[0:matchobj.end()]
            rpos = firsthead.rfind(r"\\")
            firsthead = firsthead[0:rpos] + rowheightinfo + firsthead[rpos+len(r"\\"):len(firsthead)]

            #查找第2个表头，第2个表头以“\endhead”结尾
            searchstr = r'\\endhead'
            matchobj2 = re.search(searchstr, singletablecontent, re.I | re.U)
            endhead = singletablecontent[matchobj.end():matchobj2.end()]
            rpos = endhead.rfind(r"\\")

            endhead = endhead[0:rpos]+rowheightinfo+ endhead[rpos+len(r"\\"):len(firsthead)]

            singletablecontent = firsthead+endhead+singletablecontent[matchobj2.end():len(singletablecontent)]
            
        if rowno > 1: 
            #长表格先将表头去除掉，否则表头会包含很多“\\”，但不代表行。
            #长表格的表头以“\endlastfoot”结尾
            searchstr = r'\\endlastfoot'
            matchobj = re.search(searchstr, singletablecontent, re.I | re.U)
            tablehead = singletablecontent[0:matchobj.end()]
            othercontent = singletablecontent[matchobj.end():len(singletablecontent)]
            
            #修改表格内容，因为长表格有两个表头，因此表头后的行号起始行从1开始。
            othercontent = self.__ModifyNormalRowHeight(rowno,rowheightinfo,1,othercontent)

        singletablecontent = tablehead + othercontent
        return singletablecontent

    def __ModifyTableRowHeight(self,islongtable,singletablecontent, rowlistarry):
        
        #解析行信息，包含行号和行高信息       
        for rowlist in rowlistarry:
            
            if not self.__JudgeRowlistRule(rowlist):
                continue
            rowno = rowlist[0]
            rowheightinfo = r"\\[" + rowlist[1] +r"ex]"   #要替换的行高信息
            #根据正则表达式，查找换行符"\\",每一个“\\"代表表格的一行，需要在“\\”的末尾添加行高数据：“\\[rowhtstr+ex]”
            if islongtable:
                singletablecontent = self.__ModifyLongTableRowHeight(rowno, rowheightinfo,singletablecontent)
            else:
                singletablecontent = self.__NormaltableRowHeight(rowno,rowheightinfo,singletablecontent)
                
        return singletablecontent
    
    def __StartModifyTableAttr(self, singletablecontent, columnwidthlst, islongtable, isCusHead=True,jsonmultirow=None,tablestyle_dict=None):
        #修改表格的通用属性
        searchstr = r'(\\begin{tabular}|\\begin{tabulary})(\[[a-z]\]|{\\linewidth}\[[a-z]\])([\s\S].*)'
        #为了添加表格的通用属性，先对字符串做分割
        #python会把正则表达式中的分组自动分割，因此上面的正则表达式会自动分割为三个字符串
        #加上头尾字符串总共分为5个字符串数组。要修改第1维字符串为\\being{longtable},
        #第2维字符串直接删除，第3维字符串不变
        splittable = re.split(searchstr, singletablecontent,0, re.M | re.I|re.U )
        if splittable is None or len(splittable) < 5:
            #再修改长表格属性
            searchstr = r'\\begin{longtable}([\s\S].*)'
            #为了添加表格的通用属性，先对字符串做分割
            #python会把正则表达式中的分组自动分割，因此上面的正则表达式会自动分割为三个字符串
            #加上头尾字符串总共分为5个字符串数组。要修改第1维字符串为\\being{longtable},第2维字符串直接删除，
            #第3维字符串不变
            splittable = re.split(searchstr, singletablecontent,0, re.M | re.I|re.U )
            if len(splittable) < 3:
                #至少是3维的数组，否则不是预想的内容，不做处理
                return singletablecontent
            if isCusHead:
                newtable4 = self.__ModifyLongTableHead(splittable[2], 
                                                       columnwidthlst, 
                                                       self.tablesattrobj.headtype,
                                                       jsonmultirow,
                                                       tablestyle_dict)
                # begin{longtable}必须再加上，因为Python并不认为它是正则表达式，因此不再分组里面第0个分组为空。
                singletablecontent = splittable[0]+r'\begin{longtable}'+splittable[1]+newtable4
            if self.tablesattrobj.fontsize !="":
                #设置表格字体
                singletablecontent=self.__ModifyTableFontSize(singletablecontent,self.tablesattrobj.fontsize)
            return singletablecontent,True

        #拆分后splittable应该为5个字符串的数组，拆分后便于添加通用属性
        if self.tablesattrobj.rowtype != '':
            splittable[0] += self.tablesattrobj.rowtype + '\n'

        if isCusHead is True:
            #修改表头字体颜色为白色加粗
            newtable4 = self.__ModifyTableHead(splittable[4], 
                                               columnwidthlst,
                                               self.tablesattrobj.headtype,
                                               jsonmultirow,
                                               tablestyle_dict)
        else:
            newtable4 = splittable[4]

        singletablecontent = splittable[0]+splittable[1]+splittable[2]+splittable[3]+newtable4
        #根据配置设置表格为长表格，现在基本通过将sphinx的class属性设置为longtable实现，不再通过手动设置完成。
        if islongtable is True:
            singletablecontent = self.__ModifyTableLongHeadAndTail(singletablecontent)
            
        if self.tablesattrobj.fontsize !="":
            #设置表格字体
            singletablecontent=self.__ModifyTableFontSize(singletablecontent,self.tablesattrobj.fontsize)

        return singletablecontent,False
    
    def __ModifyTableLongHeadAndTail(self,singletablecontent):
        #长表格的头尾都要替换，因此单独封装成一个函数，否则长表格的表格序号将加2
        #替换第一行为长表格
        searchstr = r'(\\begin{savenotes}[\S]*?\\sphinxattablestart)'
        splittable = re.search(searchstr, singletablecontent,re.M | re.I|re.U )
        firstend =splittable.end()
        #替换为长表格
        tablefirstline = re.sub(r'\\sphinxattablestart',r'\n\\sphinxatlongtablestart\n', splittable.group(0), 1,re.M|re.I|re.U)
        if r'\sphinxthistablewithglobalstyle' in singletablecontent:
            tablefirstline += '\\sphinxthistablewithglobalstyle\n' 
        if r'\sphinxthistablewithvlinesstyle' in singletablecontent:
            tablefirstline += '\\sphinxthistablewithvlinesstyle\n'
        #替换第2行为长表格
        searchstr = r'(\\begin{tabular}|\\begin{tabulary})(\[[a-z]\]|{\\linewidth}\[[a-z]\])([\s\S].*)'
        splittable = re.search(searchstr, singletablecontent,re.I|re.U )
        #记录表头的位置
        headstartpos= splittable.start()
        headlastpos = splittable.end()
        tablesecondline = re.sub(r'\\begin{tabular}|\\begin{tabulary}',r'\\begin{longtable}', splittable.group(0), 1,re.I|re.U)
        tablesecondline = re.sub(r'\{\\linewidth\}',r'', tablesecondline, 1,re.I|re.U)

        #查找caption
        searchstr = r'\\sphinxcaption([\s\S].*)'
        splittable = re.search(searchstr, singletablecontent, re.I|re.U)
        if splittable is not None:
            longcaption = re.sub(r'\\sphinxcaption',r'\\caption', splittable.group(0), 1,re.I|re.U)
            #添加长表格专用指令
            longcaption += r"\\*[\sphinxlongtablecapskipadjust]"
            
            #构建长表个的表头部分
            longhead = tablefirstline + tablesecondline + '\n' + r'\sphinxthelongtablecaptionisattop' + '\n'+ longcaption+'\n'
        else:
            #如果没有caption需要把中间的内容加上，否则会丢失label导致连接失效
            longhead = tablefirstline + singletablecontent[firstend:headstartpos]+tablesecondline + '\n' 

        #替换表尾
        newtablecontent = singletablecontent[headlastpos:len(singletablecontent)]
        index = newtablecontent.rfind('\\end{tabulary}')
        if index > 0:
            endtable = newtablecontent[index:len(newtablecontent)]
            othercontent = newtablecontent[0:index]
            endtable = re.sub(r'(\\end{tabular}|\\end{tabulary})', r'\\end{longtable}', endtable, 1,re.M | re.I | re.U)
            endtable = re.sub(r'\\par', r'', endtable, 1, re.M | re.I | re.U)
            endtable = re.sub(r'\\sphinxattableend', r'\\sphinxatlongtableend', endtable, 1, re.M | re.I | re.U)
            newtablecontent = othercontent + endtable
        else:
            index = newtablecontent.rfind('\\end{tabular}')
            if index >0:
                endtable = newtablecontent[index:len(newtablecontent)]
                othercontent = newtablecontent[0:index]
                endtable = re.sub(r'(\\end{tabular}|\\end{tabulary})', r'\\end{longtable}', endtable, 1,re.M | re.I | re.U)
                endtable = re.sub(r'\\par', r'', endtable, 1, re.M | re.I | re.U)
                endtable = re.sub(r'\\sphinxattableend', r'\\sphinxatlongtableend', endtable, 1, re.M | re.I | re.U)
                newtablecontent = othercontent + endtable

        singletablecontent = longhead + newtablecontent
        return singletablecontent

    #为表格增加h属性对齐方式，避免表格浮动，让表格在当前位置显示
    def __AddHAttrForTable(self,content):

        newcontent=""
        searchstr = r'(\\begin{tabular}|\\begin{tabulary}|\\begin{longtable})({\\linewidth})?(\[(?P<attr>[a-z]{1,4})\])?([\s\S].*)'
        m = re.search(searchstr, content, re.M|re.I|re.U )
        attrcontent = m.group('attr')
        posarr = m.span()
        if m.group(3) == '':   #group(3)是[htp]的组合，如果没有表格属性则添加上表格属性
            newcontent = content[0:posarr[0]]+m.group(1)+m.group(2)+r'[h]'+m.group(5)+content[posarr[1]:len(content)]
        else:
            replacestr = m.group(4)  #需要被替换的表格属性
            #判断该表格是否有P属性，如果有p属性需要删掉
            replacestr.replace('p','')
            #给该表格添加h属性，避免表格浮动，让表格在当前位置显示
            replacestr = 'h'+replacestr
            newcontent = content[0:posarr[0]]+m.group(1)+m.group(2)+'['+replacestr+']'+m.group(5)+content[posarr[1]:len(content)]

        return newcontent
    def __GetColumnParaByWidths(self,content):
        # 可能用的widths参数设置的列宽，再根据widths参数解析列宽
        columnwidthlst = []
        # 如果列宽不等于100，为了使表格不超过页面显示范围，以100为基数重新调整各列的列宽
        newcolumnwidth = "|"
        isadjustcol = False
        newcontent = content
        try:
            searchstr = r'(?<=\|)\\X\{(?P<count1>[0-9]*)\}\{(?P<count2>[0-9]*)\}(?=\|)'
            pattern = re.compile(searchstr, re.I | re.U)
            match = pattern.finditer(content)
            if match is None:
                return [],content
            for m in match:
                count1 = int(m.group('count1'))
                count2 = int(m.group('count2'))
                
                tempcount = round(count1/count2,2)
                colwidth = "%(float)s\\textwidth"
                colwidth = colwidth % {
                    'float': str(tempcount)
                }
                columnwidthlst.append(colwidth)
                if count2 != 100:
                    isadjustcol = True
                    #重新调整count1和count2的值，并加入列表
                    percolwidth = r"\X{" + str(int(tempcount * 100)) + "}{100}|"
                    newcolumnwidth = newcolumnwidth + percolwidth
                    
            if isadjustcol is True:
                searchstr = r"(?<=\{)\|\\X[\s\S]*?\|(?=\})"
                match = re.search(searchstr,content,re.I | re.U)
                if match is not None:
                    newcontent = content.replace(match.group(),newcolumnwidth,1)

        except Exception as e:
            print(e)
        finally:
            return columnwidthlst,newcontent
        
    def __GetColumnParameters(self,content):
        '''
        得到通过tablecolumns参数设置的列宽，multirow的自动换行，必须有列宽参数，否则无法实现自动换行。
        而且不能通过widths参数设置每列的列宽，否则生成的列宽无法用于multirow，依然无法实现multirow的自动换行
        :param content: 
        :return: 
        '''
        newcontent = content
        columnwidthlst=[] #保存每一列宽度的列表
        #根据表格内容，得到每一列宽度参数，方便multirow的时候自动换行
        #m垂直居中，p垂直靠上，b垂直靠下
        searchstr= r'(?<=\|)[\s\S]*?[m|p|b]\{(?P<width>[\s\S]*?)\}[\s\S]*?(?=\|)'
        pattern = re.compile(searchstr,re.I | re.U)
        match = pattern.finditer(content)
        if match is None:
            columnwidthlst,newcontent = self.__GetColumnParaByWidths(content)
            return columnwidthlst,newcontent
        
        for m in match:
            columnwidthlst.append(m.group('width'))
        if len(columnwidthlst) ==0:
            columnwidthlst,newcontent= self.__GetColumnParaByWidths(content)
        return columnwidthlst,newcontent
        
    def __FindTableHeadForFamily(self,content,isLongTable):
        '''
        根据sphinxstyletheadfamily标志查找表格的表头，有sphinxstyletheadfamily标志的表头，源文件必须以“======”分割。
        具有合并单元格的表头，源文件必须以“===”分割，而且表头最多支持两行，否则合并单元格的表头不支持渲染。
        :param content: 表格内容
        :return: headcontent 表格内容
        '''
        #查找表头的起始位置，以第一个\hline开始。
        if isLongTable is True:
            searchstr =r"(\\hline[\s\S]*)(?=\\endfirsthead)"
            matchobj = re.search(searchstr, content, re.I|re.M|re.U)
            tablehead = matchobj.group()
        else:
            searchstr = r'\\hline'
            matchobj = re.search(searchstr,content,re.I)
            if matchobj is None:
                return ""
            startpos = matchobj.start() #表头的起始位置
            familypos = content.rfind(r'\sphinxstyletheadfamily') #查找最后一个sphinxstyletheadfamily出现的位置
            if familypos ==-1:
                return ""
            
            #从familypos开始，查找到的第一个“\\”，则为表头的行结束符号。
            endpos = content.find(r"\hline",familypos)
            # 得到表头内容，表头内容从起始\hline开始，到\hline结束，这种查找方式，兼容表头为两行的情况。
            tablehead = content[startpos:endpos+len(r"\hline")] 
            
        return tablehead
    
    def __FindAndFlagPos(self,tablehead,andpos):
        if andpos != -1:
            # 查找&前面有没有转义字符，有转义字符，则&不为单元格分割符，需要再继续查找
            preflag = tablehead[andpos-1]
            if preflag == '\\':
                #如果是\&说明这个&不是单元格的分隔符,则需要继续查找
                newpos=tablehead.find("&",andpos+1)
                andpos = self.__FindAndFlagPos(tablehead,newpos)
                return andpos
            else:
                return andpos
        return andpos
    
    def __FindCellContentForHead(self,tablehead,startpos):
        #查找每个单元格的内容
        if startpos == -1:
            return 0,-1,None

        # 查找&和\\的位置，&为单元格的分割标志,\\为行结束标志，一个单元格的分割标志可能是&，也可能是\\
        cellcontent = []
        andpos = tablehead.find("&", startpos)
        # 得到实际的分隔符位置，避免&内容里面的符号，而不是单元格分隔符
        andpos = self.__FindAndFlagPos(tablehead, andpos)
        linepos = tablehead.find(r"\\", startpos)
        
        if andpos ==-1 and linepos ==-1:
            return 0,-1,None

        newpos = -1
        lineflag = 0  # 是否启动新行的标志。0说明在当前行；1说明遇到了\\，需要启动新的一行
        if andpos == -1 or linepos < andpos:
            # 该单元格的分割符为\\
            cellcontent.append(tablehead[startpos:linepos])
            cellcontent.append(r"\\")
            newpos = linepos + len(r"\\")
            lineflag = 1
        else:
            cellcontent.append(tablehead[startpos:andpos])
            cellcontent.append(r"&")
            newpos = andpos + len(r"&")
        return lineflag,newpos,cellcontent
    '''
    #暂时注释，不在使用该函数，使用上面函数的写法
    def __delNotMustStrFromContent(self, content):
        content = content.strip().strip("\n").strip("\r")
        for delstr in self.delstrlst:
            if delstr in content:
                content = content.replace(delstr,"")
        
        content = content.strip().strip("\n").strip("\r")
        print('content=%s' % content)
        strlst = content.split('\n')
        newcontent = ""
        for i in strlst:
            if i != "":
                i.strip("\r")
            if i != strlst[-1] and i != "":
                i = i + r"\\ "
            newcontent = newcontent + i
        content = self.autolinetemplat % {
            'autocontent': newcontent
        }
        return content
    '''

    def __delNotMustrowStrFromContent(self, content, ismanualwraplst=None):
        """
        该函数用于multirow的手动换行，为了更好的显示效果，multicolumn的手动换行请参见__delNotMustStrFromContent
        :param content: 
        :param ismanualwraplst: list类型，只有一个元素。是否有手动换行，为了回传，并兼容之前的调用，设计了list类型。
        :return: 
        """
        # 实现自动换行
        # 去掉首尾换行和空白字符
        content = content.strip().strip("\n").strip("\r")
        # 删除单元格内容里非必须的成分
        for delstr in self.delstrlst:
            if delstr in content:
                content = content.replace(delstr, "")
        # 正则匹配既是加了“r”，也要用转义字符，因为r只是去掉了python的转义，
        # 但正则本身也需要一层转义
        content = content.strip().strip("\n").strip("\r")
        searchstr = r"\\sphinxAtStartPar"
        match = re.search(searchstr, content, re.I | re.M)
        if match is not None:
            # 说明在字符串的起始位置匹配成功
            content = content[match.end():len(content)]
        # 再去掉首尾空格和换行
        content = content.strip().strip("\n").strip("\r")
        if r"\sphinxAtStartPar" in content:
            # 字符串替换比较，如果用了r则无需再加一次转义字符
            # 替换所有的回车换行符，否则autolinetemplat中间不能有回车换行，否则将导致错误
            content = content.replace("\n", "").replace("\r", "")
            content = content.replace(r"\sphinxAtStartPar", r"\newline ")
            if ismanualwraplst is not None:
                ismanualwraplst[0] = True
        else:
            strlst = content.split('\n')
            newcontent = ""
            for i in range(0, len(strlst)):
                str = strlst[i].strip("\r")
                if str != "":
                    strlst[i] = str
                else:
                    if i > 0 and i < (len(strlst) - 1) and strlst[i - 1] != r"\newline ":
                        strlst[i] = r"\newline "
                
            content = "".join(strlst)
            if r"\newline" in content and ismanualwraplst is not None:
                ismanualwraplst[0] = True
                
        return content
    
    def __delNotMustStrFromContent(self,content,ismanualwraplst=None):
        """
        该函数用于multicolumn的手动换行，为了更好的显示效果，multirow的手动换行请参见__delNotMustrowStrFromContent
        :param content: 
        :param ismanualwraplst: list类型，只有一个元素。是否有手动换行，为了回传，并兼容之前的调用，设计了list类型。
        :return: 
        """
        #实现自动换行
        #去掉首尾换行和空白字符
        content = content.strip().strip("\n").strip("\r")
        #删除单元格内容里非必须的成分
        for delstr in self.delstrlst:
            if delstr in content:
                content = content.replace(delstr,"")
        # 正则匹配既是加了“r”，也要用转义字符，因为r只是去掉了python的转义，
        # 但正则本身也需要一层转义
        content = content.strip().strip("\n").strip("\r")
        searchstr = r"\\sphinxAtStartPar"
        match = re.search(searchstr,content,re.I|re.M)
        if match is not None:
            #说明在字符串的起始位置匹配成功
            content = content[match.end():len(content)]
        # 再去掉首尾空格和换行
        content = content.strip().strip("\n").strip("\r")
        if r"\sphinxAtStartPar" in content:
            #字符串替换比较，如果用了r则无需再加一次转义字符
            #替换所有的回车换行符，否则autolinetemplat中间不能有回车换行，否则将导致错误
            content=content.replace("\n","").replace("\r","")
            content = content.replace(r"\sphinxAtStartPar", r"\\ ")
            content = self.autolinetemplat % {
                'autocontent': content
            }
            if ismanualwraplst is not None:
                ismanualwraplst[0] = True
        else:
            strlst = content.split('\n')
            newcontent = ""
            for i in range(0,len(strlst)):
                str = strlst[i].strip("\r")
                if str != "":
                    strlst[i]=str
                else:
                    if i>0 and i< (len(strlst)-1) and strlst[i-1] != r"\\ ":
                        strlst[i]=r"\\ "
                        
            newcontent = "".join(strlst)
            if r"\\ " in newcontent:
                content = self.autolinetemplat % {
                    'autocontent': newcontent
                }
                if ismanualwraplst is not None:
                    ismanualwraplst[0] = True
            else:
                content = newcontent
                    
        return content

    def __ModifySphinxMultirow(self,cellstr,contentdict,columnwidthlst,jsonmultirow,lineindex,colindex,multirowindex,tablestyle_dict,isvertical=False):
        '''
        修改multirow内容
        取出关键内容合并单元格的数量和单元格内容
        :param isvertical:如果是渲染第一列，multirow加\raggedright参数为了对齐
        渲染行，无需加该参数，显示效果没明显差别。
        '''
        #searchstr=r"(?<=\\sphinxmultirow){(?P<count>[0-9]*)}{(?P<flag>[0-9]*)}([\s\S]*?)\\sphinxstyletheadfamily(?P<content>[\s\S]*?)\\par([\s\S]*?)(?=\}\%)"
        searchstr=r"(?<=\\sphinxmultirow){(?P<count>[1-9]*)}{(?P<flag>[0-9]*)}{([\s\S]*?){\\sphinxcolwidth{[0-9]*}{[0-9]*}}[\s]*(\\sphinxstyletheadfamily)?(?P<content>[\s\S]*?)\\par([\s\S]*?)(?=\}\%)"
        match = re.finditer(searchstr, cellstr, re.M | re.I | re.U)
        count = ""
        flag = ""
        content = ""
        for m in match:
            if count == "":
                count = m.group('count')
            if flag == "":
                flag = m.group('flag')
            if content == "":
                content = m.group('content')

        #如果是自动换行，设置列宽
        coluwidth = "*"
        if len(columnwidthlst) > 0:
            if tablestyle_dict is not None and \
                __dictIsHasKey__(tablestyle_dict,"ismultirowautowrap"):
                if tablestyle_dict['ismultirowautowrap'] is True:
                    coluwidth = columnwidthlst[colindex-1]
            else:
                if self.tablesattrobj.ismultirowautowrap is True:
                    coluwidth = columnwidthlst[colindex-1]
            
        #if (jsonmultirow is not None) and \
        #    (multirowindex>=0 and len(jsonmultirow)>multirowindex) and \
        #    (len(columnwidthlst)>jsonmultirow[multirowindex]-1):
        #    #得到要设置自动换行multirow的列号
        #    coluwidth = columnwidthlst[jsonmultirow[multirowindex]-1]
        #    #print("coluwidth = %s" % coluwidth)
        #else:
        #    coluwidth = "*"
        
        #if r"\sphinxstyletheadfamily" in cellstr:
        #实测，multirow要支持渲染背景色，必须带上sphinxstyletheadfamily，否则会编译失败
        sphinxstyletheadfamily = r"\sphinxstyletheadfamily"
        raggedright = ""
        if (isvertical is True) or \
                (tablestyle_dict is not None and \
                 __dictIsHasKey__(tablestyle_dict, "ismultirowatupcell") and \
                 tablestyle_dict['ismultirowatupcell'] is True) or \
                self.tablesattrobj.ismultirowatupcell is True:
            #raggedright = r"{\raggedright}"
            # 得到单元格内容
            cellstr = self.commonstr % {
                'sphinxstyletheadfamily': sphinxstyletheadfamily,
                'content': self.__delNotMustrowStrFromContent(content)
            }
            multistr = self.multirowstr % {
                'count': count,
                'coluwidth': coluwidth,
                'raggedright': raggedright,
                'sphinxstyletheadfamily': sphinxstyletheadfamily,
                'content': ""
            }
            #将flag对应的内容加到字典里，以方便后续交换
            contentdict[flag] = str(int(count) + lineindex-1) + "," + multistr #为了支持超过两行的合并单元格，将count数量也带出去。第一个","之前的就是count数量
            
            #返回空的字符串
            return cellstr
        else:
            ismanualwraplst = [False]
            if (tablestyle_dict is not None and \
                    __dictIsHasKey__(tablestyle_dict, "ismultirowatlowcell") and \
                    tablestyle_dict['ismultirowatlowcell'] is True ) or \
                    self.tablesattrobj.ismultirowatlowcell is True:
                
                simplemultirow = self.simplemultirow % {
                    'count': count,
                    'coluwidth': coluwidth
                }
                
                cellstr = self.commonstr % {
                    'sphinxstyletheadfamily': sphinxstyletheadfamily,
                    'content': self.__delNotMustrowStrFromContent(content, ismanualwraplst)
                }
                
                contentdict[flag] = str(int(count) + lineindex - 1) + "," + cellstr
                
                return simplemultirow
                    
            multistr = self.multirowstr % {
                'count': count,
                'coluwidth': coluwidth,
                'raggedright': raggedright,
                'sphinxstyletheadfamily': sphinxstyletheadfamily,
                'content': self.__delNotMustStrFromContent(content)
            }
            # 将flag对应的内容加到字典里，以方便后续交换
            contentdict[flag] = str(int(count) + lineindex - 1) + "," + multistr  # 为了支持超过两行的合并单元格，将count数量也带出去。第一个","之前的就是count数量
            
            # 返回空的字符串
            return self.emptystr


    
    def __ModifyMulticolumn(self,cellstr,tablestyle_dict):
        '''
        :param cellstr: 
        :return: 
        '''
        searchstr = r"(?<=\\multicolumn){(?P<count>[1-9]*)}\{(?P<flag>[\s\S]*?)\}\{%(?P<prefix>[\s\S]*?)\\sphinxstyletheadfamily(?P<content>[\s\S]*?)\\par(?P<suffix>[\s\S]*)"
        match = re.finditer(searchstr, cellstr, re.M | re.I | re.U)
        count = ""
        content = ""
        flag=""
        prefix=""
        suffix=""
        for m in match:
            if count == "":
                count = m.group('count')
            if flag == "":
                flag = m.group('flag')
            if content == "":
                content = m.group('content')
            if prefix == "":
                prefix = m.group('prefix')
                #如果有\begin{varwidth}[t],替换为\begin{varwidth}[c]
                #[t]会使得当自动换行的时候，单元格上面会有很大的空白
                #print('===========================================')
                #print(prefix)
                searchstr = r"\\begin{varwidth}\[t\]"
                replstr = r"\\begin{varwidth}[c]"
                prefix = re.sub(searchstr,replstr,prefix,1,re.I|re.U|re.M)
                #print(prefix)
                #print('===========================================')
            if suffix == "":
                suffix = '\n\\par' + m.group('suffix')  #因为正则表达式把\par去掉了，再加上
                
        #行合并单元格，改成居中显示
        newflag=re.sub(r"[a-z]",r"c",flag,1,re.I)
        #newflag = flag
        if (tablestyle_dict is not None and \
            __dictIsHasKey__(tablestyle_dict,'ismulticolleftalign') and \
                tablestyle_dict['ismulticolleftalign'] is True) or \
                self.tablesattrobj.ismulticolleftalign is True:
                #multistr = self.comcolstr % {
                #    'count': count,
                #    'flag': flag,
                #    'sphinxstyletheadfamily': "\\sphinxstyletheadfamily",
                #    'content': self.__delNotMustStrFromContent(content)
                #}
                multistr = self.multicolumnstr % {
                    'count':count,
                    'flag': flag,
                    'prefix':prefix,
                    'sphinxstyletheadfamily': "\\sphinxstyletheadfamily",
                    'content':self.__delNotMustStrFromContent(content),
                    'suffix':suffix
                }
        else:
                multistr = self.multicolumnstr % {
                    'count':count,
                    'flag': newflag,
                    'prefix':prefix,
                    'sphinxstyletheadfamily': "\\sphinxstyletheadfamily",
                    'content':self.__delNotMustStrFromContent(content),
                    'suffix':suffix
                }
    
        #multistr = self.comcolstr % {
        #    'count': count,
        #    'flag': flag,
        #    'sphinxstyletheadfamily': "\\sphinxstyletheadfamily",
        #    'content': self.__delNotMustStrFromContent(content)
        #}
        return multistr
    
    def __ModifyCellContentForMergeCell(self,cellcontent,
                                        tablestyle_dict,islongtable,
                                        contentdict,columnwidthlst,jsonmultirow,
                                        colindex,lineindex,multirowindex,ismulticol = False):
        '''
        修改单元格内容
        :param cellcontent: 
        :param ismulticol: 当前行是否有列合并单元格。
        如果有列合并单元格，则独立的单元格需要用multcolumn命令，否则无法对齐。
        :return: 
        '''
        multirowflag = False #是否是multirow单元格
        cellstr = cellcontent[0] #取得单元格内容
        sphinxstyletheadfamily=""
        if r'\sphinxmultirow' in cellstr:
            cellcontent[0] = self.__ModifySphinxMultirow(cellstr,
                                                         contentdict,
                                                         columnwidthlst,
                                                         jsonmultirow,
                                                         lineindex,
                                                         colindex,
                                                         multirowindex,
                                                         tablestyle_dict)
            multirowflag = True
        elif r'\multicolumn' in cellstr:
            cellcontent[0] = self.__ModifyMulticolumn(cellstr,tablestyle_dict)
        else:
            if colindex == 1:
                if len(columnwidthlst)>0 and colindex > 0:
                    flag = "|m{" + columnwidthlst[colindex - 1] + "}|"
                else:
                    flag = "|l|"
            else:
                if len(columnwidthlst) > 0 and colindex > 0:
                    flag = "m{" + columnwidthlst[colindex - 1] + "}|"
                else:
                    flag = "l|"
                
            #查找\sphinxAtStartPar位置
            pos = cellstr.find(r'\sphinxstyletheadfamily')
            if pos != -1:
                cellstr = cellstr[pos+len(r'\sphinxstyletheadfamily'):len(cellstr)]
                sphinxstyletheadfamily = "\\sphinxstyletheadfamily"

            ismanualwraplst=[False]
            content = self.__delNotMustStrFromContent(cellstr,ismanualwraplst)
            if ismanualwraplst[0] is True:
                if colindex == 1:
                    flag = "|l|"
                else:
                    flag = "l|"
                colwidth = ""
                if (len(columnwidthlst) > 0) and (colindex > 0):
                    colwidth = columnwidthlst[colindex-1]
                    
                fullcontentstr = self.normalstr % {
                   'flag': flag,
                   'colwidth': colwidth,
                   'sphinxstyletheadfamily': sphinxstyletheadfamily,
                   'content': self.__delNotMustStrFromContent(cellstr)
                }
            else:
                fullcontentstr = self.comcolstr % {
                    'count': "1",
                    'flag': flag,
                    'sphinxstyletheadfamily': sphinxstyletheadfamily,
                    'content': self.__delNotMustStrFromContent(cellstr)
                }
            #if ismulticol is True:
            #必须用multicolumn命令，否则当单元格后面还有列的时候，刷新会出现问题
            #fullcontentstr = self.comcolstr % {
            #   'count': "1",
            #   'flag': flag,
            #   'sphinxstyletheadfamily': sphinxstyletheadfamily,
            #   'content': self.__delNotMustStrFromContent(cellstr)
            #}
            #else:
            #    fullcontentstr = self.commonstr % {
            #        'sphinxstyletheadfamily': "\\sphinxstyletheadfamily",
            #        'content': self.__delNotMustStrFromContent(contentstr)
            #    }
            cellcontent[0] = fullcontentstr

        return multirowflag
    
    def __FindMergeRowCountAndContent(self,multirowstr):
        #找出合并行的数量和合并单元格的内容
        count = 0
        newmultistr = multirowstr

        try:
            searchstr = "([1-9]*?)(?=,)"
            match = re.match(searchstr,newmultistr,re.I)
            if match is not None:
                count = int(match.group(0))
                startpos = match.end()
                newmultistr = multirowstr[startpos+1:len(multirowstr)]
        finally:
            return count,newmultistr

    def __AdjustMultirowContent(self,cellcontent,contentdict,lineindex,colindex,columnwidthlst):
        '''
        调整multirow单元格到对应的下面的单元格
        '''
        #查找flag数量
        cellstr = cellcontent[0]
        #查找有没有\cline指令，如果有这个指令，为了划线清晰在前面再多加一个。
        searchstr= r"\\cline\{[\s\S]*?\}"
        match = re.search(searchstr,cellstr,re.I | re.M)
        if match is not None:
            cline = match.group()
            #去掉开头的换行符，否则影响后面的正则表达式,导致正则表达式查找失败
            cellstr = cellstr.lstrip()
            cellstr = cline + cellstr

        searchstr = r"\\sphinxtablestrut\{(?P<flag>[0-9]*)\}"
        matchobj = re.search(searchstr, cellstr, re.I | re.M | re.U)
        flag = matchobj.group('flag')
        #得到修改后的multirow内容
        if __dictIsHasKey__(contentdict,flag):
            multirowstr = contentdict[flag]
            #替换实际列号
            replstr = r'\\sphinxtablestrut{' + str(colindex)+'}'
            cellstr = re.sub(searchstr,replstr,cellstr,1,re.I|re.U|re.M)
            #找出合并行的最大数量
            count,multirowstr = self.__FindMergeRowCountAndContent(multirowstr)
            if lineindex == count: #是最后一行了再合并，否则会重复
                cellcontent[0] = cellstr+"\n"+multirowstr
            else:
                cellcontent[0] = cellstr+'\n' + r"\cellcolor" + self.tablesattrobj.headtype + "{}"

        return cellcontent

    def __GetFactColIndex(self,cellcontent):
        '''
        得到真实的列索引。当有multicolumn的时候，列索引需要加合并单元格的数量。
        返回合并单元格的数量，没合并单元格返回1
        :para cellcontent: 单元格内容字符串
        '''
        if len(cellcontent) ==0:
            return 0
        searchstr = r"\\multicolumn{(?P<count>[1-9]+)}"
        match = re.search(searchstr,cellcontent,re.I|re.U)
        if match is None:
            return 1 #不是合并单元格，默认加1列
        factcount = 1
        try:
            factcount = int(match.group('count'))
        except Exception as e:
            print('------------------')
            print(e)
            # traceback.print_stack()
            traceback.print_exc()
            print('------------------')
            
        return factcount
    def __ModifyTableHeadForFamilyList(self,tablehead,isLongtable,columnwidthlst,jsonmultirow,tablestyle_dict):
        '''
        用列表的方式逐个拆解每个单元格内容，列表格式为[['cell content','分割符号&或者\\'],[],[]]
        :param tablehead: 
        :return: 
        '''
        #内容的起始位置
        startpos = tablehead.find(r'\\hline') + len(r'\\hline')

        contentdict ={} #创建一个字典，用来保存mlutirow交换内容
        #得到每个单元格的内容
        colindex = 0  #是否第一个单元格的索引
        lineindex = 1  #行索引，为了支持多行合并，默认起始在第一行
        # multirowindex：第几个multirow单元格的索引号，用于作为conf.json中multirowcolumn字段的索引值，
        # 取出当前multirow单元格真实所在列
        #得到该列的列宽，填写到multirow指令里面，实现multirow指令中的内容自动换行
        multirowindex = 0
        lineflag = 0
        newtablehead = "\\hline\n"
        while startpos !=-1:
            multirowflag = False  # 是否是multirow处理，是的话__ModifyCellContentForMergeCell返回true，默认返回false
            lineflag,startpos,cellcontent = self.__FindCellContentForHead(tablehead,startpos)
            if cellcontent is not None:
                factcolcount = self.__GetFactColIndex(cellcontent[0])
                colindex = colindex+factcolcount #当前列号
                if r"\sphinxtablestrut" in cellcontent[0]:
                    # 需要将multirow的内容copy到该单元格
                    self.__AdjustMultirowContent(cellcontent, contentdict,lineindex,colindex,columnwidthlst)
                else:
                    
                    multirowflag = self.__ModifyCellContentForMergeCell(cellcontent,
                                                                        tablestyle_dict,
                                                                        isLongtable,
                                                                        contentdict,
                                                                        columnwidthlst,
                                                                        jsonmultirow,
                                                                        colindex,
                                                                        lineindex,
                                                                        multirowindex)
                newtablehead = newtablehead+cellcontent[0] + "\n" + cellcontent[1]+ "\n"
                
            if multirowflag is True:
                multirowindex = multirowindex+1
            lineindex = lineindex + lineflag  # 计算新的行数
            if lineflag:
                # 如果开启了新行，则列索引从0开始计算
                colindex = 0

        #组成新的表头内容
        newtablehead =newtablehead + "\\hline\n"
        return newtablehead
    def __modifyTableHeadForMinusSign(self,content,isLongTable,headtype,tablestyle_dict):
        """
        如果表头不是用“=”分割的，而是用“-”分割，调用该函数修改表头
        """
        # 先找出第一行
        if isLongTable:
            searchstr = r'\\sphinxtableatstartofbodyhook(?P<content>[\s\S]*?)\\hline'
        else:
            searchstr = r'\\hline(?P<content>[\s\S]*?)\\hline'
        m = re.search(searchstr, content, re.M | re.I | re.U)
        headcontent = m.group(1)  # 匹配到的第一个即为表头内容
        posarr = m.span(1)  # 保存起始位置和结束位置，便于组成新的内容

        if 'multicolumn' in headcontent:
            return content

        if r'\sphinxstyletheadfamily' in headcontent:
            pattern = re.compile(r'(?<=\\sphinxstyletheadfamily)(?P<value>[\s\S]*?)(?=(\\unskip|&)|\\\\)',
                                 re.M | re.I | re.U)
            aftercontent = headcontent
            # pattern = re.compile(r'(?<=\\sphinxstylethead{\\sphinxstyletheadfamily)([\s\S]*?)(?=\\unskip}\\relax &)', re.M | re.I|re.U)
        else:
            aftercontent = headcontent.replace(r'\\', '&', 1)
            pattern = re.compile(r'(?P<value>[\s\S]*?)(&\s{1})', re.M | re.I | re.U)
            # pattern = re.compile(r'[\s\S]*?&|\\unskip', re.M | re.I|re.U)

        mobjarr = pattern.finditer(aftercontent)
        headlist = []
        preposlist = []
        fontcolor = ""
        for mobj in mobjarr:
            amarr = mobj.group('value')
            curposlist = [mobj.start(), mobj.start() + len(amarr)]

            # 用表头内容数组替换
            if self.tablesattrobj.headfontcolor != "":
                fontcolor = self.tablesattrobj.headfontcolor
            # 先去掉首尾空格，避免首尾有空格无法去掉回车换行符
            amarr = amarr.strip()
            amarr = amarr.strip('\r')
            amarr = amarr.strip('\n')
            amarr = amarr.strip()  # 再去掉首尾空格，避免多余的空格出现
            if amarr == '':
                continue
            amarr = self.__delNotMustStrFromContent(amarr)
            fontcolor = fontcolor.replace('{}', '{' + amarr + '}', 1)
            # 当表头没有合并单元格时，不能用cellcolor，否则表头不对齐
            # fontcolor = self.commonstr % {
            #    'sphinxstyletheadfamily':'',
            #    'content': self.__delNotMustStrFromContent(amarr)
            # }
            if len(preposlist) > 0:
                headlist.append(headcontent[preposlist[1]:curposlist[0]])
            else:
                headlist.append(headcontent[0:curposlist[0]])
            headlist.append(fontcolor)
            preposlist = curposlist

        headlist.append(headcontent[preposlist[1]:len(headcontent)])  # 把最后一个字符串加上
        headcontent = ''
        for prelist in headlist:
            headcontent = headcontent + prelist + '\n'
        newcontent = content[0:posarr[0]] + r'\rowcolor' + headtype + '\n' + headcontent + content[
                                                                                           posarr[1]:len(content)]
        return newcontent
    #修改sphinx自动生成的长表格表头
    def __ModifyLongTableHead(self,content,columnwidthlst,headtype,jsonmultirow,tablestyle_dict):

        tablehead = self.__FindTableHeadForFamily(content,True)
        if tablehead != "" and (r"\sphinxmultirow" in tablehead or r"\multicolumn" in tablehead):
            #长表格自定义sphinxcolwidth的第2个参数默认为5，否则无法实现自动换行
            #self.normalstr= self.normalstr % {
            #    'flag': '%(flag)s',
            #    'colwidth':'5',
            #    'sphinxstyletheadfamily':'%(sphinxstyletheadfamily)s',
            #    'content':'%(content)s'
            #}
            newtablehead = self.__ModifyTableHeadForFamilyList(tablehead,
                                                               True,
                                                               columnwidthlst,
                                                               jsonmultirow,
                                                               tablestyle_dict)
            return content.replace(tablehead,newtablehead,2)

        #先找出第一行
        if tablehead.strip() == r"\hline":
            #进入该分支，说明表格设置为了长表格，但是表头没有用“=”分割，用“-”做的分割。
            return self.__modifyTableHeadForMinusSign(content,True,headtype,tablestyle_dict)
        
        searchstr = r'\\hline(?P<content>[\s\S]*?)\\hline'
        pattern = re.compile(searchstr,re.M | re.I|re.U)
        matchiter = pattern.finditer(content)
        posarr = []
        i = 0
        for m in matchiter:

            if i > 1:
                break

            posarr.append([])
            posarr[i] = m.span() #保存起始位置和结束位置，便于组成新的内容

            if i ==0:
                newcontent = content[0:posarr[i][0]]
            else:
                newcontent = newcontent+content[posarr[i-1][1]:posarr[i][0]]

            newcontent += r'\hline\rowcolor'+headtype
            headcontent = m.group(1) #匹配到的第一个即为表头内容

            if 'multicolumn' in headcontent:
                return content

            headlist = []

            if r'\sphinxstyletheadfamily' in headcontent:
                pattern = re.compile(r'(?<=\\sphinxstyletheadfamily)(?P<value>[\s\S]*?)(?=(\\unskip|&)|\\\\)', re.M | re.I|re.U)
                aftercontent = headcontent
                mobjarr = pattern.finditer(aftercontent)

                preposlist = []
                for mobj in mobjarr:
                    amarr = mobj.group('value')
                    curposlist = mobj.span()

                    #用表头内容数组替换
                    fontcolor = self.tablesattrobj.headfontcolor
                    #先去掉首尾空格，避免首尾有空格无法去掉回车换行符
                    amarr = amarr.strip()
                    amarr = amarr.strip('\r')
                    amarr = amarr.strip('\n')
                    amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现
                    if amarr == '':
                        continue
                    amarr = self.__delNotMustStrFromContent(amarr)
                    fontcolor = fontcolor.replace('{}','{'+ amarr+'}',1)
                    # 当表头没有合并单元格时，不能用cellcolor，否则表头不对齐
                    #fontcolor = self.commonstr % {
                    #    'sphinxstyletheadfamily': '',
                    #    'content': self.__delNotMustStrFromContent(amarr)
                    #}
                    if len(preposlist) > 0:
                        headlist.append(headcontent[preposlist[1]:curposlist[0]])
                    else:
                        headlist.append(headcontent[0:curposlist[0]])
                    headlist.append(fontcolor)
                    preposlist = curposlist
                headlist.append(headcontent[preposlist[1]:len(headcontent)])  #把最后一个字符串加上
                headcontent = ''
                for prelist in headlist:
                    headcontent = headcontent + prelist + '\n'
                newcontent += headcontent+r'\hline'
            i +=1
        newcontent += content[posarr[i-1][1]:len(content)]
        return newcontent

    def __ModifyTableHead(self, content,columnwidthlst, headtype,jsonmultirow,tablestyle_dict):
        
        tablehead = self.__FindTableHeadForFamily(content,False)
        if tablehead != "" and (r"\sphinxmultirow" in tablehead or r"\multicolumn" in tablehead):
            #为了实现自动换行，普通表格自定义sphinxcolwidth的第2个参数默认为3，否则无法实现自动换行
            #self.normalstr= self.normalstr % {
            #    'flag': '%(flag)s',
            #    'colwidth':'3',
            #    'sphinxstyletheadfamily':'%(sphinxstyletheadfamily)s',
            #    'content':'%(content)s'
            #}
            # 得到每一列的列宽供multirow自动换行使用
            newtablehead = self.__ModifyTableHeadForFamilyList(tablehead,
                                                               False,
                                                               columnwidthlst,
                                                               jsonmultirow,
                                                               tablestyle_dict)
            return content.replace(tablehead,newtablehead,2)
        
        return self.__modifyTableHeadForMinusSign(content,False,headtype,tablestyle_dict)
        """
        #先找出第一行
        searchstr = r'\\hline(?P<content>[\s\S]*?)\\hline'
        m = re.search(searchstr, content, re.M|re.I|re.U )
        headcontent = m.group(1) #匹配到的第一个即为表头内容
        posarr = m.span(1)  #保存起始位置和结束位置，便于组成新的内容

        if 'multicolumn' in headcontent:
            return content

        if r'\sphinxstyletheadfamily' in headcontent:
            pattern = re.compile(r'(?<=\\sphinxstyletheadfamily)(?P<value>[\s\S]*?)(?=(\\unskip|&)|\\\\)', re.M | re.I|re.U)
            aftercontent = headcontent
            #pattern = re.compile(r'(?<=\\sphinxstylethead{\\sphinxstyletheadfamily)([\s\S]*?)(?=\\unskip}\\relax &)', re.M | re.I|re.U)
        else:
            aftercontent = headcontent.replace(r'\\','&',1)
            pattern = re.compile(r'(?P<value>[\s\S]*?)(&\s{1})', re.M | re.I|re.U)
            #pattern = re.compile(r'[\s\S]*?&|\\unskip', re.M | re.I|re.U)

        mobjarr = pattern.finditer(aftercontent)
        headlist = []
        preposlist = []
        fontcolor=""
        for mobj in mobjarr:
            amarr = mobj.group('value')
            curposlist = [mobj.start(),mobj.start()+len(amarr)]

            #用表头内容数组替换
            if self.tablesattrobj.headfontcolor !="":
                fontcolor = self.tablesattrobj.headfontcolor
            #先去掉首尾空格，避免首尾有空格无法去掉回车换行符
            amarr = amarr.strip()
            amarr = amarr.strip('\r')
            amarr = amarr.strip('\n')
            amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现
            if amarr == '':
                continue
            amarr = self.__delNotMustStrFromContent(amarr)
            fontcolor = fontcolor.replace('{}','{'+ amarr+'}',1)
            #当表头没有合并单元格时，不能用cellcolor，否则表头不对齐
            #fontcolor = self.commonstr % {
            #    'sphinxstyletheadfamily':'',
            #    'content': self.__delNotMustStrFromContent(amarr)
            #}
            if len(preposlist) > 0:
                headlist.append(headcontent[preposlist[1]:curposlist[0]])
            else:
                headlist.append(headcontent[0:curposlist[0]])
            headlist.append(fontcolor)
            preposlist = curposlist

        headlist.append(headcontent[preposlist[1]:len(headcontent)])  #把最后一个字符串加上
        headcontent = ''
        for prelist in headlist:
            headcontent = headcontent + prelist + '\n'
        newcontent = content[0:posarr[0]]+r'\rowcolor'+headtype+'\n'+headcontent+content[posarr[1]:len(content)]
        return newcontent
        """
        
    def __ModifySecondTableHeadForLongTable(self,headcontent):
        #如果表头没有渲染过，对于只渲染第一列的长表格，分页的表格不能有重复的表头内容，需要将其删除
        startpos = headcontent.find(r"\hline")
        newhead = headcontent[0:startpos] + "\\hline\n\\endhead" #因为前面删除了行结束符，最后再添加上
        return newhead
    
    def __GetTableRowCount(self,tablecontent):
        '''
        得到传进来的表格行数量
        统计原则：latex以“\\”+回车换行为表格的一行，因此以"\\CRLF"或者“\\LF”作为一行，
        正则表达式统计"\\CRLF"或者“\\LF”的数量作为行的数量。
        :param tablecontent: 表格内容，必须去除了latex表格的前面参数部分，否则得到的第一行也包含其内容
        :return: 返回行的数量和每一行内容的列表，如果找不到的话，返回0和None。
        '''
        #以下正则表达式还可以解析出表格的每一行内容
        #在末尾补充一个换行符，避免如果最后一行以“\\”结尾，漏掉最后一行。
        tablecontent = tablecontent+'\n'
        searchstr = "[\s\S]*?(?<count>\\\\[\r|\r\n|\n|\n\r])[\s\S]*?"
        pattern = re.compile(searchstr,re.I | re.M |re.U)
        linelst =  pattern.findall(tablecontent)
        if linelst is not None:
            return len(linelst),linelst
        
        return 0,None
        
        
    def __ModifyVerticalLongTable(self, singletablecontent,tablestyle_dict,columnwidthlst):
        #修改长表格的第一列
        #查找长表格的表头，如果是通过sphinx指令设置的长表格，表头分为三个部分，
        #第一个表头以第一个\hline开始，以\endfirsthead结束
        #第二个表头，以\endfirsthead后续的内容开始，\endhead结束
        #下页继续的说明，以\endhead后续内容开始，以\endlastfoot结束，该部分内容不做处理
        #以\endlastfoot后续内容的第一部分即第一列的内容，需要添加背景色和字体颜色。后续的\hline按正常处理即可
        #先将所有表头内容找出来
        isCusHead = tablestyle_dict['isCusHead']
        headstartpos = singletablecontent.find(r"\hline")
        headendpos = singletablecontent.find(r"\endfirsthead")
        fcontent = singletablecontent[0:headstartpos] #字符串起始内容
        fhead = singletablecontent[headstartpos:headendpos+len(r"\endfirsthead")] #第一个表头内容
        #修改第一个表头内容
        if not isCusHead: #如果表头已经渲染过了，则不再对第一个单元格再渲染一边。
            fhead = self.__ModifyTableByLine(fhead,True,True,tablestyle_dict,columnwidthlst)
        
        #获得第2个表头的原始内容
        headstartpos = singletablecontent.find(r"\endhead")
        shead = singletablecontent[headendpos+len(r"\endfirsthead"):headstartpos+len(r"\endhead")] #第2个表头的内容
        #修改第2个表头内容
        if not isCusHead: #如果表头已经渲染过了，则不再对第一个单元格再渲染一遍。
            shead = self.__ModifySecondTableHeadForLongTable(shead)
        #获得第3部分内容
        headendpos = singletablecontent.find(r"\endlastfoot")
        thead = singletablecontent[headstartpos+len(r"\endhead"):headendpos+len(r"\endlastfoot")] #第3部分表头内容
        othercontent = singletablecontent[headendpos+len(r"\endlastfoot"):len(singletablecontent)]  #其它表格内容
        
        #因为第3部分后续紧跟的内容即为下一行的内容，为了统一处理，在前面固定添加行元素，再修改完成后，再将其删除
        othercontent = "\\hline\n" + othercontent
        #修改表格内容部分
        othercontent = self.__ModifyTableByLine(othercontent,True,True,tablestyle_dict,columnwidthlst)
        #修改完成后，去掉固定添加的内容
        othercontent = othercontent[len("\\hline\n"):len(othercontent)]
        return fcontent + fhead + shead + thead + othercontent

    def __RenderTableHeadByLine(self, linecontent, lineindex, isLongtable, columnwidthlst, jsonmultirow, tablestyle_dict, contentdict):
        '''
        该函数暂时未用
        用列表的方式逐个拆解每个单元格内容，列表格式为[['cell content','分割符号&或者\\'],[],[]]
        :return: 
        '''
        # 内容的起始位置
        searchstr = r"(\\hline|\\cline|\\sphinxcline)"
        match = re.match(searchstr,linecontent)
        if match is None:
            return linecontent
        if match.group() == r"\hline":
            startpos = match.end()
            newtablehead = "\\hline\n"
        else:
            startpos = 0
            newtablehead = ""

        # 得到每个单元格的内容
        colindex = 0  # 是否第一个单元格的索引
        lineindex = 1  # 行索引，为了支持多行合并，默认起始在第一行
        # multirowindex：第几个multirow单元格的索引号，用于作为conf.json中multirowcolumn字段的索引值，
        # 取出当前multirow单元格真实所在列
        # 得到该列的列宽，填写到multirow指令里面，实现multirow指令中的内容自动换行
        multirowindex = 0
        lineflag = 0
        
        while startpos != -1:
            multirowflag = False  # 是否是multirow处理，是的话__ModifyCellContentForMergeCell返回true，默认返回false
            lineflag, startpos, cellcontent = self.__FindCellContentForHead(linecontent, startpos)
            if cellcontent is not None:
                factcolcount = self.__GetFactColIndex(cellcontent[0])
                colindex = colindex + factcolcount  # 当前列号
                if r"\sphinxtablestrut" in cellcontent[0]:
                    # 需要将multirow的内容copy到该单元格
                    self.__AdjustMultirowContent(cellcontent, contentdict, lineindex,colindex,columnwidthlst)
                else:

                    multirowflag = self.__ModifyCellContentForMergeCell(cellcontent,
                                                                        tablestyle_dict,
                                                                        isLongtable,
                                                                        contentdict,
                                                                        columnwidthlst,
                                                                        jsonmultirow,
                                                                        colindex,
                                                                        lineindex,
                                                                        multirowindex)
                newtablehead = newtablehead + cellcontent[0] + "\n" + cellcontent[1] + "\n"

            if multirowflag is True:
                multirowindex = multirowindex + 1
            lineindex = lineindex + lineflag  # 计算新的行数
            if lineflag:
                # 如果开启了新行，则列索引从0开始计算
                colindex = 0

        # 组成新的表头内容
        #newtablehead = newtablehead + "\\hline\n"
        return newtablehead
    
    def __ModifyLineContent(self,linecontent,isvertical,islongtable,lineindex,columnwidthlst,colno,maxcolno,tablestyle_dict,rowdict):
        '''
        修改行内容，当colno为-1时，则修改整行内容，否则根据colno的值修改单元格
        :param linecontent: 该行的内容，以行结束标志“\\”结束。
        :param columnwidthlst: 保存的每一列的列宽内容，根据tablecolumns参数解析出来
                               如果实现自动换行，必须用tablecolumns这个参数，不能用widths参数
        :param colno: 要修改的列号，如果为-1，则修改整行内容。否则根据colno的值修改单元格。
                      如果该行有合并列单元格，则colno为合并单元格后的列号。
        :param maxcolno: 将合并单元格拆分后，得到的最大的列号，该列号为了取得列宽，使multirow内容自动换行。
        :param rowdict: 传出参数，如果是multirow的合并单元格，则返回单元格的内容
        :return: 返回修改后的行的内容。
        '''
        if isvertical is False:
            #修改行，暂时不支持修改任意行，表头的渲染由其他函数完成
            #按表头内容渲染
            
            jsonmultirow = []  # 保存需要设置自动换行的合并单元格的列号。
            if __dictIsHasKey__(tablestyle_dict, 'multirowcolumn'):
                jsonmultirow = tablestyle_dict['multirowcolumn']
            newlinecontent = self.__RenderTableHeadByLine(linecontent,
                                                          lineindex,
                                                          islongtable,
                                                          columnwidthlst,
                                                          jsonmultirow,
                                                          tablestyle_dict,
                                                          rowdict)
            return newlinecontent
        
        #下面的内容修改列，即只修改行的第一个单元格
        linecontentstr=linecontent
        #startpos = linecontent.find('\\hline')
        #lineprefix = ""
        #if startpos != -1:
        #    linecontentstr = linecontent[startpos + len("\\hline"):len(linecontent)]
        #    lineprefix = linecontent[0:startpos+len("\\hline")]
        #得到行内容和前缀
        lineprefix = ""
        searchstr=r"\\hline|\\cline{\S+?}|\\sphinxcline{\S+?}\\sphinxfixclines{\d+?}"
        match = re.search(searchstr,linecontent,re.I|re.U|re.M)
        if match is not None:
            linecontentstr = linecontent[match.end():len(linecontent)]
            lineprefix = linecontent[0:match.start()+ len(match.group())]
            
        othercontent = ""
        newlinecontent = ""
        
        #暂时仅支持修改第一列
        lineflag,nextpos,cellcontent = self.__FindCellContentForHead(linecontentstr,0)
        if (cellcontent is None) or (r"\cellcolor" in cellcontent[0]):
            #说明该单元格已经修改过了，直接返回
            return linecontent

        othercontent = linecontentstr[nextpos:len(linecontentstr)]
        
        if r"\sphinxtablestrut" in cellcontent[0]:
            # 需要将multirow的内容copy到该单元格
            self.__AdjustMultirowContent(cellcontent, rowdict, lineindex,maxcolno,columnwidthlst)
        else:
            jsonmultirow =[1]
            cellstr = cellcontent[0]  # 取得单元格内容
            
            if r'\sphinxmultirow' in cellstr:
                cellcontent[0] = self.__ModifySphinxMultirow(cellstr,
                                                             rowdict,
                                                             columnwidthlst,
                                                             jsonmultirow,
                                                             lineindex,
                                                             maxcolno,
                                                             0,
                                                             tablestyle_dict,
                                                             isvertical)
            else:
                coluwidth = "*"
                if len(columnwidthlst) > 0:
                    coluwidth = columnwidthlst[0]
                # 查找\sphinxAtStartPar位置
                pos = cellstr.find(r'\sphinxstyletheadfamily')
                if pos != -1:
                    contentstr = cellstr[pos + len(r'\sphinxstyletheadfamily'):len(cellstr)]
                    
                    fullcontentstr = self.commonstr % {
                           'sphinxstyletheadfamily': "\\sphinxstyletheadfamily",
                           'content': self.__delNotMustStrFromContent(contentstr)
                    }
                    #fullcontentstr = self.multirowstr % {
                    #    'count': 1,
                    #    'coluwidth': coluwidth,
                    #    'sphinxstyletheadfamily': r"\sphinxstyletheadfamily",
                    #    'content': self.__delNotMustStrFromContent(contentstr)
                    #}
                    #fullcontentstr = self.normalstr % {
                    #    'flag': '|l|',
                    #    'sphinxstyletheadfamily': "\\sphinxstyletheadfamily",
                    #    'content': self.__delNotMustStrFromContent(contentstr)
                    #}
                    
                    cellcontent[0] = fullcontentstr
                else:
                    fullcontentstr = self.commonstr % {
                        'sphinxstyletheadfamily': "\\sphinxstyletheadfamily",
                        'content': self.__delNotMustStrFromContent(cellstr)
                    }
                    #fullcontentstr = self.multirowstr % {
                    #    'count': 1,
                    #    'coluwidth': coluwidth,
                    #    'sphinxstyletheadfamily': r"\sphinxstyletheadfamily",
                    #    'content': self.__delNotMustStrFromContent(cellstr)
                    #}
                    #fullcontentstr = self.normalstr % {
                    #    'flag': '|l|',
                    #    'sphinxstyletheadfamily': "\\sphinxstyletheadfamily",
                    #    'content': self.__delNotMustStrFromContent(cellstr)
                    #}
                    
                    cellcontent[0] = fullcontentstr

        newlinecontent = lineprefix +"\n" + cellcontent[0] + "\n" + cellcontent[1] + "\n" + othercontent
        return newlinecontent


    def __ModifyTableByLine(self,tablecontent,isvertical,islongtable,tablestyle_dict,columnwidthlst):
        '''
        按行渲染每一个单元格。根据正则表达式解析出每一行的内容，
        然后再解析这一行中每一个单元格的内容，根据条件判断是否需要渲染该单元格
        :param tablecontent: 
        表格内容，对于长表格去除表格头的表格，对于普通表格，整个表格内容
        :param isvertical:
        是否修改列还是修改行
        :return: 返回修改后的内容
        '''
        #找出每一行的内容
        
        searchstr = r'(\\hline|\\cline{\S+}|\\sphinxcline{\S+})(?P<content>[\s\S]*?)(?=\\hline|\\cline{\S+}|\\sphinxcline{\S+})'
        pattern = re.compile(searchstr,re.M | re.I|re.U)
        # 得到每一行的内容，以\hline或者\cline开头，以\\结尾
        linelst = pattern.findall(tablecontent)
        if len(linelst) ==0:
            return tablecontent
        
        tableprefix = ""  #得到latex表格的前缀参数部分，除表格内容外的部分
        tablepostfix = "" #得到latex表格的后缀参数部分，出表格内容外的部分
        prepos = tablecontent.find(''.join(linelst[0]))
        if prepos > 0:
            tableprefix = tablecontent[0:prepos]
        lastline = ''.join(linelst[-1])
        postpos = tablecontent.find(lastline)
        if (postpos+len(lastline)) < len(tablecontent):
            tablepostfix = tablecontent[postpos+len(lastline):len(tablecontent)]

        newtablecontent = ""
        rowdict = {} #保存合并单元格的字典，用于multirow
        for lineindex in range(0,len(linelst)):
            line =''.join(linelst[lineindex])
            newline= self.__ModifyLineContent(line,isvertical,islongtable,lineindex+1,columnwidthlst,1,1,tablestyle_dict,rowdict)
            newtablecontent = newtablecontent + newline
           
        newtablecontent = tableprefix + newtablecontent+tablepostfix
        return newtablecontent

    #渲染树型表格的第一列
    def __ModifyVerticalTable(self, singletablecontent, tablestyle_dict, columnwidthlst):
        
        #找出每一行的内容
        searchstr = r'(\\hline|\\cline{\S+}|\\sphinxcline{\S+})(?P<content>[\s\S]*?)(?=\\hline|\\cline{\S+}|\\sphinxcline{\S+})'
        pattern = re.compile(searchstr,re.M | re.I|re.U)
        matchiter = pattern.finditer(singletablecontent)
        posarr=[]  #保存位置，便于组合
        i = 0
        for m in matchiter:

            posarr.append([])
            posarr[i] = m.span()

            if i ==0:
                newcontent = singletablecontent[0:posarr[i][0]]+m.group(1)
            else:
                newcontent = newcontent+singletablecontent[posarr[i-1][1]:posarr[i][0]]+m.group(1)

            cellcontent = m.group('content')
            #匹配到的第一个即为表头内容
            #将第一个单元格内容渲染成蓝底白字
            firstcellcontent = self.__ModifyFirstColumnType(cellcontent)
            newcontent += firstcellcontent
            i+=1
        newcontent += singletablecontent[posarr[i-1][1]:len(singletablecontent)]
        return newcontent

    #渲染第一个单元格内容
    def __ModifyFirstColumnType(self,cellcontent):

        new_cellcontent = ""

        if r'\sphinxstyletheadfamily' in cellcontent:

            searchstr = r'(?<=\\sphinxstyletheadfamily)(?P<value>[\s\S]*?)(?=(\\unskip|&)|\\\\)'

            aftercontent = cellcontent.strip()
            aftercontent = aftercontent.strip('\r')
            aftercontent = aftercontent.strip('\n')
            aftercontent = aftercontent.strip()
            mobj = re.search(searchstr, aftercontent, re.M|re.I|re.U )  #匹配到的第一个既是需要修改的内容

            #修改字体颜色
            amarr = mobj.group('value')
            posarr = mobj.span()
            new_cellcontent = aftercontent[0:posarr[0]]+'\n'+r'\cellcolor'+self.tablesattrobj.headtype
            #用表头内容数组替换
            fontcolor = self.tablesattrobj.headfontcolor
            #先去掉首尾空格，避免首尾有空格无法去掉回车换行符
            amarr = amarr.strip()
            amarr = amarr.strip('\r')
            amarr = amarr.strip('\n')
            amarr = amarr.strip()  #再去掉首尾空格，避免多余的空格出现
            #if amarr == '':
            #    continue
            if (r'\textbf' or r'\textcolor') in amarr:
                return cellcontent
            fontcolor = fontcolor.replace('{}','{'+ amarr+'}',1)
            new_cellcontent +=r'{'+fontcolor + '}\n' + aftercontent[posarr[1]:len(aftercontent)]
        else:
            aftercontent = cellcontent.replace(r'\\','&',1)
            #去掉首尾空格和换行符
            aftercontent = aftercontent.strip()
            aftercontent = aftercontent.strip('\r')
            aftercontent = aftercontent.strip('\n')
            aftercontent = aftercontent.strip()

            tmplist = re.split(r'&',aftercontent)

            preposlist = 0
            #只对第一个做修改
            onelist = tmplist[0]
            #先去掉首尾空格，避免首尾有空格无法去掉回车换行符
            onelist = onelist.strip()
            onelist = onelist.strip('\r')
            onelist = onelist.strip('\n')
            onelist = onelist.strip()  #再去掉首尾空格，避免多余的空格出现
            #if onelist == '':
            #        continue
            if (r'\textbf' or r'\textcolor') in onelist:
                return cellcontent
            new_cellcontent = self.__ModifyFirstColumnContentPart(onelist)

            for i in range(1,len(tmplist)):
                if len(tmplist[i])>0:
                    new_cellcontent += '&' +tmplist[i]
            new_cellcontent+=r'\\' #将最后被替换掉的\\再加上

        return new_cellcontent + '\n'

    def __ModifyFirstColumnContentPart(self,firstcontent):
        #查找firtcontent的内容部分，因为如果存在合并单元格，第一列的内容起始部分不一定是内容，可能是latex的其它标签。
        #sphinx生成的latex表格都以“\sphinxAtStartPar”标识开始
        startpos = firstcontent.find(r"\sphinxAtStartPar")
        firstpart = ""
        if startpos==-1:
            contentpart = firstcontent
        else:
            firstpart = firstcontent[0:startpos]
            contentpart=firstcontent[startpos:len(firstcontent)]
        fontcolor = self.tablesattrobj.headfontcolor
        # 先去掉首尾空格，避免首尾有空格无法去掉回车换行符
        new_cellcontent = '\n'+r'\cellcolor'+self.tablesattrobj.headtype+r'{'+fontcolor.replace('{}','{'+ contentpart+'}',1)+r'}'+'\n'
        return firstpart+new_cellcontent
    
    def __ModifyReplaceContent(self):
        '''
        修改replace内容。为了删除sphinx在替换时多加的空格，在替换时，每个替换内容前后都加了"@@"。
        对于前后都加了“@@”的内容，删除前后的"@@"，同时删除前后的空格。
        :return: 
        '''
        searchstr = r"@@(?P<content>[\s\S]+?)@@"
        pattern = re.compile(searchstr)
        match = pattern.search(self.content)
        if match is None:
            return None
        str = match.group()
        replace = match.group('content')
        
        prepos = match.start()
        endpos = match.end()
        replaceed = str
        if prepos > 1:
            #判断前一个字符是否为空格
            prechar = self.content[prepos-1]
            if prechar==" ":
                replaceed = " "+replaceed
                
        endchar = self.content[endpos]
        if endchar == " ":
            replaceed = replaceed + " "
    
        #进行替换
        self.content = self.content.replace(replaceed,replace)

        return self.content
        
        
    def ModifyReplaceContent(self):
        
        while True:
            '''
            可能会有多个需要替换，直到找不到为止
            '''
            result = self.__ModifyReplaceContent()
            if result is None:
                break

class clsSensitiveWords:
    
    def __init__(self,entext):
        self.passkey="Hello123"
        self.entext = entext
        self.jsonwords = GetSensitiveword()
        
    def GetFullSensitiveWords(self):
        senwords = "|".join(self.jsonwords)
        detext = ""
        if self.entext !="":
            command="echo '" + self.entext +"'| openssl aes-128-cbc -d -base64 -pbkdf2 -k " + self.passkey
            process = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
            with process.stdout:
                for line in iter(process.stdout.readline, b''):
                    detext = detext + line.decode().strip()
            senwords = senwords + '|' + detext
            
        return senwords

class clsCheckDocCopyrightYear:
    '''
    检查文档是否符合规范。
    2022年07月12日更新
    -------------------
    增加对当前年份的检查，如果年份不符合要求，自动修改conf.py和copyright.rst文档中的年份。
    '''
    def __init__(self,app,copyrightfile):
        self.app=app
        self.confdir = app.confdir
        self.language = app.config.language
        self.confright = app.config.copyright
        if (app.config.latex_elements is not None) and __dictIsHasKey__(app.config.latex_elements,'preamble'):
            self.latex_preamble = app.config.latex_elements['preamble']
        else:
            self.latex_preamble = ""
            
        if len(copyrightfile) > 0:
            self.copyrightfile = copyrightfile
        elif hasattr(app.config,"copyfile_path") and len(app.config.copyfile_path)>0:
            self.copyrightfile = app.config.copyfile_path
        else:
            self.copyrightfile = ""
    
    def __FindCopyrightFile(self):
        '''
        #如果copyrightfile为空，则查找copyrightfile文件
        '''
        if self.copyrightfile =="":

            copyrightpre = "/copyright/copyright"
            filepath = self.app.srcdir + copyrightpre + ".rst"
            
            if os.path.isfile(filepath):
                return filepath
            
            if self.language=="zh_CN":
                filepath = self.app.srcdir+ copyrightpre + "_zh.rst"
            else:
                filepath = self.app.srcdir+ copyrightpre +"_en.rst"
                
            if os.path.isfile(filepath):
                return filepath
            else:
                return ""
        else:
            filepath = self.confdir + "/" + self.copyrightfile
            #转为绝对路径
            filepath = os.path.abspath(filepath)
            if os.path.isfile(filepath):
                return filepath
            else:
                return ""

    def __IsModifyfootYear(self,year):
        '''
        查找lfoot设置版权的语句年份是否需要修改
        :return: True - 需要修改年份
                 False - 不需要修改年份
        '''
        
        ismodify = False
        searchstr = r'\\lfoot{[ ]*Copyright [\s\S]*?}'
        pattern = re.compile(searchstr,re.I|re.U)
        matchiter = pattern.finditer(self.latex_preamble)
        if matchiter is None:
            return ismodify
        
        for m in matchiter:
            lfoot = m.group()
            if year in lfoot:
                #如果当前年份已在脚注中，则认为年份已修改，则不再修改。
                break
            else:
                # 如果当前年份不在脚注中，则认为年份还未修改，需要修改。
                ismodify = True
                break
        return ismodify
    
    def __ModifyYearofConfforDetail(self,year,ispremble=False):
        '''
        修改conf.py中的年份
        :param ispremble: 是否只修改ispremble中脚注的年份。
                          True-只修改脚注的年份；False-同时修改copyright和脚注的年份。
        '''
        #打开文件
        conffile = self.confdir +"/conf.py"
        fo = codecs.open(conffile, "r+", encoding='utf-8')
        textcontent = fo.read()
        fo.close()
        
        if ispremble is False:
            #修改copyright的年份
            searchstr = r"[0-9]{4}"
            matchobj = re.search(searchstr,self.confright)
            if matchobj is not None:
                copyright = re.sub(searchstr,year,self.confright)
                textcontent=textcontent.replace(self.confright,copyright,1)
                #为了看到最终效果，同时修改已经读取的变量，因为调用该函数前，conf.py的配置已经被sphinx读取
                self.app.config.copyright=copyright

        #开始替换脚注中的年份
        #得到脚注中带有年份的声明
        searchstr = r'\\lfoot{[ ]*Copyright [\s\S]*?}'
        pattern = re.compile(searchstr, re.I | re.U)
        matchiter = pattern.finditer(self.latex_preamble)
        if matchiter is not None:
            for m in matchiter:
                lfoot = m.group()
                searchstr= r"[0-9]{4}"
                matchobj = re.search(searchstr,lfoot)
                if matchobj is not None:
                    newlfoot = re.sub(searchstr,year,lfoot)
                    textcontent=textcontent.replace(lfoot,newlfoot)
                    #两个脚注相同，因此只替换一次即可
                    # 为了看到最终效果，同时修改已经读取的变量，因为调用该函数前，conf.py的配置已经被sphinx读取
                    self.app.config.latex_elements['preamble'] = self.latex_preamble.replace(lfoot,newlfoot)
                    break 
        #将替换后的内容重新写入文件
        fw = codecs.open(conffile, "w+", encoding='utf-8')
        fw.write(textcontent)
        fw.close()
        
    def __ModifyYearofCopyright(self,year):
        
        #得到copyright绝对文件名
        filepath = self.__FindCopyrightFile()
        if filepath == "":
            return

        fo = codecs.open(filepath, "r+", encoding='utf-8')
        textcontent = fo.read()
        fo.close()
        #新的版权年份
        newyear = '© ' + year
        #if newyear in textcontent:
        #    return
        isupdate = False
        #避免替换错误，循环找到最后一个进行替换
        searchstr = "©( )*([0-9]{4})"
        pattern = re.compile(searchstr, re.I | re.U)
        matchiter = pattern.finditer(textcontent)
        for m in matchiter:
            oldyear = m.group()
            if oldyear == newyear:
                continue
            textcontent = textcontent.replace(oldyear,newyear,1)
            isupdate = True
            
        if isupdate:
            fw = codecs.open(filepath, "w+", encoding='utf-8')
            fw.write(textcontent)
            fw.close()
        #pattern = re.compile(searchstr, re.I | re.U)
        #matchlst = pattern.findall(textcontent)
        #if len(matchlst)==0:
        #    #如果没有找到四数字的年份，直接返回
        #    return
        
        #i=0
        #content=""
        #matchiter = pattern.finditer(textcontent)
        #for m in matchiter:
        #    if (i+1)==len(matchlst):
        #        #对最后一个年份进行替换
        #        content = textcontent[0:m.start()] + year + textcontent[m.end():len(textcontent)]
        #        break
        #    else:
        #        i = i+1

        #fw = codecs.open(filepath, "w+", encoding='utf-8')
        #fw.write(content)
        #fw.close
        
    def __ModifyYearofConf(self,year):
        #开始修改conf.py中的年份
        if (year not in self.confright) and (self.confright != ""):
            #修改conf.py中的年份
            self.__ModifyYearofConfforDetail(year)
            return
        #开始修改conf.py中脚注的年份
        if (self.latex_preamble!="") and (self.__IsModifyfootYear(year) is True):
            #需要修改年份，走到这一步conf.py的copyright年份肯定修改过的，因此只修改foot中的年份即可
            self.__ModifyYearofConfforDetail(year,True)
            return
        
    def CheckYearIsRight(self):
        '''
        检查年份是否正确
        '''
        #得到系统当前年份
        today = dt.date.today()
        year = today.strftime('%Y')
        #开始检查conf.py中的年份
        self.__ModifyYearofConf(year)
        #开始检查copyright文件中的年份
        self.__ModifyYearofCopyright(year)
        
    def __AddAdSymbolForReplaceZh(self,content):
        '''
        解析是否需要替换，需要的话就进行替换。
        替换的条件：1.被替换字符串全部为中文，2。被替换字符的首尾有一个不为英文
        :param content: 要替换的内容
        :return: 已替换或者未替换的字符串
        '''
        #用正则表达式解析出要替换的内容
        searchstr = r"[\s\S]+ replace:: (?P<replace>[\s\S]+)"
        match = re.search(searchstr,content)
        if match is None:
            return content
        replacestr = match.group('replace').strip('\r')
        #取得字符串的首和尾字符
        #prochr = ord(replacestr[0])
        #print(prochr)
        #epichr = ord(replacestr[-1])
        #print(epichr)
        #if ((64 < prochr and prochr<91) or \
        #    (96 < prochr and prochr<123)) and \
        #    ((64 < epichr and epichr < 91) or \
        #     (96 < epichr and epichr < 123)):
        #    #首尾都为英文则无需替换空格，直接返回
        #    return content
        newreplacestr = "@@"+replacestr+"@@"
        return content.replace(replacestr,newreplacestr)
        
    def __ParseReplaceList(self,contentlst):
        '''
        为中文替换词前后添加@@符号
        :param contentlst: 需要替换的list内容
        :return: 新生成列表，如果无需替换，返回None。
        '''
        if contentlst is None or len(contentlst)==0:
            return None
        
        for i in range(0,len(contentlst)):
            content = contentlst[i]
            if content=="":
                continue
            #进行替换
            newcontent = self.__AddAdSymbolForReplaceZh(content)
            contentlst[i] = newcontent
            
        return contentlst
            
    def CheckReplaceContent(self):
        '''
        该函数用来判断conf.py文件中是否包含rst_epilog或者rst_prolog的配置
        如果有的话，修改这两个参数中的配置在需替换的名词前后都加@@符号，以方便在latex文件中删除前后的空格。
        该修改为了解决sphinx对中文的替换前后都加空格的bug。
        :return: 
        '''
        epilogstr = self.app.config.rst_epilog
        prologstr = self.app.config.rst_prolog
        
        if self.app.config.language != "zh_CN":
            '''
            如果不是中文文档，则直接返回。
            '''
            return
        
        if prologstr is not None:
            # 为rst_prolog的中文替换前后添加@@符号
            prologlst = prologstr.split('\n')
            newlst = self.__ParseReplaceList(prologlst)
            if newlst is not None:
                self.app.config.rst_prolog = '\n'.join(newlst)
            
        if epilogstr is not None:
            #为rst_epilog的中文替换前后添加@@符号
            epiloglst = epilogstr.split('\n')
            newlst = self.__ParseReplaceList(epiloglst)
            if newlst is not None:
                self.app.config.rst_epilog = '\n'.join(newlst)
            


class clsCheckHtmlRule:
    '''
    检查html配置是否符合规范
    '''
    def __init__(self,app):
        self.app=app
        #错误描述模板
        #配置参数错误
        self.configparam_error="Parameter %(param)s configuration error!"
        self.correctvalue="The correct value is %(value)s!"
        self.config_error=self.configparam_error + self.correctvalue+"\n"
        #工程名称配置不一致
        self.inconsistent_error="The project names configured by conf.py and %(rootdoc)s.rst are inconsistent!\n"
        #缺少custom.css文件
        self.csslack_error="custom.css file not found!\n"
        self.config_detail = "For detailed configuration, see wiki: pageId=46137795\n"

    def __CheckCommonParamForHtml(self, theme):
        '''
        检查相关配置是否正确
        :param theme:为主题文件，如果和配置的不一致返回错误。默认为sphinx_rtd_theme。
        '''
        error = ""
        if self.app.config.html_theme != theme:
            error = self.config_error % {
                'param': 'html_theme',
                'value': theme
            }
        if self.app.config.html_copy_source is True:
            error = error + self.config_error % {
                'param': 'html_copy_source',
                'value': 'False'
            }
        # 判断配置的css文件是否存在
        if len(self.app.config.html_css_files) == 0:
            error = error + self.configparam_error % {
                'param': 'html_css_files',
            } + \
            "Please configure a css file, such as custom.css."
            "The css file needs to be found under the path specified by html_static_path!"
        if len(self.app.config.html_static_path) == 0:
            error = error + self.configparam_error % {
                'param': 'html_static_path',
            } + \
            "Please configure the path of the file specified by html_css_files."
        return error

    def CheckProjectNameConsistent(self):
        '''
        检查conf.py中project配置工程名称和主索引文件index.rst中的标题是否一致
        两者不一致返回错误。
        该接口必须在builder-inited事件之后调用，否则得不到index.rst的标题，无法比较。
        '''
        #为了兼容性获取config对象的属性，没有的属性不做比较
        cfgattrlst=dir(self.app.config)
        #主索引文件名
        if "master_doc" in cfgattrlst:
            indexname = self.app.config.master_doc
        elif "root_doc" in cfgattrlst:
            indexname = self.app.config.root_doc
        else:
            return ""
        
        #得到主索引文件的标题
        indextitle = self.app.env.titles[indexname][0]
        #得到配置的工程名称
        projectname = self.app.config.project
        if indextitle != projectname:
            error = self.inconsistent_error % {
                'rootdoc': indexname
            }
            return error
        else:
            return ""

    def CheckCommConfigHtmlRule(self, theme):
        '''
        检查生成html前参数配置是否符合规范
        生成html的规范参见黄区wiki：https://wiki.cambricon.com/pages/viewpage.action?pageId=46137795
        '''
        # 开始检查参数配置
        error = self.__CheckCommonParamForHtml(theme)
        return error
    
def CheckProjectNameIsConsistent(app):
    '''
    检查conf.py中配置的工程名称与主索引文件的标题是否一致
    '''
    checkDocobj = clsCheckHtmlRule(app)
    return checkDocobj.CheckProjectNameConsistent()

def CheckHtmlConfigIsCorrect(app,theme='sphinx_rtd_theme'):
    '''
    检查html配置参数是否正确
    '''
    checkDocobj = clsCheckHtmlRule(app)
    return checkDocobj.CheckCommConfigHtmlRule(theme)
        
def CheckCurrentYearAndModify(app,copyrightfile=""):
    '''
    检查conf.py和copyright中的年份是否为当前年份，如果不是当前年份则进行修改。
    Conf.py主要检查latex_elements和copyright中的年份是否为当前年份
    copyright主要检查copyright.rst文件最后一句的年份是否为当前系统年份。
    :param app:
    :param copyrightfile: 相对于conf.py的相对路径。
    '''
    checkDocobj = clsCheckDocCopyrightYear(app,copyrightfile)
    #检查年份是否需要修改
    checkDocobj.CheckYearIsRight()
    #if (app.builder is not None) and \
    #    (app.builder.name=='latex' or app.builder.name=='latexpdf'):
    #    checkDocobj.CheckReplaceContent()
    
def __returnCfgDate(app):
    '''
    返回conf.py中配置的today的日期。如果today为空，则返回当前日期。
    '''
    if app.config.today != "":
        return app.config.today
    today = dt.date.today()
    if app.config.language =="zh_CN":
        return today.strftime('%Y年%m月%d日')
    else:
        return today.strftime('%b %d, %Y')

def CheckUpdateHistory(app,docname,source):
    '''
    检查是否需要添加更新历史，在source_read_handler这个事件里调用。
    为了提高效率，不写在类里，减少创建和销毁对象的消耗。
    :param app:
    :param docname:
    :param source: 源文件内容
    '''
    #判断文档中是否包含更新历史
    # 默认为英文
    title = "Update History"
    update="Date:"
    changes="Changes:"
    language = app.config.language
    if language == "zh_CN":
        title = "更新历史"
        update="更新时间："
        changes="更新内容："
    index = source.find(title)
    if index==-1:
        return source

    #说明该文档中包含更新历史，判断是否为更新历史标题
    precontent = source[0:index]
    leftcontent = source[index:len(source)]
    #按行读取剩余内容，理论上第2行为标题的下标，如果不是，则认为这个更新历史不是标题，直接忽略。
    linelst = leftcontent.splitlines(True)
    flagline = linelst[1]
    search = r"([\x21-\x2f|\x3a-\x40|\x5b-\x60|\x7b-\x7d])\1+(?=[\r|\n|\r\n])"
    matchobj=re.match(search,flagline)
    if matchobj is None:
        return source
    #说明是更新历史章节，再比较版本号是否一致
    search = r"\* \*\*V(?P<version>[0-9]{1,2}\.[0-9]{1,2}\.[0-9]{1,2})\*\*"
    matchobj=re.search(search,leftcontent)
    if matchobj is None:
        return source
    version=matchobj.group('version')
    if version==app.config.version:
        return source
    #如果版本不一致，自动添加新的版本内容
    dateformat = __returnCfgDate(app)
    updatehistory="\n* **V%(version)s**\n\n" \
                  "  **%(date)s** %(dateformat)s\n\n" \
                  "  **%(changes)s**\n\n" \
                  "  - 无内容更新。\n"
    
    upcontent= updatehistory % {
        'version':app.config.version,
        'date':update,
        'dateformat':dateformat,
        'changes':changes
        }
    #拆分列表
    titlestr =''.join(linelst[0:2])
    otherstr =''.join(linelst[2:len(linelst)+1])
    newcontent = precontent + titlestr + upcontent + otherstr
    #将新内容写入文档
    filepath = app.project.doc2path(docname)
    fw = codecs.open(filepath, "w+", encoding='utf-8')
    fw.write(newcontent)
    fw.close()
    return newcontent


def __exe_command_forprint(command):
    """
    执行 shell 命令并实时打印输出
    :param command: shell 命令
    :return: process, exitcode
    """
    process = Popen(command, stdout=PIPE, stderr=STDOUT, shell=True)
    with process.stdout:
        for line in iter(process.stdout.readline, b''):
            print(line.decode().strip())
    exitcode = process.wait()
    return process, exitcode

def CheckSensitiveWordsForLinux(app,docname,entext):
    '''
    该接口对外提供，可在conf.py的source_read_handler处理函数中调用。对sphinx工程中的文件检查敏感词。
    
    **注意：**
    
    因为检查敏感词使用的linux命令，如果敏感词中包含中文，则在不支持中文的linux系统下不能使用该接口，
    否则无法完成敏感词。
       
    :param app: sphinx环境变量，可以获取输出目录、文件后缀等相关信息。
    :param docname: 需检查的文件路径。该路径是基于源目录的相对路径，而且不包含文件扩展名，
                    因此在检查时需要补充上扩展名，并转为绝对路径。
    :param entext: 已加密的密文，如果为空，则只对conf.json中保存的敏感词做检查。
                   如何加密参考GetFullSensitiveWords的实现，加解密保持一致。
    :return: 无返回值。
    '''
    #windows系统请使用工具组开发的sensitive-filter.exe工具
    try:
        if platform.system().lower() == 'windows':
            return

        filepath = app.project.doc2path(docname,True)
        
        #print("Checking file:["+docname+"]")
        sensitivewordsobj = clsSensitiveWords(entext)
        sensitivewords = sensitivewordsobj.GetFullSensitiveWords()
        #print(sensitivewords)
        cmd = "egrep -rni --color=always '"+  sensitivewords + "' " + filepath
        __exe_command_forprint(cmd)
    except Exception as e:
        print("Sensitive words check failed, the system may not support Chinese. "
             "The detailed error information is as follows:")
        print(e)
        return
     
# 打开Makefile文件查找source和build文件夹
def OpenMakefile():
    global source_dir
    global build_dir
    source_dir = ''
    build_dir = ''
    try:
        if platform.system().lower() == 'windows':
            with open('make.bat',"r") as f:
                fstr = f.read()

            #用正则表达式查找source和build文件夹具体路径
            searchstr = r"set *SOURCEDIR *= *(\S+)"
            m = re.search(searchstr, fstr, re.M|re.I|re.U )
            source_dir = m.group(1) #匹配到的第一个即为source所在目录

            searchstr = r"set *BUILDDIR *= *(\S+)"
            m = re.search(searchstr, fstr, re.M|re.I|re.U )
            build_dir = m.group(1) #匹配到的第一个即为build所在目录
        else:
            with open('Makefile',"r") as f:
                fstr = f.read()

            #用正则表达式查找source和build文件夹具体路径
            searchstr = r"SOURCEDIR *= *(\S+)"
            m = re.search(searchstr, fstr, re.M|re.I|re.U )
            source_dir = m.group(1) #匹配到的第一个即为源所在目录

            searchstr = r"BUILDDIR *= *(\S+)"
            m = re.search(searchstr, fstr, re.M|re.I|re.U )
            build_dir = m.group(1) #匹配到的第一个即为源所在目录

    except Exception as e:
        print(e)
        return

def GetLatex_documents():
    global source_dir
    if source_dir == '':
        return
    #得到配置文件conf.py的路径
    if source_dir == '.':
        confdir = './conf.py'
    else:
        confdir = './' + source_dir +'/conf.py'

    conffile = os.path.abspath(confdir)
     #打开conf.py文件
    with codecs.open(conffile,"r+",encoding='utf-8') as f:
            fstr = f.read()
    list = []
    versioninfo = GetVersionInfo(fstr)
    fileprefile = GetFilePreInfo(fstr)
    if versioninfo=="" or fileprefile=="":
    #根据正则表达式，找出latex_documents内容
        searchstr = r"latex_documents *= *\[([\s\S]*?)\]"
        m = re.search(searchstr, fstr, re.M|re.I|re.U )
        latex_documents = m.group(1) #匹配到的第一个即为源所在目录
        #拆分二维数组，兼容多个情况
        list = latex_documents.split(")")
        for i in range(len(list)):
            if IsComment(list[i]):
                list[i]= list[i].split(",")
        list.pop()
    else:
        #创建2维列表
        for i in range(1):
            list.append([])
            for j in range(2):
                list[i].append("")
        list[0][0] = "comment"  #为了兼容文件解析内容，进行补位。
        list[0][1] = '"' + fileprefile + versioninfo + '.tex"'  #为了兼容解析过程，添加双引号。
    return list

#得到版本信息
def GetVersionInfo(fstr):

    if releasever != '':
       return releasever

    versioninfo = ""
    searchstr = r"version *= *(u*)'(.*?)'"
    m = re.search(searchstr, fstr, re.I)
    if m is not None:
        versioninfo = m.group(2) #匹配到的第一个即为源所在目录
    print("version = " + versioninfo)
    return versioninfo

#得到文件前缀信息
def GetFilePreInfo(fstr):
    filepre = ""
    searchstr = r"curfnpre *= *(u*)'(.*?)'"
    m = re.search(searchstr, fstr, re.M|re.I|re.U )
    if m is not None:
        filepre = m.group(2) #匹配到的第一个即为源所在目录
    return filepre

#判断是否为注释行
def IsComment(instr):

    if instr.strip() is None:
        return False

    rule = re.compile('^#.*$')
    if rule.match(instr.strip()) is None:
        return True
    else:
        return False

#根据正则表达式取单引号和双引号中的内容
def getquomarkcontent(strarr):
    #根据正则表达式，找出双引号和单引号中的内容
    searchstr = r"[\"|'](.*?)[\"|']"
    m = re.search(searchstr, strarr, re.M|re.I|re.U )
    if m is None:
        return None
    return m.group(1).strip() #匹配到的第一个即为源所在目录

def GetConfjsondict(app):
    '''
    得到配置文件conf.json的字典
    :param app: 
    :return: 
    '''
    global __load_dict__
    
    
    if app is not None:
        #先判断conf.json路径在配置文件中是否配置
        if hasattr(app.config, "confjson_path") and len(app.config.confjson_path) > 0:
            confpath = os.path.abspath(app.confdir + "/" + app.config.confjson_path)
        else:
            confpath = app.confdir + "/conf.json"
            
        if os.path.exists(confpath):
            __load_dict__ = __openconfjsonfile__(confpath)
        else:
            __load_dict__ = __openconfjsonfile__()
    else:
        __load_dict__ = __openconfjsonfile__()
        
        
def Modifylatex_main(build_dir,latexdocumentslst,app=None,language='zh_CN'):

    global __load_dict__
    global doc_language
    doc_language=language

    if len(__load_dict__) == 0:
        GetConfjsondict(app)


    doclen = len(latexdocumentslst)
    for i in range(0,doclen):
        #得到latex路径
        latexpath = build_dir
        desbkpaperfile= latexpath+'/chapterbkpaper.pdf'
        #copy 背景图到latex路径，背景图必须与该文件在同一个目录下
        if (app is not None) and hasattr(app.config,'chapterbkpaper_image') and (app.config.chapterbkpaper_image!=""):
            #转为绝对路径进行判断，兼容conf.py和parsejson.py不在同一目录的情况
            bkpaperimage = os.path.abspath(app.confdir+"/"+ app.config.chapterbkpaper_image)
            if os.path.isfile (bkpaperimage):
                shutil.copy(bkpaperimage,desbkpaperfile)
        elif os.path.exists('./chapterbkpaper.pdf'):
            shutil.copy('./chapterbkpaper.pdf', latexpath)
        else:
            #取配置文件所在目录下的chapterbkpaper.pdf,将其copy到latex文件夹
            bkpaperimage = os.path.abspath(app.confdir + "/chapterbkpaper.pdf")
            if os.path.isfile(bkpaperimage):
                shutil.copy(bkpaperimage, latexpath)
                
        #得到相对路径
        filename = latexdocumentslst[i][1]
        if  filename is None:
            continue
        texfilepath = latexpath + '/' + filename
        #相对路径转绝对路径
        texfile = os.path.abspath(texfilepath)
        if not os.path.exists(texfile):
            continue
        fo = codecs.open(texfile, "r+",encoding = 'utf-8')
        texcontent = fo.read()
        fo.close()

        #得到修改tex文件的对象
        ModTexobj = clsModifyTex(texcontent)
        ModTexobj.AddPackageToTex()
        ModTexobj.AddOtherTocToTex()
        ModTexobj.AddCustormOptionsToTex()
        ModTexobj.ModifyReplacePackage()
        ModTexobj.ModifyFunctionBackColor()
        ModTexobj.ModifyTablesAttributes()
        #if (app is not None) and \
        #    (app.builder.name=='latex' or app.builder.name=='latexpdf'):
        #    ModTexobj.ModifyReplaceContent()

        fw = codecs.open(texfile, "w+",encoding = 'utf-8')
        fw.write(ModTexobj.content)
        fw.close()
        
def parseargv():
    global source_dir
    global latex_dir
    if len(sys.argv) <= 1:
        return
    for i in range(1,len(sys.argv)):
        param = sys.argv[i]
        if param=='-c':  #解析conf.py所在目录
            source_dir = sys.argv[i+1] #保存相对路径
            #print("argv source=" + source_dir)
            i = i+2
        elif param=='-l': #保存输出的latex目录
            latex_dir = sys.argv[i+1] #保存相对路径
            i = i+2
            #print("latex_dir = "+latex_dir)

source_dir = '' #保存源文件所在目录
build_dir = ''  #保存生成文件所在目录
releasever = ''
latex_dir = ''
doc_language =''
__load_dict__ = {}

if __name__ == '__main__':

    parseargv()  #解析系统参数
    OpenMakefile()

    latex_documents = GetLatex_documents() #保存latex文档所在数组
    __load_dict__ = __openconfjsonfile__()

    latexdir=""
    if latex_dir == '':
        latexdir = '/latex/'
    else:
        latexdir = '/' + latex_dir + '/'

    doclen = len(latex_documents)
    for i in range(0,doclen):
        #得到latex路径
        latexpath = './' + build_dir + latexdir
        #copy 背景图到latex路径，背景图必须与该文件在同一个目录下
        if os.path.exists('./chapterbkpaper.pdf'):
            shutil.copy('./chapterbkpaper.pdf',latexpath)
        #得到相对路径
        if getquomarkcontent(latex_documents[i][1]) is None:
            continue
        texfilepath = latexpath + getquomarkcontent(latex_documents[i][1])
        #相对路径转绝对路径
        texfile = os.path.abspath(texfilepath)
        if not os.path.exists(texfile):
            continue
        fo = codecs.open(texfile, "r+",encoding = 'utf-8')
        texcontent = fo.read()
        fo.close()

        #得到修改tex文件的对象
        ModTexobj = clsModifyTex(texcontent)
        ModTexobj.AddPackageToTex()
        ModTexobj.AddOtherTocToTex()
        ModTexobj.AddCustormOptionsToTex()
        ModTexobj.ModifyFunctionBackColor()
        ModTexobj.ModifyTablesAttributes()
        ModTexobj.ModifyReplacePackage()

        fw = codecs.open(texfile, "w+",encoding = 'utf-8')
        fw.write(ModTexobj.content)
        fw.close()



