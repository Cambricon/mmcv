#!/usr/bin/python
# -*- coding: UTF-8

"""
2023年12月05日 修改历史
================================

修改内容如下：
1. 增加cnparsed-literal指令，和parse-literal指令相对应，
  但是cnparsed-literal指令可以修改字体颜色，背景颜色，字体大小等参数，对于pdf还可以修改行间距等，功能比较多。

特别注意：
* 如果安装的xelatex不包含ulem包，则cnparsed-literal和cncolor指令要使用underline和strike参数，必须将ulem.sty放到和该文件同一目录下，否则会导致编译错误。
* 两个指令参数的详细说明参见cnColorDirectivecls和cnparsedliteralDirectivecls两个类的说明。
* cnparsed-literal和cncolor指令的underline和strike参数同时使用不支持自动换行。
"""

import os
import shutil
#import sys
#import warnings
import time
import re
from sphinx.directives.other import TocTree
from docutils.parsers.rst import Directive
from docutils.parsers.rst import directives
from docutils.parsers.rst.roles import set_classes
from docutils import nodes

from sphinx.util.docutils import SphinxDirective
#from sphinx.util.typing import OptionSpec
from docutils.statemachine import ViewList
#from sphinx.util.nodes import nested_parse_with_titles
#import traceback

class CNOnly(Directive):

    '''
    cnonly指令，实现在一个列表内、一个段落内、一个表格内按定义的tag，条件输出不同内容。
    cnonly指令包含的内容为同一个列表、同一个段落、同一个表格，如果本身就是两个段落的内容，请使用原始only指令。
    cnonly指令和only指令的区别：
    1. 可以用在一个列表内，被指令包含的item与前后item之间无空行。only指令默认就是两个段落，因此指令内容和前后item之间有空行，效果比较难看。
    2. 用在一个段落内，按tag条件输出。only指令只能用于两个段落，不支持在同一段落内使用。
    3. 可用于一个表格内，按tag条件输出。only指令不支持用于表格内。
    使用cnonly注意事项：
    1. cnonly为了在同一段落中使用，默认会删除指令前的一行空行，因此如果是两个段落，cnonly指令前至少保留两个空行。
    '''

    has_content=True
    
    '''
    notailspace
    ---------------
    cnonly指令参数，是否在指令内容后添加空行，无需赋值。
    添加了该参数，则在指令内容和下一段之间不添加空行，则指令内容和下一段内容属于同一段落。
    不添加该参数，则指令内容和下一段之间添加空行，指令内容和下一段内容为独立的两个段落。
    cnonly指令默认不添加该参数。
    '''
    option=["notailspace"]
    
    def run(self):
        return []

class CNOnlyPro():

    def __init__(self,app,docname,sourcestr):
        self.tags=app.tags
        self.docname=docname
        self.source = sourcestr

    #判断cnoly指令是否存在notailspace参数，如果有该参数，则在指令内容和剩余内容间不添加空行，比如表格。
    #添加空行的目的是为了本来该是两个段落的内容，避免合并成一个段落。
    #默认没有该参数，添加空行。
    def __parsecnonlyoption(self,optionstr)->bool:
        if optionstr.strip()==":notailspace:":
            return True
        else:
            return False
    #根据tag判断cnonly指令是否应该包含
    def __parsecnonlylst(self,cnonlystr,onlyspacecount,onlylst,spacelst):

        regstr = r".. cnonly::([\s\S]*)"
        pattern = re.compile(regstr,re.I|re.U)
        tagstr = pattern.search(cnonlystr).group(1).strip()
        #判断tag是否存在
        #print('tagstr=%s,onlycontent=%d' % (tagstr,len(onlycontent)))
        tagsflag = parsejointtags(self.tags,tagstr)
        #print(tagsflag)
        if (not tagsflag) or (len(onlylst)==0):
            #不包含该tag，则tag包含内容全部删除
            return ""
        else:
            delspacecount = 0
            for i,j in enumerate(onlylst):
                if j.strip()=="":
                    continue
                if delspacecount==0:  #cnonly后的第一个列表前面添加0个空格，因为要和cnonly前的空格保持一致。
                    delspacecount= spacelst[i]-onlyspacecount
                #print("delspacecount=%d" % delspacecount)
                #后续的所有列表都删掉前面的delspacecount数量的空格
                onlylst[i] = dellefspaces(j,delspacecount)
                #print('onlylst[%d]=%s' % (i,onlylst[i]))

            onlycontentstr = "\n".join(onlylst)+'\n'

            return onlycontentstr
    #根据cnonly指令将字符串拆分为三部分
    def parsecnonlycontent(self,sourcestr):
        regstr = r"( *).. cnonly::([\s\S]*)"
        pattern = re.compile(regstr,re.I|re.U)
        matchobj = pattern.search(sourcestr)

        if matchobj != None:
            #转字符串为列表，方便操作
            startpos = matchobj.start() #得到起始位置
            if startpos>0:
                #先转成列表，删除指令前的空行
                presourcelst = sourcestr[0:startpos].split('\n')
                presourcelst.pop()
                if presourcelst[-1].strip()=="":
                    presourcelst.pop()
                presource=('\n').join(presourcelst)+'\n'
            else:
                presource=""
            leftsource = sourcestr[startpos:len(sourcestr)]
            #将带cnonly的字符串转成列表，方便解析
            sourcelst = leftsource.split('\n')
            #列表的第一个元素即为cnonly字符串
            #循环找出cnonly字符串涵盖的内容
            onlyspacecount = countspaces(sourcelst[0])
            onlycontent=[]
            spacelst=[]
            notailspace= self.__parsecnonlyoption(sourcelst[1]) #判断第一行是否为“notailspace”参数行。
            if notailspace:
                startpos=3 #忽略掉指令行、参数行、和空行
            else:
                startpos=2 #忽略掉指令行、参数行
            for i,j in enumerate(sourcelst[startpos:]):
                if j=="" or j.isspace()==True:
                    onlycontent.append(j)
                    continue
            
                spacecount=countspaces(j)
            
                if spacecount>=(onlyspacecount+3):
                    spacelst.append(spacecount)  #将空格数量加到空格数量列表
                    onlycontent.append(j)
                else:
                    break
            
            #删除指令后，内容前的空行
            while True:
                if len(onlycontent) > 0 and (onlycontent[0]=="" or onlycontent[0].isspace()==True):
                    onlycontent.pop(0)
                else:
                    break
            
            #删除指令内容后的空行
            while True:
                if len(onlycontent) > 0 and (onlycontent[-1]=="" or onlycontent[-1].isspace()==True):
                    onlycontent.pop()
                else:
                    break
            #print("i=%d" % i)
            #经过解析后onlycontent包含cnonly指令包含的所有内容
            onlycontentstr = self.__parsecnonlylst(sourcelst[0],onlyspacecount,onlycontent,spacelst)
            leftsource ='\n'.join(sourcelst[i+startpos:])
            #print(leftsource)
            leftsource = self.parsecnonlycontent(leftsource)
            if notailspace:
                return presource+onlycontentstr+leftsource
            else:
                return presource+ onlycontentstr +'\n' + leftsource
        else:
            return sourcestr
        
def parsejointnobrackets(tags,tagstr):
    '''
    解析没有括号的and or组合。因为and的优先级高于or，因此从or开始拆分字符串。
    :param tags: 
    :param tagstr: 
    :return: 
    '''
    orlst = tagstr.split(r" or ")
    if len(orlst)<=1:
        #说明全是and的组合
        return parsemultitags(tags,tagstr)
    
    result = False
    for i, j in enumerate(orlst):
        if j ==" or " or len(j)==0 or j.isspace() is True:
            continue
        #and组合的内容
        result = parsemultitags(tags,orlst[i])
        if result is True:
            break
            
    return result

def parsejointtags(tags,tagstr):
    '''
    解析带括号的逻辑判断，支持括号嵌套
    :param tags: 
    :param tagstr: 
    :return: 
    '''
    #根据正则表达式，先将括号内容筛选出来
    searchstr = r"\(([^\(\)]*?)\)"
    mobjlst = re.findall(searchstr,tagstr,re.I)
    if len(mobjlst) == 0:
        #如果没有括号，就按and or的混合组合处理
        return parsejointnobrackets(tags,tagstr)
    
    mobjarr = re.finditer(searchstr,tagstr,re.I)
    prepos = 0 #字符串拆分的起始位置
    newtagstr = ""#组合后的新的字符串
    for mobj in mobjarr:
        newtagstr += tagstr[prepos:mobj.start()]
        #判断括号中的组合结果，根据结果连接成新的字符串
        #内部第一层括号
        result = parsejointnobrackets(tags,mobj.group(1))
        newtagstr += "_"+str(result)+"_" #为了避免和True、Flase标签冲突，前后添加下划线
        prepos = mobj.end()
        
    if len(tagstr) > prepos:
        #再将剩余的内容加入到firsttaglst中
        newtagstr += tagstr[prepos:len(tagstr)]
        
    #递归调用，避免有括号嵌套
    return parsejointtags(tags,newtagstr)
        
#判断是否有指定的tag
def parsemultitags(tags,tagstr)->bool:
    #该函数判断是否有多个tags的组合，比如 or组合，and组合。
    #当前仅支持全是or或者全是and的组合，不支持or和and的混合组合。
    #or组合优先，有or组合先按or组合判断。
    strlst=tagstr.split(r" or ")
    #print("tags or:" + ",".join(strlst))
    if len(strlst) <=1:
        #如果不能根据or分割出数据，再根据and分割
        strlst=tagstr.split(r" and ")
        #print("tags and:" + ",".join(strlst))
        if len(strlst) <=1:
            #"_True_"是括号中的内容已经被解析后的结果，因此不用在重新解析了
            if ("_True_" == tagstr.strip()) or tags.has(tagstr.strip()):
                return True
            else:
                return False
        else:
            count = len(strlst)
            icount = 0
            for i in range(0,count):
                if ("_True_" == strlst[i].strip()) or tags.has(strlst[i].strip()):
                    icount+=1
            #and组合全部都有则返回true，有一个标签不包含就返回fasle
            if icount == count:
                return True
            else:
                return False
    else:
        for i in range(0,len(strlst)):
            if ("_True_" == strlst[i].strip()) or tags.has(strlst[i].strip()):
                return True
        return False
#得到字符串开头的空格数量
def countspaces(str):
    for i, j in enumerate(str):
        if j != ' ':
            return i

#删除字符串前指定数量的空格
def dellefspaces(str,count):
    if count<=0:
        return str

    for i in range(0,count):
        if str[0]==" ":
            str = str[1:]
    return str

#自定义cntoctree指令，使其支持only指令
#使用方式：使用toctree指令的地方，直接修改为cntoctree，则在cntoctree指令的下面可以使用only指令，根据条件编译包含的文件。
class TocTreeFilt(TocTree):
    '''
    该指令实现在大纲目录中按定义的条件输出，即在大纲目录中可以使用only指令。
    
    使用方法
    -----------------
    将toctree直接替换为cntoctree即可。
    
    注意
    -----------------
    cntoctree中的only指令仅支持tag的全or或者全and的组合，不支持or和and的混合组合，也不支持使用“()”括号进行分组。
    '''
    def __GetOnlyStrByReg(self,contentstr):
        #根据正则表达式，得到字符串中所有的only完整的字符串，并放到list列表里
        onlylst=[]
        regstr = r"(.. only::[\s\S]*?),"
        pattern = re.compile(regstr,re.I|re.U)
        onlylst = pattern.findall(contentstr)
        return onlylst

    #判断only的内容
    def __GetOnlyContent(self,content):

        onlystr = content[0] #第一个元素为only指令所在的字符串
        #得到only字符串前面的空格数
        onlyspacecount=countspaces(onlystr)
        #print("onlyspacecount=%d,onlystr=%s"  % (onlyspacecount,onlystr))
        #找出only包含的所有内容，以缩进的空格为依据。only指令包含的内容，必须缩进数量>(onlyspacecount+3)
        onlycontent=[]
        leftcontent=[]
        endonly=False
        for i,j in enumerate(content[1:]):
            if j=="":
                continue
            spacecount=countspaces(j)
            #print("spacecount=%d"  % spacecount)
            if spacecount>=(onlyspacecount+3) and (not endonly):
                onlycontent.append(j)
            else:
                leftcontent.append(j)
                endonly = True

        if len(onlycontent)>0:
            #判断only标签是否已定义
            #根据only字符串，返回修改后的列表
            #根据正则表达式，解析出only后的tag标签
            regstr = r".. only::([\s\S]*)"
            pattern = re.compile(regstr,re.I|re.U)
            tagstr = pattern.search(onlystr).group(1).strip()
            #print('tagstr=%s' % tagstr)
            tagsflag = parsejointtags(self.env.app.tags,tagstr)
            if not tagsflag:
               onlycontent.clear()
        return leftcontent,onlycontent

    def __filter_only(self,content):
        #解析only指令
        #print(content)
        onlystr = '.. only::'
        #将列表转为字符串，用英文逗号链接，方便判断
        contentstr = ','.join(content)
        newcontent=[]
        count = len(content)
        for i,j in enumerate(content):
            #print("j=%s,i=%d" % (j,i))

            if onlystr in j:
                #print(content[i:])
                leftcontent,onlycontent = self.__GetOnlyContent(content[i:])
                if len(onlycontent)>0:
                    newcontent += self.__filter_only(onlycontent)
                if len(leftcontent)>0:
                    newcontent += self.__filter_only(leftcontent)

                break
            else:
                newcontent.append(j)

        return newcontent

    def run(self):
        # 过滤only条件
        self.content = self.__filter_only(self.content)
        #清除目录前后的空格，否则无法找到该目录
        for i,j in enumerate(self.content):
            self.content[i]=self.content[i].strip()

        return super().run()

#---------------------------------------------------------------------------

#latex字体与pt的对应表，数值的单位为pt，pt与px的对应关系pt=px * 3/4
latex_fontsize={5:r'\tiny',7:r'\scriptsize',8:r'\footnotesize',
                9:r'\small',10:r'\normalsize',11:r'\large',
                14.4:r'\Large',17.28:r'\LARGE',20.74:r'\huge',24.88:r'\Huge'}

fontmaxsize_pt = 25  #超过25pt，字体为最大字体Huge，仅用于latex
fontmaxsieze_px = 34 #超过34px，字体为最大字体Huge，仅用于latex



class cnautowrapnode(nodes.Inline, nodes.TextElement):
    """
    cnautowrap node类，用于解析autowrap节点，支持长英文的换行
    """
    pass
            
def __ModifyPureTextforlatex(text):
    """
    修改原来的文本，如果文本中有\sphinxhyphen{}或者\PYGZhy{}
    需要添加大括号进行分割，否则\seqsplit命令无效
    """
    #用正则表达式查找\sphinxhyphen{}或者\PYGZhy{}
    newtext = ""
    startpos = 0
    #searchstr = r"\\sphinxhyphen\{\}|\\PYGZhy\{\}"
    searchstr = r"-|—|——"
    m = re.finditer(searchstr, text, re.I | re.U)
    for match in m:
        newtext = newtext + text[startpos: match.start()] + " :raw-latex:`{\PYGXhy{}}` "
        startpos = match.end()
    if len(text) > startpos:
        newtext = newtext + text[startpos:len(text)]
    
    #print(newtext)
    return newtext
        
def autowrap_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    autowrap role主函数，该函数实现对有符号连续长英文的自动换行。
    """
    #attrs ={'style':'color:RGB(255,0,0);font-weight:bold;font-style:italic;font-size:15px'}
    #判断文本中间有没有其它指令，默认有红色属性
    #puretext = text
    #print(text)
    env = inliner.document.settings.env
    if hasattr(env, 'autowrapfile') and (env.docname not in env.autowrapfile):
        #print("-----------html enter---------------------")
        env.autowrapfile.append(env.docname)
    else:
        #print("-----------html enter1---------------------")
        env.autowrapfile = []
        env.autowrapfile.append(env.docname)
        #print('===================')
        #print('make latex')
        #print('===================')
        #puretext = __ModifyPureTextforlatex(text)  #为特殊指令添加大括号，否则将失败
    nodelst = []
    nodelst.append(cnautowrapnode(rawtext,text,*content,**options))
    #print(cnautowrap(rawtext,puretext,*content,**attrs))
    return nodelst,[]


def latex_visit_autowrap_node(self, node):
    #print('-----------------')
    #print(node.attributes)
    #print(node.rawsource)
    # print(type(self))
    #print('-----------------')
    __modifyNodeforlatexraw(node)
    self.body.append(r"\seqsplit{")


def latex_depart_autowrap_node(self, node):
    #print(node)
    self.body.append('}')
    

def html_visit_autowrap_node(self, node):
    pass

def html_depart_autowrap_node(self, node):
    pass

class cncolorrolenode(nodes.Inline, nodes.TextElement):
    """
    cncolorrole node类，用于解析cncolor role节点
    """
    pass
def __GetComplexColorAttr(attrstr,colorkey):
    """
    :param colorkey: color或者bg，查找字体颜色还是背景颜色。
    """
    clrkey = colorkey +':'
    
    match = re.search(clrkey,attrstr)
    if match is None:
       return '',attrstr
        
    #如果有类似关键字，为了正则表达式的准确性，先查找是否有rgb颜色值
    searchstr = clrkey + '(RGB\((0\.)?[0-9]+,(0\.)?[0-9]+,(0\.)?[0-9]+\))'
    match = re.search(searchstr,attrstr,re.I)
    if match:
        pre = ''
        left = ''
        if match.start()>0:
            pre = attrstr[0:match.start()]
        if match.end() < len(attrstr):
            left = attrstr[match.end()+1:len(attrstr)]
        
        return match.group(1),pre + left
    attrstr +=',' #末尾加逗号，避免在最后的位置正则表达式匹配失败
    #再查找是否有纯颜色，比如：color:yellow
    searchstr = clrkey +'([a-zA-Z]+?)(?=,)'
    match = re.search(searchstr,attrstr,re.I)
    if match:
        pre = ''
        left = ''
        if match.start() > 0:
            pre = attrstr[0:match.start()]
        if match.end() < len(attrstr)-1:
            left = attrstr[match.end()+1:len(attrstr)-1] #因为正则表达式排除了逗号，因此删除剩余的逗号

        return match.group(1), pre + left
    else:
        return '', attrstr.strip(',') #去掉末尾的逗号

    
    
def __GetcomplexfzAttr(attrstr):
    attrstr += ','  # 末尾加逗号，避免在最后的位置正则表达式匹配失败
    #查找是否有字体大小属性
    searchstr ='fz:([\s\S]+?)(?=,)'
    match = re.search(searchstr,attrstr,re.I)
    if match:
        pre = ''
        left = ''
        if match.start() > 0:
            pre = attrstr[0:match.start()]
        if match.end() < len(attrstr)-1: #去掉末尾逗号
            left = attrstr[match.end()+1:len(attrstr)-1]

        return match.group(1), pre + left
    else:
        return '',attrstr.strip(',') #去掉末尾逗号
    
def __GetRGBColorandAttrlist(attrstr):
    """
    判断属性是否包含RGB()格式的自定义颜色属性
    返回RGB格式颜色和其它属性列表，无RGB格式属性，该返回值为""。
    """
    #先判断是否有独立颜色
    searchstr = 'rgb'
    matchobj = re.search(searchstr, attrstr, re.I)
    if matchobj is None:
        # 没有自定义的RGB颜色值，返回属性列表
        return "", attrstr.split(",")
    
    #在判断是否有RGB颜色
    searchstr = r'RGB\([0-9]+,[0-9]+,[0-9]+\)'
    matchobj = re.search(searchstr,attrstr)
    if matchobj:
        newattr = attrstr[0:matchobj.start()]+attrstr[matchobj.end()+1:len(attrstr)]
        return matchobj.group(),newattr.split(",")
    
    #在判断是否有rgb颜色
    searchstr = r'rgb\(0\.[1-9]+,0\.[1-9]+,0\.[1-9]+\)'
    matchobj = re.search(searchstr,attrstr)
    if matchobj:
        newattr = attrstr[0:matchobj.start()]+attrstr[matchobj.end()+1:len(attrstr)]
        return matchobj.group(),newattr.split(",")
    

def __GetAttrAndPureText(text):
    """
    判断是否有属性内容，属性内容放到纯文本的前面，被“{}”包围，如果要显示“{}”,
    请在开头添加转义字符，即“\{}”。如果内容不符合规则也会当作纯文本。
    属性内容的写法：{属性1,属性2,属性3,属性4}，中间用英文逗号“,”分割，属性无固定顺序关系。
    支持的属性：
    
    * 字体颜色属性，语法：color:<要设置的颜色>，支持的关键字：red,green,blue,black,white,yellow,RGB(,,),rgb(,,)。
      除了上面提到的固定关键子的颜色外，为了同时兼容html和latex，请使用大写RGB的方式定义颜色。
      默认字体颜色为红色，添加red属性关键字。
    * 背景颜色属性，语法：bg:<要设置的颜色>，支持的关键字同上。
    * 字体大小属性，语法：fz:<要设置的字体大小>。
    * 是否为斜体，关键字：itatic，添加了该关键字，则设置字体为斜体，默认非斜体。
    * 字体是否加粗，关键字：bold，添加了该关键字，则对字体加粗处理，默认不加粗。
       注意：斜体和粗体属性，如果同时被设置，latex只对字体加粗，斜体设置无效。html支持同时设置。
    * 是否添加下画线，关键字：underline，添加了该关键字，则对字体加添加下划线，默认不加下划线。
    * 是否添加删除线，关键字：strike，添加了该关键字，则对字体加添加删除线，默认不加删除线。
    
    比如，如果将一段字体设置为红色带下划线的斜体，则写法如下：
    :cncolor:`{color:red,bg:yellow,fz:50px,strike,itatic}描述`
    以上写法将“描述”设置为红色带下划线的斜体。
    """
    
    #判断文本开头是否符合属性定义的语法，如果有转义字符或不是以“{”开头，则直接返回。
    attrs = {}
    if ord(text[0]) == 92 or ord(text[0]) != 123:
        return {'style': 'color:red'},text
    #查找属性定义
    #searchstr = "(?<=\{)([\s\S]+?)(?=\})" #使用match匹配，不支持?<=语法
    searchstr = "{([\s\S]+?)}"
    matchobj = re.match(searchstr,text,re.I|re.M)
    if matchobj is None:
        return {'style': 'color:red'},text

    #中间有可能换行，将换行符替换掉
    attrstr = re.sub(r'\n|\r','',matchobj.group(1).strip('\n').strip('\r'),re.M)
    endpos = matchobj.end()
    puretext = text[endpos:len(text)]
    isattr = False #是否有属性内容，默认没有属性内容
    #先判断是否设置了字体颜色
    rgbcolor,leftstr = __GetComplexColorAttr(attrstr,'color')
    if rgbcolor == "":
        attrs['style'] = 'color:red'
    else:
        attrs['style'] = 'color:' + rgbcolor
        isattr = True
    #判断是否有背景色
    bgcolor,leftstr = __GetComplexColorAttr(leftstr,'bg')
    if bgcolor !='':
        attrs['style'] = attrs['style'] + ';background:' + bgcolor
        isattr = True
    #判断是否设置了字体尺寸
    fontsize,leftstr = __GetcomplexfzAttr(leftstr)
    if fontsize !='':
        attrs['style'] = attrs['style'] + ';font-size:' + fontsize
        isattr = True
    #用逗号分割leftstr
    attrlst = leftstr.split(',')
    # 判断是否有斜体属性
    if 'italic' in attrlst:
        attrs['style'] = attrs['style'] + ";font-style:italic"
        isattr = True
    if 'bold' in attrlst:
        attrs['style'] = attrs['style'] + ";font-weight:bold"
        isattr = True
    if 'underline' in attrlst:
        attrs['style'] = attrs['style'] + ";text-decoration:underline"
        isattr = True
    if 'strike' in attrlst:
        attrs['style'] = attrs['style'] + ";text-decoration:line-through"
        isattr = True
    if isattr:
        return attrs, puretext
    else:
        return attrs, text
    '''
    #根据逗号进行分割
    rgbcolor,attrlst = __GetRGBColorandAttrlist(attrstr)
    if rgbcolor !="":
        attrs['style'] = 'color:' + rgbcolor
        isattr = True
    else:
        #判断属性里是否有颜色设置，先判断是否有固定颜色值的设置
        cuscolor = list(set(colorlst) & set(attrlst))
        if len(cuscolor)==1:
            #有多个固定颜色的设置认为出错
            # 有固定颜色值设置，属性换为自定义颜色
            attrs['style'] = 'color:'+cuscolor[0]
            isattr = True
        else:
            attrs['style']= 'color:red'
        
    #判断是否有斜体属性
    if 'italic' in attrlst:
        attrs['style'] = attrs['style'] + ";font-style:italic"
        isattr = True
    if 'bold' in attrlst:
        attrs['style'] = attrs['style'] + ";font-weight:bold"
        isattr = True
    if 'underline' in attrlst:
        attrs['style'] = attrs['style'] + ";text-decoration:underline"
        isattr = True
    if 'strike' in attrlst:
        attrs['style'] = attrs['style'] + ";text-decoration:line-through"
        isattr = True

    if isattr:
        return attrs,puretext
    else:
        return attrs,text
    '''
        
def cncolor_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    cncolor role主函数，该函数实现对指定文本的颜色渲染，默认渲染为红色。
    """
    #attrs ={'style':'color:RGB(255,0,0);font-weight:bold;font-style:italic;font-size:15px'}
    #判断文本中间有没有其它指令，默认有红色属性
    #app = inliner.document.settings.env.app
    attrs,puretext = __GetAttrAndPureText(text)
    nodelst = []
    nodelst.append(cncolorrolenode(rawtext,puretext,*content,**attrs))
    #print(cncolor(rawtext,puretext,*content,**attrs))
    return nodelst,[]

def __IsStrikeAttr(attrstr):
    """
    判断属性里面是否有删除标识，有删除标志将删除标识删除掉，同时返回true。
    """
    searchstr = "text-decoration:line-through"
    matchobj = re.search("text-decoration:line-through",attrstr,re.I)
    if matchobj is None:
        return attrstr,False
    attrstr = attrstr[0:matchobj.start()]+attrstr[matchobj.end():len(attrstr)]
    return attrstr,True
    
def html_visit_cncolor_node(self, node):
    
    #判断属性里面有没有text-decoration:line-through删除标志，有删除标志使用<del></del>标签
    #否则和text-decoration:underline标志会存在冲突
    #print(node.attributes['style'])
    strikeflag = False
    attrstr,strikeflag = __IsStrikeAttr(node.attributes['style'])
    attrs={'style':attrstr}
    tag = self.starttag(node,'span','',CLASS='cncolorrole',**attrs)
    self.body.append(tag)
    if strikeflag:
        self.body.append(self.starttag(node, 'del'))
    #print(node)
    #print(node.attributes)
    #print(node.tagname)
    #print(node.astext())
    #print(type(self))


def html_depart_cncolor_node(self, node):
    strikeflag = False
    attrstr, strikeflag = __IsStrikeAttr(node.attributes['style'])
    if strikeflag:
        self.body.append('</del>')
    self.body.append('</span>')
    #print(node)
    
def __GetColorAttr(attrstr,colorkey):
    '''
    :param colorkey: color或者bg,得到字体或者背景色
    '''
    #为了正则表达式好查找，给属性字符串添加{}
    newattr = '{'+attrstr+'}'
    searchstr = colorkey + ':([\s\S]+?)(?=[;|\}])'
    matchobj = re.search(searchstr,newattr,re.I)
    if matchobj is None:
        return "{}"
    colorattr = matchobj.group(1)
    
    #得到颜色
    searchstr = 'rgb'
    matchobj = re.match(searchstr, colorattr, re.I)
    if matchobj is None:
        return "{" + colorattr + "}" # 独立颜色
    # 判断是大写RGB还是小写rgb
    # 先按大写RGB搜索
    searchstr = 'RGB\(([0-9]+,[0-9]+,[0-9]+)\)'
    matchobj = re.match(searchstr, colorattr)
    if matchobj:
        return "[RGB]{" + matchobj.group(1) + "}"
    # 再搜索是否有小写的rgb颜色设置
    searchstr = 'rgb\((0\.[1-9]+,0\.[1-9]+,0\.[1-9]+)\)'
    matchobj = re.match(searchstr, colorattr)
    if matchobj:
        return "[rgb]{" + matchobj.group(1) + "}"

def __GetLatexFontSizeFlag(attrstr,fontsize):
    attrstr +=';'  #最后添加分号，方便正则表达式查找
    searchstr = fontsize+":([0-9]+)(pt|px)(?=;)"

    match = re.search(searchstr,attrstr,re.I)
    if match is None:
        return ""
    try:
        fontvalue = int(match.group(1))
        fontunit = match.group(2)
        if fontunit == 'px':
            #将px转为pt，按96dpi进行的转化，pt = px * (72/96)
            fontvalue = fontvalue * 0.75
        if fontvalue >= fontmaxsize_pt:
            return '\Huge'
        dicttmp = {}
        #取出latex字典的所有key值
        keylst = latex_fontsize.keys()
        for value in keylst:
            diffabs = abs(fontvalue-value)
            if not dicttmp:
                dicttmp[diffabs]=value
            else:
                prevalue = next(iter(dicttmp))
                if diffabs < prevalue:
                    dicttmp[diffabs]=value
                    del dicttmp[prevalue]  #只保存最小的一个，即最接近的一个。

        #取出保存最接近的pt值，即字典的第一个，因为该字典只有一个
        value = dicttmp.get(next(iter(dicttmp)))
        return latex_fontsize[value]
    
    except Exception as e:
        print(e)
        return ""

def latex_visit_cncolor_node(self, node):
    #print('-----------------')
    # print(node.attributes['style'])
    #print(node)
    #print(type(self))
    #print('-----------------')
    
    attrstr = node.attributes['style']
    self.in_production_list = 0
    latexattr = ""
    
    #设置字体尺寸，如果是pt或者px，取和latex最相近的字体设置
    if "font-size:" in attrstr:
        latexflag = __GetLatexFontSizeFlag(attrstr,'font-size')
        if latexflag != "":
            latexattr = latexflag+'{'+latexattr
            self.in_production_list+=1
        
    #如果有斜体属性
    if "font-style:italic" in attrstr:
        latexattr = r'\textit{'+latexattr
        self.in_production_list+=1
    #如果有粗体属性
    if "font-weight:bold" in attrstr:
        latexattr = r'\textbf{'+latexattr
        self.in_production_list+=1
    #取出设置的颜色
    if 'color:' in attrstr:
        color = __GetColorAttr(attrstr,'color')
        colorattr = r'\textcolor' + color
        latexattr = colorattr + '{' + latexattr
        self.in_production_list += 1
    #如果有删除线属性
    if "text-decoration:line-through" in attrstr:
        latexattr = r"\sout{" + latexattr
        self.in_production_list += 1
    #如果有下划线属性
    if "text-decoration:underline" in attrstr:
        latexattr = r"\uline{" + latexattr
        self.in_production_list += 1
    if "background:" in attrstr:
        color = __GetColorAttr(attrstr,'background')
        colorattr = r'\colorbox' + color
        latexattr = colorattr + '{' + latexattr
        self.in_production_list += 1
        
    self.body.append(latexattr)

def latex_depart_cncolor_node(self, node):
    for i in range(self.in_production_list, 0, -1):
        self.body.append('}')
    self.in_production_list = 0
    
#--------------------------------cncolordirective------------------------------------------------

class cncolordirective(nodes.Inline, nodes.TextElement):
    """
    cncolorrole node类，用于解析cncolor role节点
    """
    pass
class cncolorliteral(nodes.literal_block):
    """
    为了能够自定义html标签和latex符号
    """
    pass
class cncolorline(nodes.literal_block):
    """
    该节点是为了逐行添加latex指令，如果cncolor指令设置了下划线和删除线选项，则只能逐行添加参数。
    """
    pass
class cncolorsection(nodes.section):
    """
    为了能够自定义html标签和latex符号
    """
    pass
class cncolortext(nodes.Text):
    """
    为了给空行添加换行符
    """
    pass
#自定义cncolor指令
class cnColorDirectivecls(SphinxDirective):
    '''
    自定义cncolor指令核心类
    8个options：
    * literal: 内容是否当作纯文本解析，默认解析文本中的指令。如果添加了该字段，则内容会被当作纯文本。
               只解析行内指令，其它sphinx指令不支持。比如list不会被解析。
    * color：字体颜色，字符串。支持red,green,blue,black,white,yellow和RGB(,,)格式的颜色设置。
    * bgcolor：背景颜色，字符串。支持red,green,blue,black,white,yellow和RGB(,,)格式的颜色设置。
    * italic：斜体的标识，添加该参数则将字体设置为斜体，不添加该参数斜体无效。
    * bold：粗体的标识，添加该参数则将字体设置为粗体，不添加该参数粗体无效。
    * underline：添加下划线，添加该参数则为字体设置下划线，不添加该参数不添加下划线。latex必须要求包含ulem包才能生效，否则不生效。
    * strike：添加删除线，添加该参数则为字体设置删除线，不添加该参数不添加删除线。latex必须要求包含ulem包才能生效，否则不生效。
             而且添加了删除线后不支持自动换行，需要手动换行。
    * istitle: chapter 或者 section。当添加了istitle后，如果内容包含标题，则认为是一级大纲chapter，否则认为是section。
    * fontsize: 字体大小，以px或者pt为单位。html直接使用已设置的值，latex会转为latex字体大小进行设置。
                latex最大字体\Huge为34px，因此当字体大小设置为34px以上时，对latex文件来说字体大小都是\Huge。
    
    注意：
    underline和strike同时使用不支持自动换行。单独使用underline和strike可以自动换行。
    要使用underline和strike参数，ulem.sty包必须和该文件放到同一目录下。为了兼容有的xelatex不包含ulem包所做的操作。
    只有ulem.sty存在，underline和strike两个参数才能生效，否则会导致编译失败。
    详细操作流程可以查看config_inited_handler和cd_build_finished_handler两个事件处理函数。
    如果xelatex本身就包含ulem包，请参考config_inited_handler和cd_build_finished_handler的处理流程，
    去掉对ulem.sty是否存在的判断，自行修改代码。
    '''
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = False
    has_content = True
    option_spec={
        'literal': directives.flag,
        'color': directives.unchanged,
        'bgcolor':directives.unchanged,
        'italic': directives.flag,
        'bold': directives.flag,
        'underline': directives.flag,
        'strike': directives.flag,
        'class': directives.class_option,
        'name': directives.unchanged,
        'istitle':directives.flag,
        'fontsize':directives.unchanged,
    }
    def __MakeSectionorTitleNode(self,sectionlst):
        '''
        生成section或者title节点
        '''
        titlelst, messages = self.state.inline_text(sectionlst[0], self.lineno)
        
        if 'istitle' in self.options.keys():
            # htmlfile列表负责将html一级标题的h0修改为h1，否则标题显示达不到预期
            if hasattr(self.env, 'htmlfile') and (self.env.docname not in self.env.htmlfile):
                # print(self.env.docname)
                self.env.htmlfile.append(self.env.docname)
            else:
                # print(self.env.docname)
                self.env.htmlfile = []
                self.env.htmlfile.append(self.env.docname)
                
        # 有title的section，在生成latex的时候，需要删除title子节点，否则生成的标题达不到预期
        # 因此有title的内容，每次都得重新编译，否则在增量编译时，无法修改title节点
        if not hasattr(self.env, 'sectionfile'):
            # print(self.env.docname)
            self.env.sectionfile = []
            self.env.sectionfile.append(self.env.docname)
        elif self.env.docname not in self.env.sectionfile:
            # print(self.env.docname)
            self.env.sectionfile.append(self.env.docname)
            
        if len(titlelst) == 1:
            # 说明标题不带sphinx行内指令，则无需生成title node。
            targetid = 'cncolorsection-%d' % self.env.new_serialno('cncolorsection')
            sectionnode = cncolorsection(
                '',
                nodes.title(text=sectionlst[0]),
                ids=[targetid],
                names=[nodes.fully_normalize_name(sectionlst[0])],
                *[],
                **self.options)
            return sectionnode
        else:
                
            title_node = nodes.title(sectionlst[0], '', *titlelst, **{})
            targetid = 'cncolorsection-%d' % self.env.new_serialno('cncolorsection')
            sectionnode = cncolorsection(ids=[targetid], names=[nodes.fully_normalize_name(sectionlst[0])])
            sectionnode.attributes = {**sectionnode.attributes, **self.options}
            sectionnode += title_node
            return sectionnode
            # 以下注释保留，为后续开发提供参考
            '''
            #rst = ViewList()
            #rst.append('\n'.join(sectionlst),'',0)
            #section = nodes.section()
            #section.document = self.state.document
            #nested_parse_with_titles(self.state, rst, section)
            
            #self.env.titledict = {sectionlst[0]:''}
            
            ##说明标题内有sphinx行内指令，需要通过nodes.title解析，然后在doctree-read处理事件中进行替换删除
            title_node = nodes.title(sectionlst[0], '', *titlelst, **self.options)
            sectionnode = cncolorsection(
                '',
                nodes.title(text=sectionlst[0]),
                ids=[nodes.make_id(sectionlst[0])],
                names=[nodes.fully_normalize_name(sectionlst[0])],
                *[],
                **self.options)
            return [title_node,sectionnode,node]
            '''
                
    def __MakeContentNode(self,contentlst):
        #print('ccccccccccccccccccccccc')
        text = '\n'.join(contentlst) + '\n'  # 末尾加一个换行符，否则如果正则表达式查不出来最后的符号行。
        #print(text)
        searchstr = r'([\x21-\x2f|\x3a-\x40|\x5b-\x60|\x7b-\x7e])\1+(?=[\r|\n|\r\n])\n'
        matchobj = re.search(searchstr,text)
        if matchobj:
            #生成新的属性，主要用于纯文本
            #newdict = self.options.copy()
            #if 'istitle' in newdict.keys():
            #    del newdict['istitle']
                
            parentnode = cncolordirective('','',*[],**self.options)
            startpos = matchobj.start()
            endpos = matchobj.end()
            if startpos > 0:
                pretext = text[0:startpos]
                if len(pretext) > 0:
                    #print(len(pretext))
                    #print('xxx')
                    #print(pretext.split('\n'))
                    #print('zzzz')
                    pretextlst = ViewList(pretext.split('\n'))
                    prenode = cncolordirective(pretext, '', *[], **self.options)
                    self.state.nested_parse(pretextlst, 0, prenode)
                    parentnode += prenode
                    
            titlenode=nodes.Text(matchobj.group())
            parentnode +=titlenode
            
            if endpos < len(text):
                endtext = text[endpos:len(text)]
                if len(endtext) >0:#
                    #print(len(endtext))
                    #print('xxx2')
                    #print(endtext)
                    #print('zzzz2')
                    newlist = ViewList(endtext.split('\n'))
                    leftnode = self.__MakeContentNode(newlist)
                    parentnode += leftnode
                    
            return parentnode
            
        else:
            #print('~~~~~~~~~~~~~~~~~~~~~~')
            #print(text)
            #print('~~~~~~~~~~~~~~~~~~~~~~')
            node = cncolordirective(text.strip('\n'), '', *[], **self.options)
            self.state.nested_parse(contentlst, 0, node)
            return node

    def __MakeContentNode_v2(self, contentlst):
        # print('ccccccccccccccccccccccc')
        text = '\n'.join(contentlst) + '\n'  # 末尾加一个换行符，否则如果正则表达式查不出来最后的符号行。
        # print(text)
        searchstr = r'([\x21-\x2f|\x3a-\x40|\x5b-\x60|\x7b-\x7e])\1+(?=[\r|\n|\r\n])\n'
        matchobj = re.search(searchstr, text)
        if matchobj:
            # 生成新的属性，主要用于纯文本
            # newdict = self.options.copy()
            # if 'istitle' in newdict.keys():
            #    del newdict['istitle']

            parentnode = nodes.paragraph()
            startpos = matchobj.start()
            endpos = matchobj.end()
            if startpos > 0:
                pretext = text[0:startpos]
                if len(pretext) > 0:
                    # print(len(pretext))
                    # print('xxx')
                    # print(pretext.split('\n'))
                    # print('zzzz')
                    pretextlst = ViewList(pretext.split('\n'))
                    prenode = nodes.inline()
                    self.state.nested_parse(pretextlst, 0, prenode)
                    parentnode += prenode
                    
            titlenode = nodes.Text('\n'+matchobj.group().strip('\n'))
            parentnode += titlenode

            if endpos < len(text):
                endtext = text[endpos:len(text)].strip('\n')
                if len(endtext) > 0:  #
                    # print(len(endtext))
                    # print('xxx2')
                    # print(endtext)
                    # print('zzzz2')
                    newlist = ViewList(endtext.split('\n'))
                    leftnode = self.__MakeContentNode_v2(newlist)
                    parentnode += leftnode
            
            return parentnode

        else:
            #print('~~~~~~~~~~~~~~~~~~~~~~')
            #print(contentlst)
            #print('~~~~~~~~~~~~~~~~~~~~~~')
            node =nodes.paragraph()
            self.state.nested_parse(contentlst, 0, node)
            return node
        
    def __GetNoliteralNode(self):
                                
        # 解析是否有标题
        if len(self.content) > 1:
            sectionlst, contentlst = self.__GetContentDictbySection(self.content)
        else:
            sectionlst = []
            contentlst = self.content
            
        newdict = {}
        if len(self.arguments)>0:
            newdict['caption'] = self.arguments[0]

        parentnode = cncolordirective('', '', *[], **{**self.options,**newdict})

        node = self.__MakeContentNode_v2(contentlst)
        node.line = self.content_offset + 1
        self.add_name(node)
        if sectionlst:
            # 增加section
            sectionnode = self.__MakeSectionorTitleNode(sectionlst)
            parentnode += node
            return [sectionnode,parentnode]
        else:
            parentnode +=node
            return [parentnode]

    def __GetLineNode(self):
        '''
        解析每一行，为每一行添加下划线和删除线指令
        '''
        #print(self.content)
        newdict = {}
        if len(self.arguments)>0:
            newdict['caption'] = self.arguments[0]

        parentnode = cncolordirective('', '', *[], **{**self.options,**newdict})
        lineno = 0
        linecount = len(self.content)
        for line in self.content:
            lineno +=1
            if len(line)>0:
                text_nodes, messages = self.state.inline_text(line, lineno)
                node = cncolorline(line, '', *text_nodes, **self.options)
                parentnode+=node
                if linecount > 1 and lineno < linecount:
                    #为每一行加一行换行符
                    nodetext = cncolortext("")
                    parentnode += nodetext
            else:
                node = cncolortext(line)
                parentnode+=node
        return [parentnode]
    
    def run(self):
        
        set_classes(self.options)
        classes = ['cncolorblock']
        if 'classes' in self.options:
            classes.extend(self.options['classes'])
        self.assert_has_content()

        if hasattr(self.env, 'cncolorfile') and (self.env.docname not in self.env.cncolorfile):
            # print("-----------html enter---------------------")
            self.env.cncolorfile.append(self.env.docname)
        else:
            # print("-----------html enter1---------------------")
            self.env.cncolorfile = []
            self.env.cncolorfile.append(self.env.docname)
        
        if 'literal' in self.options.keys():
            del self.options['literal'] #先将非必须的option删除
            # 将标题保存传出去
            newdict={}
            if len(self.arguments) > 0:
                newdict['caption'] = self.arguments[0]

            #全部当作纯文本解析
            text = '\n'.join(self.content)
            text_nodes, messages = self.state.inline_text(text, self.lineno)
            node = cncolorliteral(text, '',*text_nodes, **{**self.options,**newdict})
            node.line = self.content_offset + 1
            self.add_name(node)
            return [node] + messages
        elif ('underline'  in self.options.keys() or \
                'strike' in self.options.keys() or \
                'isparsedliteral' in self.options.keys()):
            return self.__GetLineNode()
        else:
            return self.__GetNoliteralNode()
        

    def __GetContentDictbySection(self,content):
        '''
        **注意：**
        cncolor指令下的内容，要么开始就是标题名称，要么不包含标题。
        
        带有section的内容使用literal_block或者nested_parse解析，都会给出以下告警::
        
        CRITICAL: Unexpected section title.
        
        因此检查文本中是否带有section内容，并根据section内容对文本进行拆分，
        返回标题名称列表和段落内容列表。
        '''
        #搜索可能的标题
        searchstr = r"^([\x21-\x2f|\x3a-\x40|\x5b-\x60|\x7b-\x7e])\1+$"
        text = content[1]
        #判断第2个是否为标题符号行
        match = re.match(searchstr,text)
        if match and len(content[0])>0:
            #说明内容的前两个是标题内容
            return content[0:2],content[2:len(content)]
        
        return None,content
        
class cnparsedliteralDirectivecls(cnColorDirectivecls):
    '''
    自定义cnparsed-literal指令类，模拟sphinx parsed-literal的指令。
    cnparsed-literal相对于parsed-literal指令，有以下优点：
    1.可以自动换行，避免了因为verbatimmaxunderfull的配置导致编译失败，或者预留换行宽度太宽的问题。
    2.可以自定义背景色、字体颜色等相关属性。
    3.可以动态调整行间距和字体大小，当因为换行超出文本块可显示宽度时，可以通过动态调整这两个参数达到最好的显示效果。
    cnparsed-literal指令因为只是为了模仿latex下和parsed-literal的显示效果一致，因此只用于pdf输出，对html没效果。
    如果也想影响html，请使用cncolor指令。
    * spacing: 设置行间距，浮点数或整数形式，如果不是浮点数或整数形式，会导致异常。
    * font-size：只支持设置latex定义的字体大小，对html无效。html字体设置请使用fontsize选项。
      latex支持的字体大小：'tiny','scriptsize','footnotesize','small','normalsize','large','Large','LARGE','huge','Huge'。
    * 其它参数同cncolor的处理。
    cnparsed-literal指令因为用的是tcolorbox包，因此不支持自动分页。
    如果需要分页，要么用官方的parsed-literal指令，要么手动分页。即在需要分页的地方用两个cnparsed-literal指令分开。
    '''
    option_spec={
        'spacing': directives.unchanged,
        'font-size': directives.unchanged,
        'literal': directives.flag,
        'color': directives.unchanged,
        'bgcolor': directives.unchanged,
        'underline': directives.flag,
        'strike': directives.flag,
        'italic': directives.flag,
        'bold': directives.flag,
        'class': directives.class_option,
        'name': directives.unchanged,
        'istitle': directives.flag,
    }

    def run(self):
        self.options['isparsedliteral'] = True
        set_classes(self.options)
        classes = ['cncolorblock']
        if 'classes' in self.options:
            classes.extend(self.options['classes'])
        self.assert_has_content()
        
        if hasattr(self.env, 'cncolorfile') and (self.env.docname not in self.env.cncolorfile):
            # print("-----------html enter---------------------")
            self.env.cncolorfile.append(self.env.docname)
        else:
            # print("-----------html enter1---------------------")
            self.env.cncolorfile = []
            self.env.cncolorfile.append(self.env.docname)
        return super(cnparsedliteralDirectivecls,self).run()
                        
def __GetHtmlStyleAttr(nodeattr):
    '''
    根据node属性生成，html标签的style属性。
    node属性是一个字典对象
    '''
    attr={}
    attr['style'] =''
    delflag = False #是否有删除标志
    keys = nodeattr.keys()
    #指令参数名称即为字典的key值
    if 'color' in keys:
        attr['style'] = 'color:' + nodeattr['color']
    if 'bgcolor' in keys:
        attr['style'] = attr['style'] + ';background-color:' + nodeattr['bgcolor']
    if 'italic' in keys:
        attr['style'] = attr['style'] + ';font-style:italic'
    if 'bold' in keys:
        attr['style'] = attr['style'] + ';font-weight:bold'
    if 'underline' in keys:
        attr['style'] = attr['style'] + ';text-decoration:underline'
    if 'fontsize' in keys:
        attr['style'] = attr['style'] + ';font-size:' + nodeattr['fontsize']
    if 'strike' in keys:
        delflag = True
    #print(attr)
    return attr,delflag

def __GetColorAttrValue(attrstr):
    '''
    判断颜色属性是否为rgb格式，如果是rgb格式需要重定义颜色
    如果不是rgb格式，就当作独立颜色
    '''
    #先判断开头是否为rgb或者RGB。
    searchstr = 'rgb'
    matchobj = re.match(searchstr,attrstr,re.I)
    if matchobj is None:
        return attrstr,False #独立颜色
    #判断是大写RGB还是小写rgb
    #先按大写RGB搜索
    searchstr = 'RGB\(([0-9]+,[0-9]+,[0-9]+)\)'
    matchobj = re.match(searchstr, attrstr)
    if matchobj:
        return "{RGB}{" + matchobj.group(1)+"}",True
    #再搜索是否有小写的rgb颜色设置
    searchstr = 'rgb\((0\.[1-9]+,0\.[1-9]+,0\.[1-9]+)\)'
    matchobj = re.match(searchstr, attrstr)
    if matchobj:
        return "{rgb}{"+matchobj.group(1)+"}",True
    
def __GetlatexStyleAttr(self,nodeattr):
    '''
    根据node属性生成latex属性，latex暂时不支持下划线和删除线。
    node属性是一个字典对象
    '''
    attr = ''
    fontupper = ''
    colorlst = []  #自定义颜色列表
    keys = nodeattr.keys()
    #指令参数名称即为字典的key值
    if 'color' in keys:
        colorvalue,flag = __GetColorAttrValue(nodeattr['color'])
        if flag:
            #含有rgb的颜色，需要定义
            #生成唯一的颜色名称
            colorid = 'definecoltext%d' % self.settings.env.new_serialno('coltext=' + colorvalue)
            definecolor = r'\definecolor{'+colorid+'}'+colorvalue+'\n'
            attr +="coltext=" + colorid + ','
            colorlst.append(definecolor)
        else:
            attr +="coltext=" + nodeattr['color'] + ','
    if 'bgcolor' in keys:
        colorvalue,flag = __GetColorAttrValue(nodeattr['bgcolor'])
        if flag:
            #含有rgb的颜色，需要定义
            #生成唯一的颜色名称
            colorid = 'definecolback%d' % self.settings.env.new_serialno('colback=' + colorvalue)
            definecolor = r'\definecolor{'+colorid+'}'+colorvalue+'\n'
            attr +="colback=" + colorid +','
            colorlst.append(definecolor)
        else:
            attr +="colback=" + nodeattr['bgcolor'] + ','
    elif 'isparsedliteral' in keys:
        # isparsedliteral 默认采用自定义shadecolor，shadecolor这个颜色的定义参见conf.json文件
        attr += "colback=shadecolor,colframe=shadecolor,left*=0mm," 
    else:
        attr += "colback=white,colframe=white,left*=0mm," #没有配置背景色，则采用白色背景色，制造没有背景色的假象。否则默认为灰色。
        
    if 'caption' in keys:
        attr += 'title='+nodeattr['caption'] +r',fonttitle=\bfseries,'
        #给title添加背景色，否则title默认为白色，显示不出来
        attr += 'colbacktitle = black!50!blue,'

    if 'isparsedliteral' in keys:
        fontupper += r'\ttfamily' #和sphinx的parsed-literal指令保持一致，默认为等宽字体
    if 'italic' in keys:
        fontupper += r'\itshape'
    if 'bold' in keys:
        fontupper += r'\bfseries'
    if 'fontsize' in keys:
        # 得到latex字体大小
        # 组合下面的接口可以认识的字符串
        attrstr = 'font-size:' + nodeattr['fontsize']
        latexflag = __GetLatexFontSizeFlag(attrstr, 'font-size')
        fontupper += latexflag
    if 'font-size' in keys:
        fontupper += '\\' + nodeattr['font-size']
    elif 'isparsedliteral' in keys:
        fontupper += r'\small'  #sphinx parsed-literal默认字体比正常字体小一号，因此默认用small大小字体
    if len(fontupper) > 0:
        fontupper = 'fontupper=' + fontupper
    
    #print(attr+fontupper)
    return attr+fontupper, colorlst

def __ModifyNodeAttrtoHtmlStyle(nodeattr):
    '''
    根据node属性生成，html标签的style属性。
    node属性是一个字典对象
    '''
    delflag = False #是否有删除标志
    keys = nodeattr.keys()
    #指令参数名称即为字典的key值
    if 'color' in keys:
        nodeattr['style'] = 'color:' + nodeattr['color']
        del nodeattr['color']
    if 'bgcolor' in keys:
        nodeattr['style'] = nodeattr['style'] + ';background-color:' + nodeattr['bgcolor']
        del nodeattr['bgcolor']
    if 'italic' in keys:
        nodeattr['style'] = nodeattr['style'] + ';font-style:italic'
        del nodeattr['italic']
    if 'bold' in keys:
        nodeattr['style'] = nodeattr['style'] + ';font-weight:bold'
        del nodeattr['bold']
    if 'underline' in keys:
        nodeattr['style'] = nodeattr['style'] + ';text-decoration:underline'
        del nodeattr['underline']
    if 'strike' in keys:
        del nodeattr['strike']
    if 'caption' in keys:
        del nodeattr['caption']
    #print(attr)
    return nodeattr

def html_visit_cncolorsection_node(self, node):
    #print('sssssssssssssssssssssssssssssssss')
    #print(node)
    #print(node.attributes)
    #print(node.tagname)
    #print(node.astext())
    #print(type(self))
    #print('ssssssssssssssssssssssssssssssss')
    self.strikeflag = False
    attrs, self.strikeflag = __GetHtmlStyleAttr(node.attributes)
    tag = self.starttag(node, 'div','',CLASS='cncolorsection', **attrs)
    self.body.append(tag)
    if self.strikeflag:
        self.body.append(self.starttag(node, 'del'))

def html_depart_cncolorsection_node(self, node):
    if self.strikeflag:
        self.body.append('</del>')
        self.strikeflag = False
    self.body.append('</div>\n')

def latex_visit_cncolorsection_node(self, node):
    
    if 'istitle' in node.attributes:
        self.body.append(r'\chapter{')
    else:
        self.body.append(r'\section{')

def latex_depart_cncolorsection_node(self, node):
    self.body.append('}')

def html_visit_cncolordirective_node(self, node):
    self.strikeflag = False #是否有删除标志
    attrs,self.strikeflag = __GetHtmlStyleAttr(node.attributes)
    node_keys = node.attributes.keys()
    if 'isparsedliteral' in node_keys:
        #html和sphinx的parsed-literal保持一致的行为
        tag = self.starttag(node, 'pre', '', CLASS='literal-block', **attrs)
    elif ('underline' in node_keys) or ('strike' in node_keys):
        #按行解析的需要用pre标签，不能用div标签否则格式不是预期的格式
        tag = self.starttag(node, 'pre', '', CLASS='cncolorblock', **attrs)
    else:
        tag = self.starttag(node,'div','',CLASS='cncolorblock',**attrs)
    self.body.append(tag)
    
    if self.strikeflag:
        self.body.append(self.starttag(node, 'del'))
    if 'caption' in node.attributes.keys():
        self.body.append("<h4 class='cncolorcaption'>"+node.attributes['caption']+'</h4>\n')
        
    if node.tagname =='cncolorliteral':
        self.body.append(self.starttag(node,"pre"))
    

def html_depart_cncolordirective_node(self, node):
    if self.strikeflag:
        self.body.append('</del>')
        self.strikeflag = False
    if node.tagname == 'cncolorliteral':
        self.body.append('</pre>')
    if 'isparsedliteral' in node.attributes.keys():
        self.body.append('</pre>\n')
    else:
        self.body.append('</div>')

def latex_visit_cncolordirective_node(self, node):
    
    latexattr,colorlst = __GetlatexStyleAttr(self,node.attributes)
    self.body.append('\n')
    for color in colorlst:
        self.body.append(color)
        self.body.append('\n')
    if 'isparsedliteral' in node.attributes.keys():
        if 'spacing' in node.attributes.keys():
            latexpre = r'\begin{spacing}{' + node.attributes['spacing'] + '}\n'+r'\begin{tcolorbox}[arc=0mm,boxrule=-1mm,left=0mm,right=0mm,' + latexattr + ',breakable]\n'
        else:
            latexpre = r'\begin{spacing}{1.3}'+'\n'+r'\begin{tcolorbox}[arc=0mm,boxrule=-1mm,left=0mm,right=0mm,' + latexattr + ',breakable]\n'
    else:
        latexpre = r'\begin{tcolorbox}[arc=0mm,boxrule=0mm,left=0mm,' + latexattr + ',breakable]\n'
    #print(latexpre)
    self.body.append(latexpre)
    if node.tagname == 'cncolorliteral':
        self.body.append(r'\begin{Verbatim}[commandchars=\\\{\}]' + '\n')

def latex_depart_cncolordirective_node(self, node):
    
    #self.body.append('\n')
    if node.tagname == 'cncolorliteral':
        self.body.append('\n'+r'\end{Verbatim}' + '\n')

    self.body.append('\n'+ r'\end{tcolorbox}' + '\n')
    if 'isparsedliteral' in node.attributes.keys():
        self.body.append('\n'+r'\end{spacing}'+'\n')
def html_visit_cncolorline_node(self,node):
    pass
def html_depart_cncolorline_node(self,node):
    #self.body.append('\n')
    pass
    
def latex_visit_cncolorline_node(self,node):
    #print('cncolorline')
    latexattr = ''
    self.linecount = 0
    if 'strike' in node.attributes:
        latexattr = r"\sout{" + latexattr
        self.linecount += 1
    if 'underline' in node.attributes:
        #print("underline")
        latexattr = r"\uline{" + latexattr
        self.linecount += 1
    if self.linecount > 0:
        self.body.append(latexattr)
    
def latex_depart_cncolorline_node(self,node):
    for i in range(self.linecount, 0, -1):
        #print(self.linecount)
        self.body.append('}')
    #self.body.append('\\\\'+'\n')
    self.linecount = 0
def html_visit_cncolortext_node(self,node):
    pass
def html_depart_cncolortext_node(self,node):
    self.body.append('\n')
def latex_visit_cncolortext_node(self,node):
    pass
    
def latex_depart_cncolortext_node(self,node):
    self.body.append('\\\\'+'\n')

def doctree_resolved_process_titlenodes(app, doctree, fromdocname):
    #if fromdocname=='preface/preface':
    #    print(doctree)
    pass
    #以下注释保留，为后续开发提供参考
    '''
    env = app.builder.env
    if not hasattr(env, 'titledict'):
        return
    
    if fromdocname=='index':
        
        #print(doctree)
        for node in doctree.traverse(addnodes.toctree):
            print('=======================')
            toc = app.env.resolve_toctree(fromdocname, app.builder, node)
            print(type(toc))
            print(toc)
            print('=======================')
            for nodechild in toc.children:
                print('----------------------')
                print(nodechild.astext())
                print('----------------------')
                if nodechild.astext() in env.titledict.keys():
                    newnode = env.titledict[nodechild.astext]
                    nodechild.replace_self(newnode)
                    node.remove(nodechild)
    '''
def __modifyNodeforlatexraw(node):
    #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    #print(node.astext())  # 或者其他处理逻辑
    #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    text = node.astext()
    if "-" in text:
        #print(node.children)
        node.children = []  # 先清空老节点内容
        # 对文本进行拆分
        textlst = text.split("-")
        #print(textlst)
        for i in range(0, len(textlst)):
            textnode = nodes.Text(textlst[i], textlst[i])
            node.append(textnode)
            if i < len(textlst) - 1:
                rawnode = nodes.raw('', r"{\PYGZhy{}}", format='latex')
                node.append(rawnode)
                
def __addpygzhychildnode(doctree):
    """
    如果文本中含有“-”或者“——”或者“—”，则增加latex子节点
    """
    def traverse(node):
        if isinstance(node, cnautowrapnode):
            __modifyNodeforlatexraw(node)
            #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            #print(node.astext())  # 或者其他处理逻辑
            #print('~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            #text = node.astext()
            #if "-" in text:
            #    print(node.children)
            #    node.children=[] #先清空老节点内容
            #    #对文本进行拆分
            #    textlst = text.split("-")
            #    print(textlst)
            #    for i in range(0,len(textlst)):
            #        textnode = nodes.Text(textlst[i],textlst[i])
            #        node.append(textnode)
            #        if i < len(textlst)-1:
            #           rawnode = nodes.raw('', r"{\PYGZhy{}}", format='latex')
            #           node.append(rawnode)
                    
                #newtext = newtext.replace("-",r"{\PYGZhy{}}")
                #newtext = newtext.replace("_",r"\_")
                #rawnode = nodes.raw('', newtext, format='latex')
                #node.append(rawnode) #再添加新的子节点
                
        for child in node.children:
            traverse(child)

    traverse(doctree)
    #print(doctree)
    
def doctree_read_modify_titlenodes(app, doctree):
    #print(doctree)
    #docname = app.env.docname
    #print(docname)
    #print(doctree)
    if app.builder.name=='latex' or app.builder.name=='latexpdf':
        chaptercount = 0
        for node in doctree.children:
            if node.tagname=='chapter':
                chaptercount += 1
            if node.tagname=='section':
                for nodechild in node.children:
                    if nodechild.tagname=='cncolorsection':
                        for childrentitle in nodechild.children:
                            if childrentitle.tagname =='title':
                                newchildren = []
                                for childnode in childrentitle.children:
                                    newchildren.append(childnode)
                                childrentitle.replace_self(newchildren)
            elif node.tagname=='cncolorsection':
                if chaptercount == 0:
                    node.attributes['istitle']=True
                for childrentitle in node.children:
                    if childrentitle.tagname =='title':
                        newchildren = []
                        for childnode in childrentitle.children:
                            newchildren.append(childnode)
                        childrentitle.replace_self(newchildren)
        '''
        如果是latex需要把section中间的title节点删除，否则生成的tex文件章节标题中包含\part指令，
        在latex标题中包含\part指令将导致无法编译通过
        以下代码只支持sphinx 4.x以上版本，否则findall函数找不到。
        '''
        #for node in doctree.findall(cncolorsection):
        #    for childrentitle in node.children:
        #        if childrentitle.tagname =='title':
        #            newchildren = []
        #            for childnode in childrentitle.children:
        #                newchildren.append(childnode)
        #            childrentitle.replace_self(newchildren)
    #print(doctree)
    #pass
    #以下注释保留，为后续开发提供参考
    '''
    #print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
    #print(doctree)
    #print(type(doctree))
    #print("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
    #判断是否有cncolortitle节点，有该节点取出解析后的title，替换cncolorsection节点内的title，然后删除cncolortitle节点。
    env = app.builder.env
    if not hasattr(env, 'titledict'):
        return
    
    for node in doctree.findall(cncolortitle):
        #生成新的title节点
        newnodes = nodes.title()
        for childrentitle in node.children:
            newnodes +=childrentitle
        print(newnodes)
 
        index = node.parent.index(node)
        sectionnode = node.parent[index+1]
        sectionstr = sectionnode.astext()
        #if env.titledict:
        env.titledict[sectionstr] = newnodes.deepcopy()
        print(newnodes.deepcopy())
        print(type(newnodes.deepcopy()))
        print(env.titledict)
        #对老节点进行替换删除
        for sectionchildren in sectionnode.children:
            if sectionchildren.tagname == 'title':
                sectionchildren.replace_self(newnodes)
        node.parent.remove(node)
    '''
def __ModfiyTitleh0toh1(context):
    #修改标题h0为h1，因为cncolorsection为自定义node，当标题在一级时，级别为h0,应该为h1
    searchstr = '<div class="cncolorsection"[\s\S]+?(<h0>([\s\S]+?)</h0>)[\s\S]+?</div>'
    matchobj = re.search(searchstr,context,re.I|re.U|re.M)
    if matchobj:
        #group()和group(0)相同
        match = matchobj.group()
        h0str = matchobj.group(1)
        index = match.find(h0str)
        pre = context[0:matchobj.start()+index]
        left = context[matchobj.start()+index+len(h0str):len(context)]
        newcontent = pre+"<h1>"+matchobj.group(2)+"</h1>"+left
        __ModfiyTitleh0toh1(newcontent)
        return newcontent
    else:
        return context
        
def html_page_context_handle(app, pagename, templatename, context, doctree):
    if hasattr(app.env,"htmlfile"):
        if pagename in app.env.htmlfile:
            #修改body内容
            context['body']= __ModfiyTitleh0toh1(context['body'])

def modfiletimestamp(filename):
    '''
    强制修改文件时间戳，为了make html后，强制make latex强制刷新文件
    '''
    #stinfo = os.stat(filename)
    #print(filename)
    #print(stinfo)
    current_time = time.time()
    os.utime(filename, (current_time, current_time))
    #stinfo = os.stat(filename)
    #print(stinfo)
def env_get_outdated_handler(app,env,docnames):
    #实现增量编译，将带cncolor指令的文件重新编译，否则latex可能无法生成
    if hasattr(env, 'sectionfile'):
        for file in env.sectionfile:
            if file not in docnames:
                docnames.append(file)
    if hasattr(env, 'autowrapfile'):
        #print('----outdated enter----------')
        for file in env.autowrapfile:
            if file not in docnames:
                #print(file)
                docnames.append(file)
    #if hasattr(env, 'cncolorfile'):
    #    print('----outdated enter----------')
    #    print(env.cncolorfile)
    #    print(docnames)
    #    for file in env.cncolorfile:
    #        if file not in docnames:
    #            print(file)
    #            docnames.append(file)
    #for file in docnames:
    #    modfiletimestamp(env.doc2path(file))
        
def env_updated_handler(app,env):
    #print('-----updated------------')
    #print(env.all_docs)
    #print('-----updated------------')
    pass
    
def config_inited_handler(app,config):
    #加载ulem包，否则latex不支持下划线和删除线
    if 'extrapackages' not in app.config.latex_elements.keys():
        app.config.latex_elements['extrapackages'] = r'\usepackage{seqsplit}'+'\n'
        if os.path.exists(ulempath):
            app.config.latex_elements['extrapackages'] += r'\usepackage[normalem]{ulem}' + '\n'
    else:
        if r'\usepackage{seqsplit}' not in app.config.latex_elements['extrapackages']:
            app.config.latex_elements['extrapackages'] += r'\usepackage{seqsplit}'+'\n'
        if os.path.exists(ulempath) and \
                (r'\usepackage[normalem]{ulem}' not in app.config.latex_elements['extrapackages']):
            app.config.latex_elements['extrapackages'] += r'\usepackage[normalem]{ulem}'+'\n'
            
def cd_build_finished_handler(app, exception):
    if os.path.exists(ulempath) and \
            (app.builder.name == "latex" or \
        app.builder.name == "latexpdf"):
        destpath = os.path.join(app.outdir,'ulem.sty')
        shutil.copy(ulempath, destpath)

#得到当前路径，用于判断需要的ulem.sty文件是否存在，如果存在，直接copy到latex目录下。
curpath = os.path.dirname(os.path.realpath(__file__))
ulempath = os.path.join(curpath,'ulem.sty')

def setup(app):
    app.add_directive('cntoctree', TocTreeFilt)
    app.add_directive('cnonly',CNOnly)
    app.connect('doctree-resolved', doctree_resolved_process_titlenodes)
    app.connect('doctree-read', doctree_read_modify_titlenodes)
    app.connect('html-page-context',html_page_context_handle)
    app.connect('env-before-read-docs',env_get_outdated_handler)
    app.connect('env-updated', env_updated_handler)
    app.connect('config-inited', config_inited_handler)
    app.connect('build-finished', cd_build_finished_handler)

    #-----------cncolor directive----------------------------------------------------------------
    app.add_directive('cncolor',cnColorDirectivecls)
    app.add_directive('cnparsed-literal', cnparsedliteralDirectivecls)
    app.add_node(cncolorliteral,
                 html=(html_visit_cncolordirective_node, html_depart_cncolordirective_node),
                 latex=(latex_visit_cncolordirective_node, latex_depart_cncolordirective_node))
    app.add_node(cncolordirective,
                 html=(html_visit_cncolordirective_node, html_depart_cncolordirective_node),
                 latex=(latex_visit_cncolordirective_node, latex_depart_cncolordirective_node))
    app.add_node(cncolorsection,
                 html=(html_visit_cncolorsection_node, html_depart_cncolorsection_node),
                 latex=(latex_visit_cncolorsection_node, latex_depart_cncolorsection_node))
    app.add_node(cncolorline,
                 html=(html_visit_cncolorline_node, html_depart_cncolorline_node),
                 latex=(latex_visit_cncolorline_node, latex_depart_cncolorline_node))
    app.add_node(cncolortext,
                 html=(html_visit_cncolortext_node, html_depart_cncolortext_node),
                 latex=(latex_visit_cncolortext_node, latex_depart_cncolortext_node))
    #------------cncolor role--------------------------------------------------------------------
    app.add_node(cncolorrolenode,
                 html=(html_visit_cncolor_node, html_depart_cncolor_node),
                 latex=(latex_visit_cncolor_node, latex_depart_cncolor_node))
    app.add_role('cncolor',cncolor_role)
    
    app.add_node(cnautowrapnode,
                 html=(html_visit_autowrap_node, html_depart_autowrap_node),  
                 latex=(latex_visit_autowrap_node, latex_depart_autowrap_node))
    app.add_role('autowrap',autowrap_role)
    
    #增加并行能力，否则，在其它地方再定义事件处理函数会有告警。
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
