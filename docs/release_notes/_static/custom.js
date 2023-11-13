function insertFloatbtn(){
var targetElement = document.getElementsByClassName("document")[0];

//接着，我们可以创建一个新的HTML元素
var newElement = document.createElement("div");
newElement.className="dldpdf-btn";
newElement.textContent = "下载PDF";
//console.log(newElement)

//然后，我们将新的HTML元素添加到目标位置
targetElement.insertBefore(newElement,targetElement.firstElementChild);
//console.log(targetElement)
}

window.addEventListener('load', function() {

insertFloatbtn()

var floatingBtn = document.querySelector('.dldpdf-btn');
var lastScrollTop = 0;

floatingBtn.addEventListener("click", function(){ 
//得到pdf路径
var scripts = document.getElementsByTagName("script");
var pdfPathName="";
var relativePath="";

for (var i = 0; i < scripts.length; i++ ){
    var path = scripts[i].getAttribute("data-url_root");
    if(path!=null){
        relativePath = path;
        //console.log(i);
        //console.log(relativePath);
    }
    var filename = scripts[i].getAttribute("pdfname");
    if(filename!=null){
        pdfPathName = filename;
        //console.log(i);
        //console.log(pdfPathName);
    }
    if(relativePath!="" && pdfPathName!=""){
        break;
    }
    
    //var pos = file.indexOf("_static"); //判断路径中是否存在_static目录,根据_static这个目录查找pdf所在目录
    //if(pos!=-1){
    //    //查找相对目录
    //    relativepath = file.substring(0,pos);
    //    console.log(relativepath);
    //    pdfPathName = relativepath + "Dove-CNMPS-Developer-Guide-CN.pdf";
    //    break;
    //}
}
pdfPathName = relativePath + pdfPathName;
if(pdfPathName!=""){
    window.open(pdfPathName); 
}else{
    alert("No PDF File!");
}

});

window.addEventListener('scroll', function() {
var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
if (scrollTop >lastScrollTop) {
//滚动向下
floatingBtn.style.transform = 'translateY(0px)';
}  else {
//滚动向上
floatingBtn.style.transform = 'translateY(0px)';
}
lastScrollTop = scrollTop;
});
 });

document.addEventListener("mouseup", function(){
    var selectedText = window.getSelection().toString();
    console.log(selectedText);
});