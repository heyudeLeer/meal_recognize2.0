#!/bin/bash

root=`pwd`
imglib=$1
validation_split=$2
if [ ! -n "$2" ]; then
    validation_split = 0.1
fi
if [ $# -ne 2 ]; then
	echo
	echo "用法：split.sh 图片原始库 验证集百分比"
	echo 'get train validation lib on path of split.sh'
	exit
fi
cd $imglib
imglibAbsPath=`pwd`
cd $root

dir="train"
if [  -d $dir ]; then
   rm -rf $dir
fi
mkdir $dir

dir="validation"
if [  -d $dir ]; then
   rm -rf $dir
fi
mkdir $dir
   
cd $imglib
ls > $root"/classes"

for class in ` cat $root"/classes" `
do      
	mkdir -p $root'/train/'$class
	mkdir -p $root'/validation/'$class
	classPath=$imglibAbsPath'/'$class
	cd $classPath
        echo "coming "$classPath
        ls > $root"/files"
        nums=`cat $root'/files' | wc -l` 
	echo "files nums "$nums
	fra=$(echo "$nums*$validation_split"|bc)
	fra=$(echo ${fra%.*}) #取整
	shuf -n$fra $root'/files' > $root'/randomfiles'
	cp -v $imglibAbsPath'/'$class"/"* $root"/train/"$class"/"
        for file in `cat $root"/randomfiles"`
        do
                if [ $? -ne 0 ];then
			break
		fi
		mv -v  $root'/train/'$class"/"$file $root"/validation/"$class"/"
        done
	
	if [ $? -ne 0 ];then
                        break
        fi

done

rm $root'/classes'
rm $root'/files'
rm $root'/randomfiles'
