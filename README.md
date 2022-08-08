# KanjiRecognizer
Using the ETL character database to develop a CNN that will recognize the top 3000 kanji characters. Then to use that model to make a web app that allows
users to handwrite unknown kanji and search it up. 

*Basic Information*

Kanji are essentially a collection of Chinese characters that were imported into Japan and integrated into their writing system. Although there are 
over 10,000 kanji in Japanese, most educated natives know and see only about 2500-3000 in their daily lives. 

*Technologies*

Using Keras (python framework) with a Tensorflow base. 

*Database*

In this project, I used the ETL Character database, which contains thousands of images of handwritten Kanji from individuals around Japan. Previously, 
machine learning work has been done using the ETL database (specifically ETL-1 and ETL-8), which contain the katakana alphabet, the hiragana alphabet and 
879 kanji respectively. However the goal of this project, is to create a convolutional neural network that can recognize all the "daily-use" kanji; thus,
I will be using the ETL-9 database.

ETL-9 contains 50 files, with each containing 4 datasets (4 different people), each in which a set of 3036 kanji are written in that individual's
handwritting. However, these files are distributed as Unix executables, thus actually figuring out how to unnpack them into a format that tensorflow can 
read was the first major challenge. After searching the internet and analyzing code of others that have used this database, I found a method that works to
unpack these files (credits to github users melodyfs and yukoba). 

*Building the model*
Initially, I worked with the 879 character database, as it is much smaller, making preprocessing and epoch run times much faster. Ultimately, using ideas 
I found from Charlie Tsai's paper, "Recognizing Handwritten Japanese Characters Using Deep Convolutional Neural Networks", I was able to create a model that
classifies those 879 to 96% accuracy. Now the goal of this project is to apply that same model to the 3000+ character database of ETL-9.

*Current Progress* (8/8/22)
Currently, I have coded up the model for ETL-9, however I'm facing a lack of RAM memory (both on my personal machine, as well as Google Colab). Thus, 
it might be necessary to cut down on the size of the data, and take only 2000 of the 3000 kanji.

*References*
http://etlcdb.db.aist.go.jp
https://github.com/melodyfs/Build-OCR
https://github.com/yukoba/CnnJapaneseCharacter
http://cs231n.stanford.edu/reports/2016/pdfs/262_Report.pdf
