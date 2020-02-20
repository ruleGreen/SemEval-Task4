在Task A这个任务上直接运行 sh a_cross.sh 就可以
但是 a_cross.sh 里面定义了不同的model，每次运行需要改一下里面的参数

里面的参数共有 data task k rand model 这五个
第一个代表data, 我们这里直接定2就可以了
第一个代表task, 这里是A
第三个代表几折验证, 这里默认是10
第四个代表把数据集划分为训练集和测试集，2代表20%的作为测试集
第五个代表不同的model, 在TaskA里面可以选1 2 3 6 9

然后还需要改下试验参数
有cuda_device epoch等
如果在gpu跑的话, epoch可能设为100比较好

然后再 sh a_cross.sh 即可

model运行结束后，找一个效果最好的model去进行预测

然后在当前目录下运行这条命令
allennlp predict --output-file datasets/TestingData/a_res.jsonl --use-dataset-reader --dataset-reader-choice train --predictor a-classifier --include-package my_project tmp/semeval4a_model3_output/model.tar.gz datasets/TestingData/subtaskA_test_data.csv

然后进入datasets/TestingData
运行一下process.py文件就能得到预测结果了