在Task A这个任务上直接运行 sh a_cross.sh 就可以
但是 a_cross.sh 里面定义了不同的model，每次运行需要改一下里面的参数

里面的参数共有 data task k rand model 这五个
第一个代表data, 我们这里直接定2就可以了
第一个代表task, 这里是A
第三个代表几折验证, 这里默认是10
第四个代表把数据集划分为训练集和测试集，2代表20%的作为测试集
第五个代表不同的model, 在TaskA里面可以选1 2 3 6 9

