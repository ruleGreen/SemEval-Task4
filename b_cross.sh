# this is for k fold cross validation
# author Carrey Wang

if [ -e datasets/k_fold.py ]; then
  # data = input("please input the datasets 1 -> trial data 2 -> more data")
  # task = input("please input the task(A,B,C): ")
  # k = int(input("please input the k value of k fold cross validation: "))
  # rand = int(input("please input the random seed(integer): "))
  # model = int(input("please input the model you want to use: 1 -> single glove 2 -> single elmo 3 -> ensemble, 6 -> single bert"))
  # model = int(input("please input the model you want to use: 4 -> seq2seq + self-attention 5 -> glove"))
  # model = 9 # this is for test model 3
  python3 datasets/k_fold.py 1 B 10 2 2
else
  echo "Sorry, you do not have k_fold.py file at the datasets directory."
  exit 1
fi
