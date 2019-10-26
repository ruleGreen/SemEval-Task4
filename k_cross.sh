# this is for k fold cross validation
# author Carrey Wang

if [ -e datasets/k_fold.py ]; then
  python3 datasets/k_fold.py
else
  echo "Sorry, you do not have k_fold.py file at the datasets directory."
  exit 1
fi