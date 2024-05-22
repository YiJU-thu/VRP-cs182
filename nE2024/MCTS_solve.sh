#!/bin/bash
# author: 
rm -r ./MCTS
g++ ./utils/MCTS.cpp -o ./MCTS -std=c++20 -pthread

threads=$1
Temp_City_Num=$2
tsp="./tsp20_test_concorde.txt"
for ((i=0;i<$threads;i++));do
{
	./test $i $tsp $Temp_City_Num
}&
done
wait
echo "Done."