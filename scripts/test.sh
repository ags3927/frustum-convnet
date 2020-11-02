declare -a x=(1 2 3 4 5)
declare -a y=(1 2 3 4 5)

for i in "${x[@]}"
do
    for j in "${y[@]}"
    do
        echo $i, $j
    done
done