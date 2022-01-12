limit="$3"
count=0

for file in "$1"/*
do
    printf "Executing: ./Segmentator.exe on %s\n" "$file"
    ./Segmentator.exe "-i $file -o $2 -mt hough --size 250 --mode segmentation"
    ((count=count+1))
    if [ "$count" -eq "$limit" ]; then
        break
    fi
    printf "\n\n"
done
