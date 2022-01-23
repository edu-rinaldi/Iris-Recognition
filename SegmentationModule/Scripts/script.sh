limit="$3"
count=0

for file in "$1"/*
do
    printf "Executing: ./isisApp.exe %s\n" "$file"
    ./isisApp.exe "-s" "$file","$2"
    sleep 2
    ((count=count+1))
    if [ "$count" -eq "$limit" ]; then
        break
    fi
    printf "\n\n"
done
