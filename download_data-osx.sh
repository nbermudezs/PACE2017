SOURCE='http://lbai5.web.engr.illinois.edu/data/'
mkdir data
mkdir data/gowalla
cd data
curl $SOURCE/inter_small.pkl -o inter_small.pkl
curl $SOURCE/testdata_small.pkl -o testdata_small.pkl
curl $SOURCE/traindata_small.pkl -o traindata_small.pkl
cd gowalla
curl $SOURCE/gowalla/visited_spots.txt -o visited_spots.txt
curl $SOURCE/gowalla/spot_category.txt -o spot_category.txt
curl $SOURCE/gowalla/user_network.txt -o user_network.txt
curl $SOURCE/gowalla/visited_spots.txt -o visited_spots.txt
curl $SOURCE/gowalla/Readme.txt -o Readme.txt


