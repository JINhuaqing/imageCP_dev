source="/home/hujin/jin/MyResearch/imageCP_dev/"  
dest="Wynton-data:/wynton/home/jianglab/hjin/MyResearch/imageCP_dev/"
options="-avzh"
exs="--exclude 'results/backups*/'"

echo $source
echo $dest

rsync -avhz  --exclude 'data/' $source $dest
#rsync $options $exs $source $dest

if [ $? -eq 0 ]; then
  echo "同步成功！"
else
  echo "同步失败！"
fi
