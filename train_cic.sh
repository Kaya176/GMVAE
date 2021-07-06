
size=2
num_classes=2
batch_size=8192

w_cat=1
w_gauss=1
w_rec=1
# rec_type="mse"
rec_type="bce"
python train_cic.py --epochs 100 --gaussian_size $size --num_classes $num_classes --batch_size $batch_size --learning_rate 1e-3 --w_gauss $w_gauss --w_categ $w_cat --w_rec $w_rec --rec_type $rec_type